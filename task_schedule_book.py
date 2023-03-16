import base64
import copy
import socket
from _thread import start_new_thread
from threading import Lock
import cv2
import os
import time
from pathlib import Path
import argparse
import subprocess
import shutil
import json
from multiprocessing.pool import ThreadPool
import numpy as np
import psutil
from common.networking import send_msg, recv_msg
from common.logger import logger
from diff import diff_detector, diff_houchao
from _thread import start_new_thread
from common.opt import solver_long_tail, task_assignment, fit, quantize, config_v_w

parser = argparse.ArgumentParser(description='Please specify the directory of data set')

parser.add_argument('--data_dir', type=str, default='book',
                    help="the directory which contains the pictures set.")
parser.add_argument('--data_collect_dir', type=str, default='data/collect',
                    help="the directory which contains the pictures set.")
parser.add_argument('--output_dir', type=str, default='data/test_results',
                    help="the directory which contains the final results.")
parser.add_argument('--best_dir', type=str, default='data/gt',
                    help="the directory which contains the final results.")
parser.add_argument('--parameter', type=str, default='data/parameter/sfm_data_walk.json',  #
                    help="th directory which contains the pictures set.")
parser.add_argument('--reconstructor', type=str, default='mmm_parl.py',  #
                    help="the directory which contains the reconstructor python script.")
parser.add_argument('--resolution', type=float, default=1.0,
                    help="the directory which contains the reconstructor python script.")
parser.add_argument('--start', type=int, default=946,
                    help="the directory which contains the reconstructor python script.")

args = parser.parse_args()

Path(args.data_collect_dir).mkdir(parents=True, exist_ok=True)
Path(args.output_dir).mkdir(parents=True, exist_ok=True)

m = Lock()


class Scheduler:
    def __init__(self, th=0.06, quality=0.7, start=400, end=2200):
        self.done = False
        self.most_recent_3d_id = None
        self.s = None
        self.servers = []
        self.start = start
        self.end = end
        ##############################
        self.in_eva = False
        ##############################
        self.init = True
        ##############################
        self.quality_req = quality
        self.th = th
        self.resolution = 1.0
        self.most_recent_3d_id = 0
        self.most_recent_3d_model_id = start
        self.reconstructed_ids = []
        self.done = False
        self.online_start_timestamp, self.online_end_timestamp = start, start
        self.delivery_task = False
        ##############################
        self.timestamp = 0
        self.long_term_f = []
        self.tasks = []
        self.total_task_q = 0
        self.total_task = 0
        self.tps = []
        self.fs = []
        self.rs = []
        self.ths = []
        self.x1s = []
        self.x2s = []
        self.feqs = []
        self.predicted_feqs = []
        self.dd_speed = 0
        self.quality_queue = 0
        self.exc = True
        self.converged = False
        self.DD_stop = False
        ##############################
        self.gama = 0.9
        self.obs_d_frequency = {}
        self.obs_d_f_score = {}
        self.diff_th = None
        self.f1_d = None
        self.f1_error = []
        self.f1_error_long_term = []
        self.s_error = []
        self.s_error_long_term = []
        self.cts = []
        self.intervals = [2]
        self.t = 0
        ##############################
        # server computation time
        self.server_paras = [
            (0.1016771987765127, 2.609735151185393, 7.36594987756804),
            (0.06157433135916346, 3.549801161121459, 4.314966329861685)
        ]

        # f1-res
        self.f1_res = (1.5198856981333275, 5.26164086391182, 0.9491640358096844)
        #####################
        self.interval = 2
        self.init_tp = 5
        self.dd_speed = 5

        # self.window = int(self.init_tp * self.dd_speed)
        self.pre_quality = self.quality_req
        #####################
        self.v = 20
        self.window = 50

    def config_v_w(self, total_task, total_quality, max_task, quality_q, server_paras, diff_th, f1_res, f1_d):
        print(total_task, total_quality, max_task, quality_q, self.quality_req)
        opt_v = None
        opt_w = None
        for v in [1, 3, 5, 10]:
            times = []
            f = []
            ws = [25, 50, 75, 100, 150]
            opt = None
            min_time = None
            max_q = None
            for window in ws:
                finished_task = 0
                finished_quality = 0
                finished_q = quality_q
                total_time = 0
                for i in range(int(2000 / window)):
                    f1, feq, x1, x2, _, _, _, z = solver_long_tail(v, self.quality_req, finished_q, window,
                                                                   server_paras, diff_th,
                                                                   f1_res, f1_d)
                    finished_quality += f1 * window
                    finished_task += window
                    total_time += max(z, window / self.dd_speed)
                    finished_q = finished_q + (self.quality_req - f1) * window

                times.append(total_time)
                f.append(finished_quality / finished_task)
                if opt is None or total_time < min_time:
                    opt = window
                    min_time = total_time
                    max_q = round(finished_quality / finished_task, 4)

            print(v, times)
            print(v, f)
            print(opt, min_time, max_q)
            print("#######################")
            if self.quality_req - max_q > 0.005:
                break
            else:
                opt_v = v
                opt_w = opt
        print("Remaining tasks =", max_task)
        print("Current queue =", quality_q)
        print(f"current v/w = {self.v}/{self.window}", "opt_v, opt_w = ", opt_v, opt_w)
        return opt_v, opt_w

    def profile(self):
        profile_time_start = time.time()

        items = [
            (0.015, self.timestamp, self.window, self.servers[1]["socket"]),
            (0.10, self.timestamp, self.window, self.servers[0]["socket"])
        ]

        #with ThreadPool(len(items)) as poc:
        #   results = poc.map(self.profile_task, items)

        items = [
            (0.035, self.timestamp, self.window, self.servers[0]["socket"]),
            (0.06, self.timestamp, self.window, self.servers[1]["socket"])
        ]

        #with ThreadPool(len(items)) as poc:
        #    results.append(poc.map(self.profile_task, items))

        results = [(1.0, 1.0), (0.04, 0.8427), [(0.6, 0.9595), (0.24, 0.9084)]]

        print("results=", results)
        self.t += 1
        print(logger() + f"profile finish in {round(time.time() - profile_time_start, 4)}, results={results}")

        s1, f1 = results[0]
        s2, f2 = results[1]
        s3, f3 = results[2][0]
        s4, f4 = results[2][1]

        self.obs_d_frequency = {
            0.015: s1,
            0.035: s3,
            0.06: s4,
            0.10: s2
        }
        self.obs_d_f_score = {
            0.015: f1,
            0.035: f3,
            0.06: f4,
            0.10: f2
        }

        print(self.obs_d_frequency)
        print(self.obs_d_f_score)
        print("###################################")
        self.diff_th = self.update_diff_th()
        self.f1_d = self.update_f1_d()
        print("###################################")

    @staticmethod
    def profile_task(para):
        th, start, window, target = para
        msg = {"th": th, "start": start, "window": window, "type": "profile"}
        encoding = json.dumps(msg).encode("utf-8")
        send_msg(target, encoding)
        while True:
            data = recv_msg(target)
            info = json.loads(str(data.decode('utf-8')))
            return info["s"], info["f"]

    def update_diff_th(self):
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        obs = []
        x = []
        for key in sorted(self.obs_d_frequency):
            obs.append(round(self.obs_d_frequency[key], 2))
            x.append(key)
        print("---d", x)
        print("---frequency", obs)
        print("###################################")
        while len(obs) > 3:
            changed = False
            for i in range(1, len(obs)):
                if obs[i] > obs[i - 1]:
                    obs.pop(i)
                    x.pop(i)
                    changed = True
                    break
            if not changed:
                break
            else:
                pass
                # print("---frequency", obs)
        print("---d", x)
        print("---frequency", obs)
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        return fit(obs, x=x)

    def update_f1_d(self):
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        obs = []
        x = []
        for key in sorted(self.obs_d_f_score):
            obs.append(round(self.obs_d_f_score[key], 2))
            x.append(key)
        print("---d", x)
        print("---f-score", obs)
        print("#################################")
        while len(obs) > 3:
            changed = False
            for i in range(1, len(obs)):
                if obs[i] > obs[i - 1]:
                    obs.pop(i)
                    x.pop(i)
                    changed = True
                    break
            if not changed:
                break
            else:
                pass
                # print("---f-score", obs)
        print("---d", x)
        print("---f-score", obs)
        print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        return fit(obs, x=x)

    def save_d_frequency(self, total_task, th):
        frequency = round(len(self.reconstructed_ids) / total_task, 4)
        a_ = quantize(round(th, 2))
        if a_ not in self.obs_d_frequency:
            self.obs_d_frequency[a_] = frequency
        else:
            self.obs_d_frequency[a_] = self.obs_d_frequency[a_] * (1 - self.gama) + frequency * self.gama

    # a = A * F, so A = a/F
    def save_d_f_score(self, score, resolution, th):
        a, b, c = self.f1_res
        raw_score = score
        score = score / (c - a * np.exp(-b * resolution))
        a_ = quantize(th)
        if a_ not in self.obs_d_f_score:
            self.obs_d_f_score[a_] = score
        else:
            self.obs_d_f_score[a_] = self.obs_d_f_score[a_] * (1 - self.gama) + score * self.gama
        print(
            f"add resolution={resolution}, th={th}, r_score={raw_score}, f_score={score}, r_score={(c - a * np.exp(-b * resolution))}")

    def listen(self, port=8001):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind(("", port))
        self.s.listen(5)
        print(logger() + "start to listening:{} -> server......".format(port))
        while True:
            server_socket, server_info = self.s.accept()
            ip, p = server_info
            print(logger() + "server connected, start to communicate with server [{}]".format(ip))
            # send_msg(server_socket, json.dumps({"status": "ok"}).encode("utf-8"))
            self.servers.append({"socket": server_socket, "ip": ip})
            if len(self.servers) == 2:
                break

        self.timestamp = self.start
        self.profile()
        self.p1(self.init_tp, self.window)
        self.task_selection(self.start, self.end)

    def schedule(self, paras):
        ip, ids, res = paras
        start_t = time.time()
        for server in self.servers:
            if server["ip"] == ip:
                target = server["socket"]
                msg = {"ids": ids, "res": res, "type": "reconstruction"}
                encoding = json.dumps(msg).encode("utf-8")
                send_msg(target, encoding)
        while True:
            data = recv_msg(target)
            info = json.loads(str(data.decode('utf-8')))
            for i in range(len(ids)):
                model = info["data"][i]
                Path(os.path.join(args.output_dir, str(ids[i]) + "_output")).mkdir(parents=True, exist_ok=True)
                path = os.path.join(args.output_dir, str(ids[i]) + "_output", "scene_dense.ply")
                with open(path, 'wb') as file:
                    file.write(base64.b64decode(model))
            break
        print(
            logger() + f"ip {ip} is done in {info['time']}/+{round(time.time() - start_t, 4)}, networking={round(time.time() - start_t - info['time'], 4)}")
        return info["status"]

    def optimization(self):
        pass

    def p2(self, th, resolution):
        x1, x2, pre_tp_length = task_assignment(len(self.reconstructed_ids), self.server_paras, resolution)

        pre_tp_length = pre_tp_length + max(x1, x2) * 0.3

        ids = [[], []]

        if len(self.reconstructed_ids) > 0:
            if x1 > 0:
                for i in range(x1):
                    ids[0].append(self.reconstructed_ids[i])
            if x2 > 0:
                for i in range(x1, x1 + x2):
                    ids[1].append(self.reconstructed_ids[i])

        assignment = {
            '146.95.252.28': ids[0],
            '146.95.252.25': ids[1]
        }

        self.x1s.append(len(ids[0]))
        self.x2s.append(len(ids[1]))

        if len(self.reconstructed_ids) > 0:
            self.feqs.append(
                round(len(self.reconstructed_ids) * 1.0 / (self.online_end_timestamp - self.online_start_timestamp), 4))
        else:
            self.feqs.append(0)

        self.s_error[-1] = np.abs(self.feqs[-1] - self.s_error[-1])
        self.s_error_long_term.append(round(np.average(self.s_error), 4))

        self.delivery_task = False

        start_new_thread(self.batch_processing, (self.online_start_timestamp,
                                                 self.online_end_timestamp,
                                                 copy.deepcopy(self.reconstructed_ids),
                                                 assignment,
                                                 resolution,
                                                 th))

        """
        self.batch_processing(self.online_start_timestamp,
                                                 self.online_end_timestamp,
                                                 copy.deepcopy(self.reconstructed_ids),
                                                 assignment,
                                                 resolution,
                                                 th)
        """
        print(
            logger() + f"resolution = {resolution}, pred_tp_length = {pre_tp_length}, pred_next_tp_tasks = {min(self.window, int(pre_tp_length * self.dd_speed))}")
        return pre_tp_length

    def p1(self, pre_tp_length, pre_number_of_tasks):
        # predict queue
        predict_quality = max(0, (self.quality_req - self.pre_quality) * pre_number_of_tasks)
        # self.window = pre_tp_length * self.dd_speed #Arb.
        print(self.v, self.quality_req, self.quality_queue + predict_quality,
              int(self.window),
              self.server_paras,
              self.diff_th,
              self.f1_res,
              self.f1_d)

        f1, feq, x1, x2, self.th, self.resolution, _, _ = solver_long_tail(self.v, self.quality_req,
                                                                           round(self.quality_queue + predict_quality, 1),
                                                                           # self.quality_queue,
                                                                           int(self.window),
                                                                           self.server_paras,
                                                                           self.diff_th,
                                                                           self.f1_res,
                                                                           self.f1_d)

        self.f1_error.append(f1)
        self.predicted_feqs.append(round(feq, 4))
        self.s_error.append(round((x1 + x2) * 1.0 / min(self.window, int(pre_tp_length * self.dd_speed)), 4))
        self.rs.append(self.resolution)
        self.ths.append(self.th)
        print(logger() + f"difference threshold = {self.th}, resolution = {self.resolution}")
        self.pre_quality = f1

    def task_selection(self, start_timestamp, end_timestamp):
        print(logger() + f"start to select tasks from {start_timestamp} to {end_timestamp}")
        start_t = time.time()
        ####################
        self.reconstructed_ids.append(start_timestamp)
        self.most_recent_3d_id = start_timestamp
        self.timestamp = start_timestamp
        ####################
        read = 0
        while True:
            ############################
            while self.in_eva:
                time.sleep(0.1)
            ############################

            # and self.timestamp <= self.end and self.timestamp - self.online_start_timestamp <= self.window  fixed

            # and self.timestamp - self.online_start_timestamp <= self.window
            if len(self.reconstructed_ids) < 20 and self.timestamp <= self.end and self.timestamp - self.online_start_timestamp <= self.window:
                str_timestamp = str(self.timestamp)
                if read % 10 == 0:
                    print(logger() + f"dd timestamp = {self.timestamp}")

                dirs = ["0/Images", "1/Images", "2/Images",
                        "3/Images",
                        "4/Images"]

                img_file_name = str_timestamp + ".jpg"

                src = []
                dest = []
                for image_dir in dirs:
                    src.append(os.path.join('book', image_dir, str(self.most_recent_3d_id) + ".jpg"))
                    dest.append(os.path.join('book', image_dir, img_file_name))

                diffs = diff_houchao.diff_2d(src, dest)
                avg_diff = round(np.average(diffs), 5)

                if (
                        avg_diff >= self.th or self.most_recent_3d_id == 0) and self.timestamp not in self.reconstructed_ids:
                    self.most_recent_3d_id = self.timestamp
                    self.reconstructed_ids.append(self.timestamp)

                self.timestamp = min(end_timestamp + 1, self.timestamp + self.interval)
                read = min(end_timestamp + 1, self.timestamp + self.interval)
            else:
                time.sleep(0.1)

            #m.acquire()
            if self.delivery_task or (self.init and time.time() - start_t >= self.init_tp):
                self.online_end_timestamp = self.timestamp
                total_task = self.online_end_timestamp - self.online_start_timestamp
                ###########################################
                print(
                    logger() + f"enter new TP, total tasks = {total_task}, {self.online_start_timestamp}-{self.online_end_timestamp}")
                pre_tp_length = self.p2(self.th, self.resolution)  # run tasks
                ###########################################
                self.save_d_frequency(total_task, self.th)
                self.diff_th = self.update_diff_th()
                ###############################
                # self.profile()
                self.p1(pre_tp_length, total_task)
                ##################################################
                self.online_start_timestamp = self.timestamp + 1
                self.reconstructed_ids = [self.online_start_timestamp]
                self.most_recent_3d_model_id = self.online_start_timestamp
                read = 0
                ###################################################
                self.init = False
                ###################################################
            #m.release()

            if self.timestamp > end_timestamp and self.reconstructed_ids[0] >= end_timestamp + 1:
                break

        self.done = True
        print("done.......................")
        while True:
            time.sleep(1)

    def batch_processing(self, start, end, scheduled, assignments, resolution, th):
        """

        Args:
            start: 1
            end: 20
            scheduled: [1, 4, 10, 12, 20]
            assignments: {'127.0.0.1':[1, 4], '127.0.0.1':[10, 12, 20]}
            resolution: 1.0
        """
        start_t = time.time()
        self.total_task += end - start
        self.tasks.append(end - start)
        print(logger() + f"scheduled = {len(scheduled)}, assignment = {assignments}")
        finished = []
        if len(scheduled) > 0:
            task_info = []
            for server, assign in assignments.items():
                if len(assign) > 0:
                    task_info.append((server, assign, resolution))

            with ThreadPool(len(task_info)) as poc:
                finished = poc.map(self.schedule, task_info)

        time_elapsed = time.time() - start_t
        self.cts.append(round(time_elapsed, 1))

        time_wait = time.time()

        while time.time() - start_t < self.init_tp:  # Abr 3 Fix self.init_tp
            # print("ct wait")
            time.sleep(1)

        tp = round(time.time() - start_t, 1)

        self.tps.append(tp)
        print(logger() + f"TP length = {tp}, start to compute F-score for {start} to {end}")

        #m.acquire()
        self.in_eva = True
        #m.release()

        compares = []
        for timestamp in range(start, end + 1):
            if timestamp not in scheduled:
                compares.append((timestamp, self.most_recent_3d_model_id))
            else:
                self.most_recent_3d_model_id = timestamp
                compares.append((timestamp, timestamp))

        models = []
        names = []
        for timestamp in scheduled:
            path = os.path.join(args.output_dir, str(timestamp) + "_output", "scene_dense.ply")
            with open(path, 'rb') as file:
                data = file.read()
            models.append(base64.encodebytes(data).decode("utf-8"))
            names.append(str(timestamp) + "_scene_dense.ply")

        items = []
        if len(compares[:int(len(compares) / 2)]) > 0:
            items.append((self.servers[0]["socket"], compares[:int(len(compares) / 2)], models, names))
        if len(compares[int(len(compares) / 2):]) > 0:
            items.append((self.servers[1]["socket"], compares[int(len(compares) / 2):], models, names))

        ft = time.time()
        with ThreadPool(len(items)) as poc:
            fs = poc.map(self.compute_f_score, items)
        print("............. f1 done", time.time() - ft)

        print(fs)

        tp_quality = round(np.sum(fs) / len(compares), 4)
        self.f1_error[-1] = np.abs(tp_quality - self.f1_error[-1])
        self.f1_error_long_term.append(round(np.average(self.f1_error), 4))
        self.save_d_f_score(tp_quality, resolution, th)
        self.f1_d = self.update_f1_d()

        # go to next TP
        m.acquire()
        self.total_task_q += tp_quality * (end - start)
        self.fs.append(tp_quality)
        self.long_term_f.append(round(self.total_task_q * 1.0 / self.total_task, 4))
        self.quality_queue = max(0., self.quality_queue - tp_quality * (end - start) + self.quality_req * (end - start))
        print(
            f"current TP F-score = {tp_quality}, TP accumulated = {round(tp_quality * (end - start), 4)}, quality queue = {round(self.quality_queue, 4)}")
        print(f"score = {self.long_term_f[-1]}", self.long_term_f)
        print(f"fscore = {self.fs}")
        print(f"cts = {self.cts}")
        print(f"tps = {self.tps}")
        print(f"tasks = {np.sum(self.tasks)}/{self.tasks}")
        print(f"tps = {np.sum(self.tps)}")
        print(f"resolution = {self.rs}")
        print(f"ths = {self.ths}")
        print("########################################################")
        print(f"interval = {self.intervals}")
        print(f"x1 = {self.x1s}")
        print(f"x2 = {self.x2s}")
        print("########################################################")
        print(f"f1_error = {self.f1_error}")
        print(f"f1_error long-term = {self.f1_error_long_term}")
        print(f"s_error = {self.s_error}")
        print(f"s_error long-term = {self.s_error_long_term}")
        print("########################################################")
        print(f"feqs = {self.feqs}")
        print(f"predicted_feqs = {self.predicted_feqs}")
        print("########################################################")
        self.delivery_task = True
        self.in_eva = False
        m.release()
        return np.average(fs)

    @staticmethod
    def compute_f_score(paras):
        target, compares, models, names = paras
        msg = {"compares": compares, "models": models, "names": names, "type": "f1"}
        print("send f_score req")
        encoding = json.dumps(msg).encode("utf-8")
        send_msg(target, encoding)

        while True:
            data = recv_msg(target)
            info = json.loads(str(data.decode('utf-8')))
            return info["f1"]


if __name__ == '__main__':
    s = Scheduler()
    s.listen()
    print(logger() + "ALL server connected")
