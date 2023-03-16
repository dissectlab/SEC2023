import base64
import socket
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
from common.networking import recv_msg, send_msg
import open3d
from diff_test import do


parser = argparse.ArgumentParser(description='Please specify the directory of data set')

parser.add_argument('--data_dir', type=str, default='walk',
                    help="the directory which contains the pictures set.")
parser.add_argument('--data_collect_dir', type=str, default='data/collect',
                    help="the directory which contains the pictures set.")
parser.add_argument('--output_dir', type=str, default='data/server_results',
                    help="the directory which contains the final results.")
parser.add_argument('--best_dir', type=str, default='data/gt',
                    help="the directory which contains the final results.")
parser.add_argument('--parameter', type=str, default='data/parameter/sfm_data_walk.json',  #
                    help="the directory which contains the pictures set.")
parser.add_argument('--reconstructor', type=str, default='mmm_parl.py',  #
                    help="the directory which contains the reconstructor python script.")
parser.add_argument('--resolution', type=float, default=1.0,
                    help="the directory which contains the reconstructor python script.")
parser.add_argument('--start', type=int, default=946,
                    help="the directory which contains the reconstructor python script.")
parser.add_argument('--port', type=int, default=8001,
                    help="the directory which contains the reconstructor python script.")

args = parser.parse_args()

Path(args.data_collect_dir).mkdir(parents=True, exist_ok=True)
Path(args.output_dir).mkdir(parents=True, exist_ok=True)


def f_score(g, m, ths=[0.003]):
    gt = open3d.io.read_point_cloud(g)
    pr = open3d.io.read_point_cloud(m)
    d1 = gt.compute_point_cloud_distance(pr)
    d2 = pr.compute_point_cloud_distance(gt)
    f = []
    for i in range(0, len(ths)):
        if len(d1) and len(d2):
            recall = float(sum(d < ths[i] for d in d2)) / float(len(d2))
            precision = float(sum(d < ths[i] for d in d1)) / float(len(d1))
            if recall + precision > 0:
                fscore = 2 * recall * precision / (recall + precision)
            else:
                fscore = 0
        else:
            fscore = 0
            precision = 0
            recall = 0
        f.append(fscore)
    return round(np.average(f), 4)


class Server:
    def __init__(self, port=8004):
        self.host = "146.95.252.24" #"146.95.252.28"
        self.port = port
        self.s = socket.socket()
        self.s.connect((self.host, self.port))

        while True:
            data = recv_msg(self.s)
            info = json.loads(str(data.decode('utf-8')))
            if info["type"] == "reconstruction":
                print(info)
                items = []
                for i in info["ids"]:
                    items.append((i, info["res"]))
                start_t = time.time()
                with ThreadPool(min(10, len(items))) as p:
                    results = p.map(self.batch_processing, items)
                end_t = time.time() - start_t
                models = []
                for i in info["ids"]:
                    path = os.path.join(args.output_dir, str(i) + "_output/mvs/", "scene_dense.ply")
                    with open(path, 'rb') as file:
                        data = file.read()
                    models.append(base64.encodebytes(data).decode("utf-8"))
                    shutil.rmtree(os.path.join(args.output_dir, str(i) + "_output"))
                msg = {"data": models, "status": 1, "time": round(end_t, 4)}
                send_msg(self.s, json.dumps(msg).encode("utf-8"))
                print("sent", results, round(end_t, 4))
            elif info["type"] == "profile":
                # do profile
                # msg = {"th": th, "start": start, "window": window, "type": "profile"}
                print(info)
                feq, f = do(info["th"], info["start"], info["window"])
                msg = {"s": feq, "f": f}
                send_msg(self.s, json.dumps(msg).encode("utf-8"))
                print("sent profile")
            else:
                # calculate F1-score
                # msg = {"compares": compares, "models": models, "names": names}
                ft = time.time()

                for i in range(len(info["models"])):
                    path = os.path.join(args.output_dir, info["names"][i])
                    with open(path, 'wb') as file:
                        file.write(base64.b64decode(info["models"][i]))
                f = []
                k = 0
                for item in info["compares"]:
                    f.append(self.compute_f_score(item))
                    print(k, f[-1])
                    k += 1

                msg = {"f1": np.sum(f)}
                send_msg(self.s, json.dumps(msg).encode("utf-8"))
                print("sent f1, speed = ", len(info["compares"])/(time.time() - ft))

    @staticmethod
    def compute_f_score(paras):
        g_timestamp, m_timestamp = paras
        f = f_score(os.path.join(args.best_dir, str(g_timestamp) + "_scene_dense.ply"),
                                 os.path.join(args.output_dir, str(m_timestamp) + "_scene_dense.ply"))
        return f

    @staticmethod
    def batch_processing(items):
        timestamp, t_resolution = items

        time_info = []

        str_timestamp = str(timestamp)
        try:
            # pass
            shutil.rmtree(os.path.join(args.output_dir, str_timestamp + "_output"))
        except:
            pass
        collect_dir = os.path.join(args.data_collect_dir, str_timestamp)
        Path(collect_dir).mkdir(parents=True, exist_ok=True)

        start = time.time()
        # go through each camera
        inx = 0

        dirs = ["walking_all_frame_cam1", "walking_all_frame_cam2", "walking_all_frame_cam3",
                "walking_all_frame_cam4",
                "walking_all_frame_cam5"]

        img_file_name = str_timestamp + ".jpg"

        for image_dir in dirs:
            shutil.copy2(os.path.join(args.data_dir, image_dir, img_file_name),
                         os.path.join(collect_dir, image_dir + ".jpg"))
            if t_resolution < 1.0:
                src = cv2.imread(os.path.join(collect_dir, image_dir + ".jpg"))
                output = cv2.resize(src, (int(1920 * t_resolution), int(1080 * t_resolution)),
                                    interpolation=cv2.INTER_LANCZOS4)
                cv2.imwrite(os.path.join(collect_dir, image_dir + ".jpg"), output)
            inx += 1

        # copy the sfm_data_gold.json to local
        with open(args.parameter, "r") as jsonFile:
            sfm = json.load(jsonFile)

        sfm["root_path"] = "/home/edge/dist/a_new_project/" + os.path.join(collect_dir)
        if t_resolution < 1.0:
            for item in sfm["views"]:
                item["value"]["ptr_wrapper"]["data"]["width"] = int(1920 * t_resolution)
                item["value"]["ptr_wrapper"]["data"]["height"] = int(1080 * t_resolution)

            sfm["intrinsics"][0]["value"]["ptr_wrapper"]["data"]["width"] = int(1920 * t_resolution)
            sfm["intrinsics"][0]["value"]["ptr_wrapper"]["data"]["height"] = int(1080 * t_resolution)
            sfm["intrinsics"][0]["value"]["ptr_wrapper"]["data"]["focal_length"] = 2573.74101418868 * t_resolution
            sfm["intrinsics"][0]["value"]["ptr_wrapper"]["data"]["principal_point"] = [
                1003.4292834283402 * t_resolution,
                546.2331346997867 * t_resolution,
            ]

        Path(os.path.join(args.output_dir, str_timestamp + "_output/sfm/matches")).mkdir(parents=True,
                                                                                         exist_ok=True)
        with open(os.path.join(args.output_dir, str_timestamp + "_output/sfm/matches/sfm_data.json"), 'w',
                  encoding='utf-8') as f:
            json.dump(sfm, f, indent=4)

        # start to run openMvg + openMvs for foreground
        print("start to reconstruct {}".format(str_timestamp))

        p = subprocess.Popen(
            ["python3", args.reconstructor, collect_dir, os.path.join(args.output_dir, str_timestamp + "_output"),
             "--preset", "MVG_FIX"])  # SEQUENTIAL_FIX
        p.wait()
        if p.returncode != 0:
            return 999

        p = subprocess.Popen(
            ["python3", args.reconstructor, collect_dir, os.path.join(args.output_dir, str_timestamp + "_output"),
             "--preset", "MVS_FIX"])
        p.wait()
        if p.returncode != 0:
            return 999

        time_info.append(round(time.time() - start, 4))
        return round(sum(time_info), 4)


if __name__ == '__main__':
    s = Server(args.port)
