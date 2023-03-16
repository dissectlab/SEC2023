from _thread import start_new_thread
from threading import Lock
import matplotlib.pyplot as plt
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
from diff import diff_detector
import GPUtil

parser = argparse.ArgumentParser(description='Please specify the directory of data set')

parser.add_argument('--data_dir', type=str, default='walk',
                    help="the directory which contains the pictures set.")
parser.add_argument('--data_collect_dir', type=str, default='data/collect',
                    help="the directory which contains the pictures set.")
parser.add_argument('--output_dir', type=str, default='data/gold_results',
                    help="the directory which contains the final results.")
parser.add_argument('--parameter', type=str, default='data/parameter/sfm_data_walk.json', #
                    help="the directory which contains the pictures set.")
parser.add_argument('--reconstructor', type=str, default='mmm_parl.py', #
                    help="the directory which contains the reconstructor python script.")
parser.add_argument('--resolution', type=float, default=1.0,
                    help="the directory which contains the reconstructor python script.")
parser.add_argument('--start', type=int, default=946,
                    help="the directory which contains the reconstructor python script.")

args = parser.parse_args()


Path(args.data_collect_dir).mkdir(parents=True, exist_ok=True)
Path(args.output_dir).mkdir(parents=True, exist_ok=True)

m = Lock()

cpus = []
times = []
mem = []
gpus = []
mvg_finished = []
mvs = []
mvs_finished = []
allow = 0
cc = []
for i in range(16):
    cc.append(1)


def CPU():
    start = time.time()
    while True:
        m.acquire()
        cpus.append(round(psutil.cpu_percent(0.1), 1))
        gpus.append(GPUtil.getGPUs()[0].load*100)
        times.append(time.time() - start)
        mem.append(psutil.virtual_memory().percent)
        m.release()
        time.sleep(0.1)


def update_sfm(t_resolution, str_timestamp, collect_dir):
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
        sfm["intrinsics"][0]["value"]["ptr_wrapper"]["data"]["focal_length"] = int(1920 * t_resolution * 1.34)
        sfm["intrinsics"][0]["value"]["ptr_wrapper"]["data"]["principal_point"] = [
            int(1920 * t_resolution / 1.94),
            int(1080 * t_resolution / 1.97)]

    Path(os.path.join(args.output_dir, str_timestamp + "_output/sfm/matches")).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.output_dir, str_timestamp + "_output/sfm/matches/sfm_data.json"), 'w',
              encoding='utf-8') as f:
        json.dump(sfm, f, indent=4)


def do_mvg(timestamp, cpu_allocation, t_resolution):
    start = time.time()
    str_timestamp = str(timestamp)
    try:
        # pass
        shutil.rmtree(os.path.join(args.output_dir, str_timestamp + "_output"))
    except:
        pass
    collect_dir = os.path.join(args.data_collect_dir, str_timestamp)
    Path(collect_dir).mkdir(parents=True, exist_ok=True)
    # go through each camera
    inx = 0
    dirs = ["walking_all_frame_cam1", "walking_all_frame_cam2", "walking_all_frame_cam3", "walking_all_frame_cam4",
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
    update_sfm(t_resolution, str_timestamp, collect_dir)
    c_str = ''
    for i in range(len(cpu_allocation)):
        if i != len(cpu_allocation) - 1:
            c_str += str(cpu_allocation[i]) + ','
        else:
            c_str += str(cpu_allocation[i])
    print(f"openMVG {str_timestamp}, cpus in", c_str)
    p = subprocess.Popen(
        ['taskset', '-c', c_str, "python3", args.reconstructor, collect_dir, os.path.join(args.output_dir, str_timestamp + "_output"),
         "--preset", "MVG_FIX"])
    p.wait()
    if p.returncode != 0:
        pass
    m.acquire()
    mvg_finished.append(round(time.time() - start, 4))
    mvs.append(timestamp)
    for item in cpu_allocation:
        cc[item] = 1
    m.release()


def do_mvs(timestamp, cpu_allocation):
    start = time.time()
    str_timestamp = str(timestamp)
    collect_dir = os.path.join(args.data_collect_dir, str_timestamp)
    c_str = ''
    for i in range(len(cpu_allocation)):
        if i != len(cpu_allocation) - 1:
            c_str += str(cpu_allocation[i]) + ','
        else:
            c_str += str(cpu_allocation[i])
    print(f"openMVS {str_timestamp}, cpus in", c_str)
    p = subprocess.Popen(
        ['taskset', '-c', c_str, "python3", args.reconstructor, collect_dir, os.path.join(args.output_dir, str_timestamp + "_output"),
         "--preset", "MVS_FIX"])
    p.wait()
    if p.returncode != 0:
        pass
    print(f"\topenMVS {str_timestamp}, done")
    m.acquire()
    mvs_finished.append(round(time.time() - start, 4))
    for item in cpu_allocation:
        cc[item] = 1
    m.release()


if __name__ == '__main__':
    p1 = start_new_thread(CPU, ())
    start_t = time.time()
    s, start_id = 245, 245
    number = 3
    openmvg_cpus = 4
    openmvs_cpus = 4
    while True:
        if False and sum(cc) >= openmvs_cpus and len(mvs) > 0:
            cpu_allocation = []
            m.acquire()
            for i in range(16):
                if cc[i] == 1:
                    cpu_allocation.append(i)
                    cc[i] = 0
                if len(cpu_allocation) == openmvs_cpus:
                    break
            m.release()
            start_new_thread(do_mvs, (mvs[0], cpu_allocation))
            mvs.pop(0)
            #time.sleep(0.5)
        elif sum(cc) >= openmvg_cpus and s < start_id + number:
            m.acquire()
            cpu_allocation = []
            for i in range(16):
                if cc[i] == 1:
                    cpu_allocation.append(i)
                    cc[i] = 0
                if len(cpu_allocation) == openmvg_cpus:
                    break
            m.release()
            start_new_thread(do_mvs, (s, cpu_allocation))
            s += 1
            #time.sleep(0.5)
        if s >= start_id + number:
            if len(mvs_finished) == number:
                break
            else:
                time.sleep(0.01)

    # total_opt = [10.3137, 13.5192, 16.0256,  18.5773,  23.6598, 28.2126, 33.4638, 39.0158, 48.7056]
    # total     = [10.3137, 12.9014, 15.8522,  19.3106,  26.8603, 34.966, 43.4742, 52.1625, 63.0455]
    # number = [1, 2, 3, 4, 6, 8, 10, 12, 16]
    print(mvg_finished)
    print(mvs_finished)
    print("total time=", time.time() - start_t)
    print("CPU=", round(np.average(cpus), 4), "GPU=", round(np.average(gpus), 4))

    print("cpus=", cpus)
    print("gpus=", gpus)
    print("times=", times)

    m.acquire()
    plt.subplot(1, 2, 1)
    plt.scatter(times, cpus)
    plt.ylim([0, 100])
    plt.xlabel("TimeLine")
    plt.ylabel("CPU%")

    plt.subplot(1, 2, 2)
    plt.scatter(times, gpus)
    plt.ylim([0, 100])
    plt.xlabel("TimeLine")
    plt.ylabel("GPU%")

    plt.show()
    m.release()