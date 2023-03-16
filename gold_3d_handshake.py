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
import psutil


parser = argparse.ArgumentParser(description='Please specify the directory of data set')

parser.add_argument('--data_dir', type=str, default='handshake',
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


def do(items):
    global cpus, gpus

    start, t_resolution = items

    most_recent_3d_id = 0
    reconstructed_ids = [0]
    time_info = []

    for timestamp in range(start, start + 1):
        ############################
        if timestamp % 5 == 0 and timestamp > 0:
            print("time=", time_info)
        ############################
        start = time.time()
        str_timestamp = str(timestamp)
        print("current timestamp: ", str_timestamp, '-' * 50)

       # args.output_dir = args.output_dir + '_' + str(t_resolution)

        try:
            #pass
            shutil.rmtree(os.path.join(args.output_dir, str_timestamp + "_output"))
        except:
            pass

        collect_dir = os.path.join(args.data_collect_dir, str_timestamp)

        Path(collect_dir).mkdir(parents=True, exist_ok=True)

        start = time.time()
        # go through each camera
        inx = 0

        dirs = ["walking_all_frame_cam1", "walking_all_frame_cam2", "walking_all_frame_cam3", "walking_all_frame_cam4",
                "walking_all_frame_cam5"]

        img_file_name = str_timestamp + ".jpg"

        if timestamp > start:
            src = []
            dest = []
            for image_dir in dirs:
                src.append(os.path.join(args.data_dir, image_dir, str(most_recent_3d_id) + ".jpg"))
                dest.append(os.path.join(args.data_dir, image_dir, img_file_name))
            diffs = diff_detector.diff_2d(src, dest)

            avg_diff = round(np.average(diffs), 5)
            print("diffs=", diffs, "avg=", avg_diff)

            if avg_diff < 0.001:
                print("using most_recent_3d_id=", most_recent_3d_id)
                Path(os.path.join(args.output_dir, str(timestamp) + "_output", "mvs")).mkdir(parents=True,
                                                                                             exist_ok=True)
                shutil.copy2(
                    os.path.join(args.output_dir, str(most_recent_3d_id) + "_output", "mvs", "scene_dense.ply"),
                    os.path.join(args.output_dir, str(timestamp) + "_output", "mvs", "scene_dense.ply"))
                print("reconstructed_ids=", reconstructed_ids)
                time_info.append(round(time.time() - start, 4))
                continue
            else:
                most_recent_3d_id = timestamp
                reconstructed_ids.append(timestamp)

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

        Path(os.path.join(args.output_dir, str_timestamp + "_output/sfm/matches")).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(args.output_dir, str_timestamp + "_output/sfm/matches/sfm_data.json"), 'w',
                  encoding='utf-8') as f:
            json.dump(sfm, f, indent=4)

        # start to run openMvg + openMvs for foreground
        # start = time.time()
        print("start to reconstruct {}".format(str_timestamp))


        p = subprocess.Popen(
            ["python3", args.reconstructor, collect_dir, os.path.join(args.output_dir, str_timestamp + "_output"),
             "--preset", "MVG_FIX"]) # SEQUENTIAL_FIX
        p.wait()
        if p.returncode != 0:
            break

        p = subprocess.Popen(
            [ "python3", args.reconstructor, collect_dir, os.path.join(args.output_dir, str_timestamp + "_output"),
             "--preset", "MVS_FIX"])
        p.wait()
        if p.returncode != 0:
            break

        time_info.append(round(time.time() - start, 4))

    print("time=", time_info)
    print("total time=", round(sum(time_info), 4))
    return round(sum(time_info), 4)


if __name__ == '__main__':
    p1 = start_new_thread(CPU, ())

    s = 245 + 40

    items = [] #
    id = 0

    resolution = 0.6
    args.output_dir = args.output_dir + '_' + str(resolution)
    for i in range(s, s+8):
        items.append((i, resolution))
        id += 1
        #resolution -= 0.1

    # total_opt = [10.3137, 13.5192, 16.0256,  18.5773,  23.6598,  28.2126, 33.4638, 39.0158, 48.7056]
    # total     = [10.3137, 12.9014, 15.8522,  19.3106,  26.8603,  34.966,  43.4742, 52.1625, 63.0455]
    # total_03  = [1.95,  2.4034,  2.814,    3.221,    4.621,    5.632, 7.435]
    # total_05  = [3.7128,  4.6152,  5.3114,   6.3173,   8.83,   10.84,  13.6474]
    # total_08 =  [7.321 ,  9.224 ,  10.83 ,   12.833,   17.850,   23.092, 28.833]
    # number = [1, 2, 3, 4, 6, 8, 10, 12, 16]

    # obs3 = [1.95, 2.711, 3.71, 4.817, 6.016, 7.321, 9.023,  10.324]
    # xx = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    start_t = time.time()
    with ThreadPool(len(items)) as p:
        results = p.map(do, items)

    print(results, (time.time() - start_t))
    print("CPU=", round(np.average(cpus), 4), "GPU=", round(np.average(gpus), 4))
