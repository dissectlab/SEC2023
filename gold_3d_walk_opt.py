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
from PIL import Image


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
finished = []
mvs = []
mvs_finished = []
allow = 0


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
    #if t_resolution < 1.0:
    #    args.parameter = "data/parameter/sfm_data_" + str(t_resolution) + '.json'

    with open(args.parameter, "r") as jsonFile:
        sfm = json.load(jsonFile)

    sfm["root_path"] = "/home/edge/dist/a_new_project/" + os.path.join(collect_dir)

    focal_length = {
        '0.95': 2573.74101418868
    }
    principal_point = {
        '0.95': [1003.4292834283402, 546.2331346997867]
    }

    if t_resolution <= 1.0:

        for item in sfm["views"]:
            item["value"]["ptr_wrapper"]["data"]["width"] = int(1920 * t_resolution)
            item["value"]["ptr_wrapper"]["data"]["height"] = int(1080 * t_resolution)

        sfm["intrinsics"][0]["value"]["ptr_wrapper"]["data"]["width"] = int(1920 * t_resolution)
        sfm["intrinsics"][0]["value"]["ptr_wrapper"]["data"]["height"] = int(1080 * t_resolution)

        # t_resolution = 0.999999

        sfm["intrinsics"][0]["value"]["ptr_wrapper"]["data"]["focal_length"] = 2573.74101418868 * t_resolution
        sfm["intrinsics"][0]["value"]["ptr_wrapper"]["data"]["principal_point"] = [
            1003.4292834283402 * t_resolution,
            546.2331346997867 * t_resolution,
        ]

        pass

    Path(os.path.join(args.output_dir, str_timestamp + "_output/sfm/matches")).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.output_dir, str_timestamp + "_output/sfm/matches/sfm_data.json"), 'w',
              encoding='utf-8') as f:
        json.dump(sfm, f, indent=4)


def do_mvg(timestamp, t_resolution):
    global allow
    start = time.time()
    str_timestamp = str(timestamp)
    print("start to reconstruct-mvg {}".format(str_timestamp), '-' * 50)
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
        if t_resolution <= 1.0:

            src = cv2.imread(os.path.join(collect_dir, image_dir + ".jpg"))
            output = cv2.resize(src, (int(1920 * t_resolution), int(1080 * t_resolution)),
                                interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(os.path.join(collect_dir, image_dir + ".jpg"), output)
            """
            picture = Image.open(os.path.join(collect_dir, image_dir + ".jpg"))
            picture.save(os.path.join(collect_dir, image_dir + ".jpg"),
                         "JPEG",
                         optimize=True,
                         quality=4)
            """
        inx += 1
    update_sfm(t_resolution, str_timestamp, collect_dir)
    p = subprocess.Popen(
        ["python3", args.reconstructor, collect_dir, os.path.join(args.output_dir, str_timestamp + "_output"),
         "--preset", "MVG_FIX"]) #MVG_FIX
    p.wait()
    if p.returncode != 0:
        pass
    #print("openMVG done in", time.time() - start)


    p = subprocess.Popen(
        ["python3", args.reconstructor, collect_dir, os.path.join(args.output_dir, str_timestamp + "_output"),
         "--preset", "MVS_FIX"])
    p.wait()
    if p.returncode != 0:
        pass

    finished.append(round(time.time() - start, 4))
    #mvs.append(timestamp)
    #mvs_finished.append(round(time.time() - start, 4))


def do_mvs(timestamp, t_resolution):
    start = time.time()
    str_timestamp = str(timestamp)
    print("start to reconstruct-mvs {}".format(str_timestamp), '-' * 50)
    collect_dir = os.path.join(args.data_collect_dir, str_timestamp)
    p = subprocess.Popen(
        ["python3", args.reconstructor, collect_dir, os.path.join(args.output_dir, str_timestamp + "_output"),
         "--preset", "MVS_FIX"])
    p.wait()
    if p.returncode != 0:
        pass
    mvs_finished.append(round(time.time() - start, 4))


if __name__ == '__main__':
    p1 = start_new_thread(CPU, ())

    t_resolution = 0.5
    args.output_dir = args.output_dir + '_' + str(t_resolution)

    start_id = 245
    number = 4
    pre_timestamp, timestamp = start_id, start_id
    start_t = time.time()
    start_pre = start_t
    start_new_thread(do_mvg, (timestamp, t_resolution))
    timestamp += 1
    """
    start_new_thread(do_mvg, (timestamp, t_resolution))
    timestamp += 1
    start_new_thread(do_mvg, (timestamp, t_resolution))
    timestamp += 1
    time.sleep(1.5)
    """
    init = True
    while True:
        if timestamp < start_id + number and GPUtil.getGPUs()[0].memoryUtil*100 <= 90 and round(psutil.cpu_percent(0.25), 1) <= 70:
            print("new MVG task at CPU load=", round(psutil.cpu_percent(0.25), 1), "GPUMEM=", GPUtil.getGPUs()[0].memoryUtil*100)
            start_new_thread(do_mvg, (timestamp, t_resolution))
            timestamp += 1
            #allow -= 1
            start_pre = time.time()
            #time.sleep(1)
        else:
            if len(finished) == number:
                break
            else:
                time.sleep(0.1)

    print(finished)
    print("\ttotal=", round((time.time() - start_t), 4), "avg=", round((time.time() - start_t)/number, 4))

    # total_opt_75 = [7.1332,  8.8366,  10.0284,  11.6337 , 14.3093, 17.3747,  20.083]
    # total_opt_05 = [3.5876,  4.5081,  5.3114,   6.0523,   7.6675,  10.1451,  11.8045]
    # total_opt    = [10.3137, 13.5192, 16.0256,  18.5773,  23.6598, 28.2126, 33.4638, 39.0158, 48.7056]
    # total        = [10.3137, 12.9014, 15.8522,  19.3106,  26.8603, 34.966,  43.4742, 52.1625, 63.0455]
    # total_05     = [3.5876,  4.4213,  5.3529,   6.1259,   8.8994,  10.9449,  13.731]
    # number = [1, 2, 3, 4, 6, 8, 10, 12, 16]

    # print(results, np.average(results)/len(items))
    print("CPU=", round(np.average(cpus), 4), "GPU=", round(np.average(gpus), 4))

    """
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
    """