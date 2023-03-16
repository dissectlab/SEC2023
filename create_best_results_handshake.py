import cv2
import os
import time
from pathlib import Path
import argparse
import subprocess
import shutil
import json
from multiprocessing.pool import ThreadPool


parser = argparse.ArgumentParser(description='Please specify the directory of data set')

parser.add_argument('--data_dir', type=str, default='handshake',
                    help="the directory which contains the pictures set.")
parser.add_argument('--data_collect_dir', type=str, default='data/collect',
                    help="the directory which contains the pictures set.")
parser.add_argument('--output_dir', type=str, default='data/gold_results_1.0',
                    help="the directory which contains the final results.")
parser.add_argument('--parameter', type=str, default='data/parameter/sfm_data_handshake.json',  #
                    help="the directory which contains the pictures set.")
parser.add_argument('--reconstructor', type=str, default='mmm_parl.py',  #
                    help="the directory which contains the reconstructor python script.")
parser.add_argument('--resolution', type=float, default=1.0,
                    help="the directory which contains the reconstructor python script.")
parser.add_argument('--start', type=int, default=946,
                    help="the directory which contains the reconstructor python script.")

args = parser.parse_args()

Path(args.data_collect_dir).mkdir(parents=True, exist_ok=True)
Path(args.output_dir).mkdir(parents=True, exist_ok=True)


class Server:
    def __init__(self):
        #self.delete()
        self.test()
        pass

    """
    [1,     2,     3,     4,     6,     8,    10]
    [10.72, 13.83, 16.43, 19.53, 27.96, 35.29, 44.11]
    """
    def test(self, i=1045):
        while i <= 2045:
            r = 0.95
            items = [(i, r), (i + 1, r), (i + 2, r), (i + 3, r), (i + 4, r)]
            start_t = time.time()
            with ThreadPool(len(items)) as p:
                p.map(self.batch_processing, items)
            end_t = time.time() - start_t
            print(f"total time={end_t}, timestamp={i}/{i+4}")
            i += 5
            print("##################### sleep 30s")
            time.sleep(30)

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

        dirs = ["0/Images", "1/Images", "2/Images",
                "3/Images",
                "4/Images"]

        img_file_name = str_timestamp + ".jpg"

        for image_dir in dirs:
            shutil.copy2(os.path.join(args.data_dir, image_dir, img_file_name),
                         os.path.join(collect_dir, image_dir[:1] + ".jpg"))
            if t_resolution < 1.0:
                src = cv2.imread(os.path.join(args.data_dir, image_dir, img_file_name))
                output = cv2.resize(src, (int(1920 * t_resolution), int(1080 * t_resolution)),
                                    interpolation=cv2.INTER_LANCZOS4)
                cv2.imwrite(os.path.join(collect_dir, image_dir[:1] + ".jpg"), output)
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
            sfm["intrinsics"][0]["value"]["ptr_wrapper"]["data"]["focal_length"] = 2600.2429848099407 * t_resolution
            sfm["intrinsics"][0]["value"]["ptr_wrapper"]["data"]["principal_point"] = [
                1006.7495251592154 * t_resolution,
                587.4608061256939 * t_resolution,
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

        shutil.copy2("data/gold_results_1.0/" + str_timestamp + "_output/mvs/scene_dense.ply",
                     "data/gt_handshake_f_test/" + str_timestamp + "_" + str(int(t_resolution * 100)) + "_scene_dense.ply")
        shutil.rmtree(os.path.join(args.output_dir, str_timestamp + "_output"))

        return round(sum(time_info), 4)


if __name__ == '__main__':
    s = Server()
    #s.test()
