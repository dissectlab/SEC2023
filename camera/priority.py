import os
import subprocess

import ujson as json
import time
from itertools import combinations
import numpy as np


def compute_cam_cpointM(file_path):
    with open(file_path) as data_file:
        data = json.load(data_file)
        num_views = len(data['extrinsics'])
        points3d = data['structure']
        num_pts = len(points3d)
        # Create matrix
        m = np.zeros((num_views, num_pts))

        pt_cnt = 0
        for key in points3d:
            matches = [k['key'] for k in key['value']['observations']]
            for cam_id in matches:
                # put 1's where there is match
                m[cam_id][pt_cnt] = 1
                # print("cam id = ", cam_id, "\tpt_cnt = ", pt_cnt)
            pt_cnt += 1
        return m, num_views, num_pts


def my_getk_optimal_cams(k, json_file_path):
    points_ik, num_views, num_pts = compute_cam_cpointM(json_file_path)
    view_range = range(0, num_views)
    combos = list(combinations(view_range, k))
    p = []
    for item in points_ik:
        p.append(sum(item))
    """
    points_sat = {}
    # Loop through each combination and return one with most points satisfied

    for c in combos:
        # initialize cameras bools
        bools = np.zeros((num_views, 1))
        for i in c:
            bools[i, 0] = 1
        # multiply each col by camera bools
        p = points_ik * bools
        # sum total amount of cameras covering each point
        p = p.sum(axis=0)
        # return count of >= 2
        num_pts_covered = np.count_nonzero(p >= 2, 0)
        points_sat[num_pts_covered] = c
    best_key = max(points_sat.keys())
    best_k = points_sat[best_key]
    print(combos)
    """
    return p


if __name__ == "__main__":
    a_p = [[], [], [], [], []]
    for i in range(0, 100):
        output_dir = "/home/edge/dist/a_new_project/data/gold_results/" + str(i) + "_output"

        pChange = subprocess.Popen(
            [os.path.join("openMVG_main_ConvertSfM_DataFormat"), "-i",
             output_dir + "/sfm/sfm_data.bin",
             "-o", output_dir + "/sfm/sfm_data.json"])
        pChange.wait()

        p = my_getk_optimal_cams(5, output_dir + "/sfm/sfm_data.json")
        for j in range(5):
            a_p[j].append(round(p[j]/sum(p), 4))
        if i > 10:
            th = 10
            print(i, round(np.average(a_p[0][-th:]), 4), round(np.average(a_p[1][-th:]), 4), round(np.average(a_p[2][-th:]), 4),
                  round(np.average(a_p[3][-th:]), 4), round(np.average(a_p[4][-th:]), 4))
