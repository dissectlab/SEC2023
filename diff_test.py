from multiprocessing.pool import ThreadPool
import numpy as np
import open3d
import os
import time
from diff import diff_houchao


def com(g, m, ths = [0.002, 0.004, 0.006, 0.008, 0.01]):
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


def do(th, st,  window):
    fscores = []
    most_recent_3d_id = st
    reconstructed_ids = [st]
    for timestamp in range(st+1, st + window):
        ############################
        if timestamp % 5 == 0 and timestamp > st:
            print(th, timestamp, "-> reconstructed_ids=", len(reconstructed_ids), "model", most_recent_3d_id)
            pass
        ############################
        str_timestamp = str(timestamp)

        #dirs = ["walking_all_frame_cam1", "walking_all_frame_cam2", "walking_all_frame_cam3", "walking_all_frame_cam4",
        #        "walking_all_frame_cam5"]

        dirs = ["0/Images", "1/Images", "2/Images",
                "3/Images",
                "4/Images"]

        img_file_name = str_timestamp + ".jpg"

        if timestamp > 0:
            src = []
            dest = []
            for image_dir in dirs:
                src.append(os.path.join('book', image_dir, str(most_recent_3d_id) + ".jpg"))
                dest.append(os.path.join('book', image_dir, img_file_name))

            diffs = diff_houchao.diff_2d(src, dest)
            avg_diff = round(np.average(diffs), 5)

            if avg_diff >= th:
                most_recent_3d_id = timestamp
                reconstructed_ids.append(timestamp)

            g = '/home/edge/dist/a_new_project/data/gt_book/' + str(most_recent_3d_id) + '_scene_dense.ply'
            m = '/home/edge/dist/a_new_project/data/gt_book/' + str(timestamp) + '_scene_dense.ply'

            fscores.append(com(g, m, ths=[0.003]))

    print(th, "-> number of reconstructed=", len(reconstructed_ids), "fscores=", round(np.average(fscores), 4))
    return round(len(reconstructed_ids)/window, 4), round(np.average(fscores), 4)


if __name__ == "__main__":
    most_recent_3d_id = 0
    x = [0.01, 0.035, 0.06]
    start = time.time()
    # items =[[0.08, 245 + 100, most_recent_3d_id]]
    #s, f = do(0.015, 400,  50)
    print(time.time() - start)

    f = []
    feq = [50, 48, 30, 12, 6, 2]
    for i in feq:
        f.append(i/50)
    print(f)

    # book 400 - 450
    # d   = [0.01, 0.02, 0.035, 0.06, 0.08, 0.10]
    # feq = [50, 48, 30, 12, 6, 2]
    # f1  = [1.0, 0.9963, 0.9595, 0.9084, 0.8867, 0.8427]

    # handshake 245 - 295
    # d   = [0.01, 0.02, 0.035, 0.06, 0.08, 0.09, 0.12]
    # feq = [42, 17, 7, 3, 2, 1, 1]
    # f1  = [0.9874, 0.94, 0.8782, 0.7833, 0.7522, 0.7233, 0.7233]

    # handshake 295 - 345
    # d   = [0.01, 0.02, 0.035, 0.06, 0.08, 0.09, 0.12]
    # feq = [49, 25, 13, 5, 3, 2, 1]
    # f1  = [0.9985, 0.9566, 0.9218, 0.8514, 0.8087, 0.7832, 0.6904]

