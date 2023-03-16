import sys
import os

import cv2

import numpy as np

import time

import matplotlib.pyplot as plt

from scipy.linalg import norm
from scipy import sum, average
from multiprocessing import Pool
from threading import Thread, Lock
import threading
from concurrent.futures import ThreadPoolExecutor
import pickle


def compare_images(items):
    src, dest = items
    img1 = cv2.imread(src)
    img2 = cv2.imread(dest)

    # read images as 2D arrays (convert to grayscale for simplicity)
    gray_img1 = to_grayscale(img1)
    gray_img2 = to_grayscale(img2)

    norm_img1 = normalize(gray_img1)
    norm_img2 = normalize(gray_img2)

    # compare
    precent_change = cal_ndts(norm_img1, norm_img2)

    # print(precent_change)

    return precent_change


def cal_ndts(img1, img2):

    array_ndts = (img2 - img1) / (img2 + img1)
    array_square_ndts = array_ndts * array_ndts

    threshold = 0.0075
    threshold_array_square_ndts = array_square_ndts > threshold
    threshold_array_square_ndts = threshold_array_square_ndts.astype(int)

    percent_change = sum(threshold_array_square_ndts) / threshold_array_square_ndts.size
    return percent_change


def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def normalize(gray_image):

    norm_image = cv2.normalize(gray_image, None, alpha=0.1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    return np.asarray(norm_image)


def cal(items):
    img1, img2 = items
    return round(compare_images(img1, img2), 5)


def diff_2d(src, dest):
    items = []
    # st = time.time()
    for i in range(len(src)):
        items.append((src[i], dest[i]))
    # diffs = None
    """
    with Pool(len(items)) as p:
        diffs = p.map(cal, items)
    # print(time.time() - st)
    """
    with ThreadPoolExecutor(max_workers=16) as executor:
        results = executor.map(compare_images, items)
    diffs = []
    for result in results:
        diffs.append(result)
    #print(diffs)
    return diffs


if __name__ == "__main__":
    path = "/home/edge/dist/a_new_project/walk/walking_all_frame_cam1/"
    src1 = [path + "246.jpg", path + "256.jpg", path + "266.jpg", path + "276.jpg", path + "286.jpg"]
    dest1 = [path + "247.jpg", path + "248.jpg", path + "349.jpg", path + "250.jpg", path + "251.jpg"]
    start = time.time()
    print(diff_2d(src1, dest1), time.time() - start)
