from numpy import arange
from pandas import read_csv
from scipy.optimize import curve_fit
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
import numpy as np
import math


def func(x, a, b, c):
    return a * (x**2) + b*x + c


def fit(time_avg_small_obs, x=[], eva=[], gt=[], l=None):

    popt, error = curve_fit(func, x, time_avg_small_obs)
    a, b, c = popt
    print('%.15f, %.15f, %.15f', (a, b, c))

    pyplot.scatter(x, time_avg_small_obs)

    pred = [func(i, a, b, c) for i in eva]

    pyplot.plot(eva, pred, '--',  label=l)

    pyplot.scatter(eva, gt)

    print(mean_squared_error(pred, gt, squared=False))

    print(pred)

    #pyplot.show()

    return a, b, c


if __name__ == "__main__":

    xx = [1, 2, 3, 4, 6, 8, 10]

    obs = np.array([10.313, 12.9014, 15.8522, 19.3106, 26.8603, 34.966, 43.4742]) # server #1

    avg = []
    for x in range(len(xx)):
        avg.append(obs[x]/xx[x])

    print("avg1=", avg)

    a, b, c = fit(obs, x=xx, eva=xx, gt=obs, l="200_250_prt")

    obs = np.array([8.0173, 11.6042, 15.4985, 19.2794, 28.2017, 36.459, 46.0])

    avg = []
    for x in range(len(xx)):
        avg.append(obs[x] / xx[x])

    print("avg1=", avg)

    a, b, c = fit(obs, x=xx, eva=xx, gt=obs, l="200_250_prt")

    pyplot.legend()
    pyplot.show()

