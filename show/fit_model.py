import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
import numpy as np


def func_long_tail(x, a, b, c):
    return a * np.exp(-b * x) + c


def func_2nd(x, a, b, c):
    return a * (x ** 2) + b * x + c


def fit(time_avg_small_obs, x=[], eva=[], gt=[], l=None):

    popt1, error1 = curve_fit(func_long_tail, x, time_avg_small_obs, p0=[2.1336807292877547,82.88051139640463,0.02538739144646269], maxfev=10000) #, p0=[2.1336807292877547,82.88051139640463,0.02538739144646269]
    a, b, c = popt1
    pred = [func_long_tail(i, a, b, c) for i in eva]
    e1 = mean_squared_error(pred, gt, squared=True)

    """
    popt2, error2 = curve_fit(func_2nd, x, time_avg_small_obs, maxfev=10000)
    a, b, c = popt2
    pred = [func_2nd(i, a, b, c) for i in eva]
    e2 = mean_squared_error(pred, gt, squared=True)

    if e1 < e2:
        label = "func_long_tail"
        a, b, c = popt1
    else:
        label = "func_2nd"
        a, b, c = popt2
    """
    print('%.15f, %.15f, %.15f', (a, b, c), e1)
    pyplot.scatter(x, time_avg_small_obs)
    pyplot.plot(eva, pred, '--',  label=l)
    pyplot.scatter(eva, gt)
    print("mean_squared_error", mean_squared_error(pred, gt, squared=False))
    print(pred)

    return a, b, c


if __name__ == "__main__":

    xx = [0.01, 0.02, 0.035, 0.06, 0.08, 0.09, 0.12]
    obs = [0.9874, 0.94, 0.8782, 0.7833, 0.7522, 0.7233, 0.7233]
    a, b, c = fit(obs, x=xx, eva=xx, gt=obs, l="old")

    obs = [0.9985, 0.9566, 0.9218, 0.8514, 0.8087, 0.7832, 0.6904]
    a, b, c = fit(obs, x=xx, eva=xx, gt=obs, l="old")

    pyplot.legend()
    pyplot.show()
