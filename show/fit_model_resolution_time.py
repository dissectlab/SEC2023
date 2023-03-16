from numpy import arange
from pandas import read_csv
from scipy.optimize import curve_fit
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
import numpy as np
import math


def func(x, a, b, c):
    return  a * (x**2) + b*x + c # a * np.exp(-b * x) + c


def fit(time_avg_small_obs, x=[], eva=[], gt=[], l=None):

    popt, error = curve_fit(func, x, time_avg_small_obs)
    a, b, c = popt
    print('%.15f, %.15f, %.15f', (a, b, c))

    pyplot.scatter(x, time_avg_small_obs)

    pred = [func(i, a, b, c) for i in eva]

    pyplot.plot(eva, pred, '--',  label=l)

    pyplot.scatter(eva, gt)

    print(mean_squared_error(pred, gt, squared=False))

    print("pred", pred)

    #pyplot.show()

    return a, b, c


if __name__ == "__main__":

    xx = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    obs = np.array([1.95, 2.711, 3.71, 4.817, 6.016, 7.321, 9.023,  10.324])
    obs = np.array([1.4032, 2.0043, 2.704, 3.606, 4.6081, 5.710, 7.01, 8.0173])
    # xx = np.array([576*324, 768*432, 960*540, 1152*648, 1344*756, 1536*864, 1728*972, 1920*1080])

    print(list(obs))
    a, b, c = fit(obs, x=xx, eva=xx, gt=obs, l="200_250_prt")

    for i in range(len(xx)):
        obs[i] = round(func(xx[i], a, b, c) * xx[i], 4)

    print(list(obs))

    pyplot.legend()
    pyplot.show()

