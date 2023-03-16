import random
import time
from scipy.optimize import curve_fit
from matplotlib import pyplot
import numpy as np
from gekko import GEKKO
from sklearn.metrics import mean_squared_error


def quantize(val):
    to_values = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08]
    best_match = None
    best_match_diff = None
    for other_val in to_values:
        diff = abs(other_val - val)
        if best_match is None or diff < best_match_diff:
            best_match = other_val
            best_match_diff = diff
    return best_match


def func_long_tail(x, a, b, c):
    return a * np.exp(-b * x) + c


def func_2nd(x, a, b, c):
    return a * (x ** 2) + b * x + c


def fit(time_avg_small_obs, x=[]):
    popt1, error1 = curve_fit(func_long_tail, x, time_avg_small_obs,
                              maxfev=10000)  # , p0=[2.1336807292877547,82.88051139640463,0.02538739144646269]
    a, b, c = popt1
    pred = [func_long_tail(i, a, b, c) for i in x]
    e1 = mean_squared_error(pred, time_avg_small_obs, squared=True)

    popt2, error2 = curve_fit(func_2nd, x, time_avg_small_obs, maxfev=10000)
    a, b, c = popt2
    pred = [func_2nd(i, a, b, c) for i in x]
    e2 = mean_squared_error(pred, time_avg_small_obs, squared=True)

    if e1 < e2:
        label = "func_long_tail"
        a, b, c = popt1
    else:
        label = "func_long_tail"
        a, b, c = popt1

    print("a, b, c, type = ", a, b, c, label)
    return a, b, c, label


def task_assignment(task_number, server_paras, r):
    # server computation time
    a1, b1, c1 = server_paras[0]
    a2, b2, c2 = server_paras[1]

    # time-res
    a6, b6, c6 = 6.186904829165969, 4.124880732440143, 0.1140477326972007
    a7, b7, c7 = 5.1460114860852615, 2.9943873782169566, -0.007844731472244959

    m = GEKKO(remote=False)

    x1 = m.Var(1, lb=0, ub=task_number, integer=True)
    x2 = m.Var(1, lb=0, ub=task_number, integer=True)
    Z = m.Var(None)

    m.Minimize(Z)
    m.Equation(x1 + x2 == task_number)
    m.Equation([x1 >= 0, x2 >= 0])
    m.Equations([Z >= (a6 * (r**2) + b6*r + c6) * (a1 * (x1**2) + b1*x1 + c1) / 10.324, Z >= (a7 * (r**2) + b7*r + c7) * (a2 * (x2**2) + b2*x2 + c2) / 8.0173])
    m.solve(disp=False)
    print('x1: ', x1.value[0])
    print('x2: ', x2.value[0])
    print('Z: ', Z.VALUE[0])
    return [int(x1.value[0]), int(x2.value[0]), round(Z.VALUE[0], 4)]


def solver_long_tail(v, A, quality_q, task_number, server_paras, diff_th, f1_res, f1_d):
    # server computation time
    a1, b1, c1 = server_paras[0]
    a2, b2, c2 = server_paras[1]

    # d
    a3, b3, c3, diff_th_type = diff_th

    # f1-res
    a4, b4, c4 = f1_res

    # f1 - d
    a5, b5, c5, f1_d_type = f1_d

    # time-res
    a6, b6, c6 = 6.186904829165969, 4.124880732440143, 0.1140477326972007
    a7, b7, c7 = 5.1460114860852615, 2.9943873782169566, -0.007844731472244959

    start_t = time.time()

    m = GEKKO(remote=False)
    m.options.SOLVER = 1

    x1 = m.Var(1, lb=0, ub=task_number, integer=True)
    x2 = m.Var(1, lb=0, ub=task_number, integer=True)
    d = m.Var(1, lb=0.005, ub=0.07, integer=False)
    r = m.Var(1, lb=0.2, ub=0.95, integer=False)
    Z = m.Var(None)

    if f1_d_type == "func_long_tail":
        m.Minimize(v * Z - quality_q * (c4 - a4 * m.exp(-b4 * r)) * (a5 * m.exp(-b5 * d) + c5) * task_number)
    else:
        m.Minimize(v * Z - quality_q * (c4 - a4 * m.exp(-b4 * r)) * (a5 * (d * d) + b5 * d + c5) * task_number)

    if diff_th_type == "func_long_tail":
        m.Equation(x1 + x2 == (a3 * m.exp(-b3 * d) + c3) * task_number)
    else:
        m.Equation(x1 + x2 == (a3 * (d * d) + b3 * d + c3) * task_number)

    m.Equation([x1 >= 0, x2 >= 1])
    m.Equations([Z >= (a6 * (r ** 2) + b6 * r + c6) * (a1 * (x1 ** 2) + b1 * x1 + c1) / 10.324,
                 Z >= (a7 * (r ** 2) + b7 * r + c7) * (a2 * (x2 ** 2) + b2 * x2 + c2) / 8.0173])
    m.solve(disp=False)

    if f1_d_type == "func_long_tail":
        q = (c4 - a4 * np.exp(-b4 * r.value[0])) * (a5 * np.exp(-b5 * d.value[0]) + c5)
    else:
        q = (c4 - a4 * np.exp(-b4 * r.value[0])) * (a5 * (d.value[0] * d.value[0]) + b5 * d.value[0] + c5)

    if diff_th_type == "func_long_tail":
        feq = a3 * np.exp(-b3 * d.value[0]) + c3
    else:
        feq = a3 * (d.value[0] * d.value[0]) + b3 * d.value[0] + c3

    print("queue=", quality_q)
    print("F1=", q)
    print('x1: ', x1.value[0])
    print('x2: ', x2.value[0])
    print('d: ', d.value[0])
    print('r: ', r.value[0])
    print('Z:  ', Z.value[0])
    print("feq:", feq, "total task:", task_number)

    return [q, max(feq, 1. / task_number), int(x1.value[0]), max(1, int(x2.value[0])), round(d.value[0], 4),
            round(r.value[0], 4), round(q, 4), round(Z.value[0], 4)]


def config_v_w(A, quality_q, server_paras, diff_th, f1_res, f1_d):
    opt_v = None
    opt_w = None
    for v in [1, 3, 5, 10]:
        times = []
        f = []
        ws = [25, 50, 75, 100, 150]
        opt = None
        min_time = None
        max_q = None
        for window in ws:
            total_task = 0
            total_quality = 0
            quality_q = 0
            total_time = 0
            for i in range(int(1160 / window)):
                f1, feq, x1, x2, _, _, _, z = solver_long_tail(v, A, quality_q, window, server_paras, diff_th, f1_res, f1_d)
                f1 = f1
                total_quality += f1 * window
                total_task += window
                total_time += max(z, window / 5)
                quality_q = quality_q + (A - f1) * window

            times.append(total_time)
            f.append(total_quality / total_task)
            if opt is None or total_time < min_time:
                opt = window
                min_time = total_time
                max_q = round(total_quality / total_task, 4)

        print(v, times)
        print(v, f)
        print(opt, min_time, max_q)
        print("#######################")
        if A - max_q > 0.01:
            break
        else:
            opt_v = v
            opt_w = opt
    print("opt_v, opt_w = ", opt_v, opt_w)
    return opt_v, opt_w


if __name__ == '__main__':

    """
    config_v_w(0.8, 0,  [(0.1016771987765127, 2.609735151185393, 7.36594987756804), (
                                                       0.06157433135916346, 3.549801161121459, 4.314966329861685)],
                                                   (1.5623283286278888, 101.72954737620907, 0.02666130617607339,
                                                    'func_long_tail'),
                                                   (1.3690845352745729, 4.812264580079275, 0.956047986254025),
                                                   (0.2729603426237055, 23.08876265571103, 0.7583748391471873,
                                                    'func_long_tail')
                                                   )

   
    
    solver_long_tail(1, 0.9, 28.1, 225,
               [(0.1016771987765127, 2.609735151185393, 7.36594987756804), (0.06157433135916346, 3.549801161121459, 4.314966329861685)],
               (1.5846245992203345, 94.74105309570432, 0.011904457365174101, 'func_long_tail'),
               (1.3690845352745729, 4.812264580079275, 0.956047986254025),
               (0.24095616067017608, 49.96429071480146, 0.8178944912077853, 'func_long_tail')
               )
     """

    a, b, c = 1.3690845352745729, 4.812264580079275, 0.956047986254025

    a, b, c = 1.513081102759687, 5.485489453291869, 0.9630077817664481

    a, b, c = 1.5198856981333275, 5.26164086391182, 0.9491640358096844

    for i in range(2, 100):
        r = i/100.
        f = (c - a * np.exp(-b * r))
        if f >= 0.9:
            print(r, f)
            break

    # handshake = [1202, 1499, 1983], [532, 679, 1771]
    # walk = [1409, 1783, 2470], [345, 514, 1805]

    #r = 0.41

    x1, x1, z = task_assignment(20, [(0.1016771987765127, 2.609735151185393, 7.36594987756804), (0.06157433135916346, 3.549801161121459, 4.314966329861685)], r)

    print(z * 1800/20)

    """
    total_task = 0
    total_quality = 0
    quality_q = 0
    total_time = 0
    window = 50
    i = 0
    while i < 2000:
        f1, feq, x1, x2, _, _, _, z = solver_long_tail(True, 0.85, quality_q, window,
                                                       [(0.1016771987765127, 2.609735151185393, 7.36594987756804), (
                                                           0.06157433135916346, 3.549801161121459, 4.314966329861685)],
                                                       (1.622043243211939, 99.36347478916126, 0.013345967953400671,
                                                        'func_long_tail'),
                                                       (1.3690845352745729, 4.812264580079275, 0.956047986254025),
                                                       (0.2328482265627525, 48.31745872564334, 0.8171457568785775,
                                                        'func_long_tail')
                                                       )
        mu, sigma = 0, 0.1
        f1 = f1
        total_quality += f1 * window
        total_task += window
        total_time += max(z, window/5)
        quality_q = quality_q + (0.85 - f1) * window

        print(i, window, z, x1+x2, quality_q, total_quality / total_task, total_time, total_task)

        i = i + window

        if 0.85 - total_quality / total_task >= 0.3:
            window = 50
        else:
            window = min(100, int(z * 5.))

        if i + window > 2000:
            window = window - (i + window - 2000)

        # 940.6961999999999, 0.8346678037974067

    """
