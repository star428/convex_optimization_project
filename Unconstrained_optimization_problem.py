#!/usr/bin/env python
# encoding: utf-8
"""
@author: star428
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited 
@contact: yewang863@gmail.com
@software: pycharm
@file: Unconstrained_optimization_problem.py
@time: 2020/11/23 22:48
@desc:
"""
import numpy as np


def Function(x):
    result = np.exp(x[0] + 3 * x[1] - 0.1) + np.exp(x[0] - 3 * x[1] - 0.1) + \
             np.exp(-x[0] - 0.1)

    return result


def Jacobian(x):
    return np.array([np.exp(x[0] + 3 * x[1] - 0.1) +
                     np.exp(x[0] - 3 * x[1] - 0.1) -
                     np.exp(-x[0] - 0.1),

                     3 * np.exp(x[0] + 3 * x[1] - 0.1) -
                     3 * np.exp(x[0] - 3 * x[1] - 0.1)
                     ])


def Gradient_descent_method(x0, a, b, epz):
    """x0此时为初始点，a为α，b为β，epz为精度"""
    # x0 = np.array([1, 1])
    xmin = np.array([-0.5 * np.log(2), 0])
    x = x0

    the_list = [str(a) + "," + str(b)]
    the_inside_list = []
    the_inside_list.append(Function(x0) - Function(xmin))

    ifStop = np.linalg.norm(Jacobian(x)) - epz
    while ifStop > 0:
        d = -1 * Jacobian(x)

        t = 1

        while Function(x + t * d) >= Function(x) - a * t * \
                np.linalg.norm(Jacobian(x)):
            t = b * t
            if t < 10 ** -60:
                break

        x = x + t * d
        ifStop = np.linalg.norm(Jacobian(x)) - epz
        the_inside_list.append(Function(x) - Function(xmin))
        if t < 10 ** -60:
            break

    the_list.append(the_inside_list)
    return the_list


if __name__ == "__main__":
    x0 = np.array([1, 1])
    the_list = Gradient_descent_method(x0, 0.1, 0.7, 0.001)
    print(the_list)

    # print(Function(x0))
