#!/usr/bin/env python
# encoding: utf-8
"""
@author: star428
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited 
@contact: yewang863@gmail.com
@software: pycharm
@file: Equality_constraint_optimization.py
@time: 2020/11/24 15:04
@desc:
"""
import numpy as np
from numpy.linalg import matrix_rank
import random
import time


def Function(x):
    result = 0
    for i in range(0, 100):
        result += x[i] * np.log(x[i])

    return result


def Jacobian(x):
    the_list = []
    for i in range(0, 100):
        the_list.append(1 + np.log(x[i]))

    return np.array(the_list)


def Hessian(x):
    the_list = []
    for i in range(0, 100):
        the_list.append(1 / x[i])

    return np.diag(the_list)


def make_A():
    Stop = False
    while not Stop:
        matrix = np.random.random((30, 100))
        if matrix_rank(matrix) == 30:
            Stop = True

    return matrix


def make_X():
    x = []
    for i in range(0, 100):
        x.append(random.random())
    return np.array(x)


def Matrix_h_s_stack(hessianX, A):
    x = np.hstack((hessianX, A.T))
    y = np.hstack((A, np.zeros((30, 30))))
    return np.vstack((x, y))


def return_D(hessianX, A, x):
    detx = np.hstack((Jacobian(x), np.zeros(30)))
    dy = -1 * np.linalg.inv(Matrix_h_s_stack(hessianX, A)).dot(detx)

    return dy[0:100]


def return_V(hessianX, A, x):
    detx = np.hstack((Jacobian(x), np.zeros(30)))
    dy = -1 * np.linalg.inv(Matrix_h_s_stack(hessianX, A)).dot(detx)

    return dy[100:]


def Available_point_newton_method(x, A, a0, b0, epz):
    time_start = time.time()
    d = return_D(Hessian(x), A, x)
    test = A.dot(d)
    numda_two = (d.dot(Hessian(x))).dot(d)
    # print(numda_two)

    index = 0

    while 0.5 * numda_two > epz:
        t = 1
        index += 1  # 计数，计量回溯周期
        while Function(x + t * d) >= Function(x) - a0 * t * numda_two:
            t = b0 * t

        x = x + t * d
        d = return_D(Hessian(x), A, x)
        numda_two = (d.dot(Hessian(x))).dot(d)
        # print(x)
        # print(Function(x))
        # print(numda_two)
    print("x*:")
    print(x)
    print("v*:")
    print(return_V(Hessian(x), A, x))
    print("p*")
    print(Function(x))
    time_end = time.time()
    mid_time = time_end - time_start
    print("total time:")
    print(mid_time)
    print("per time:")
    print(mid_time / index)


def r(x, v, A, b):
    tempx = Jacobian(x) + A.T.dot(v)
    tempy = A.dot(x) - b
    return np.hstack((tempx, tempy))


def return_dy(hessianx, x, v, A, b):
    dy = -1 * np.linalg.inv(Matrix_h_s_stack(hessianx, A)).dot(r(x, v, A, b))
    return dy[0:100], dy[100:130]


def Unailable_point_newton_method(x, v, A, b, a0, b0, epz):
    time_start = time.time()
    dx, dv = return_dy(Hessian(x), x, v, A, b)
    # print(A.dot(x + dx) - b)

    index = 0
    while np.linalg.norm(r(x, v, A, b)) > epz:
        t = 1
        index += 1
        tempx = np.linalg.norm(r(x + t * dx, v + t * dv, A, b))
        tempy = (1 - a0 * t) * np.linalg.norm(r(x, v, A, b))
        while tempx > tempy:
            t = b0 * t

        x = x + t * dx
        v = v + t * dv
        dx, dv = return_dy(Hessian(x), x, v, A, b)
        # print(x)
        # print(Function(x))
    print("x*:")
    print(x)
    print("v*:")
    print(v)
    print("p*")
    print(Function(x))
    time_end = time.time()
    mid_time = time_end - time_start
    print("total time:")
    print(mid_time)
    print("per time:")
    print(mid_time / index)


def Function_star(y):
    result = 0
    for i in range(0, 100):
        result += np.exp(y[i] - 1)

    return result


def Jacobin_Function_star(y):
    the_list = []
    for i in range(0, 100):
        the_list.append(np.exp(y[i] - 1))
    return np.array(the_list)


def Hessian_Function_star(y):
    the_list = []
    for i in range(0, 100):
        the_list.append(np.exp(y[i] - 1))
    return np.diag(the_list)


def G(A, b, v):
    return b.dot(v) + Function_star(-1 * A.T.dot(v))


def G_Jacobin(A, b, v):
    return b - A.dot(Jacobin_Function_star(-1 * A.T.dot(v)))


def G_Hessian(A, v):
    return (A.dot(Hessian_Function_star(-1 * A.T.dot(v)))).dot(A.T)


def Dual_method_use_Newton(v, A, b, a0, b0, epz):
    time_start = time.time()
    d = -1 * (np.linalg.inv(G_Hessian(A, v))).dot(G_Jacobin(A, b, v))
    numda_two = (d.dot(G_Hessian(A, v))).dot(d)
    # print(G(A, b, v))
    index = 0
    while 0.5 * numda_two > epz:
        t = 1
        index += 1
        while G(A, b, v + t * d) > (G(A, b, v) - a0 * t * numda_two):
            t = b0 * t

        v = v + t * d
        # print(G(A, b, v))
        d = -1 * (np.linalg.inv(G_Hessian(A, v))).dot(G_Jacobin(A, b, v))
        numda_two = (d.dot(G_Hessian(A, v))).dot(d)

    the_list = []
    temp = A.T.dot(v)
    for i in range(0, 100):
        the_list.append(np.exp(-temp[i] - 1))
    x = np.array(the_list)
    print("x*:")
    print(x)
    print("v*:")
    print(v)
    print("p*")
    print(-G(A, b, v))
    time_end = time.time()
    mid_time = time_end - time_start
    print("total time:")
    print(mid_time)
    print("per time:")
    print(mid_time / index)


if __name__ == "__main__":
    x0 = make_X()
    v0 = np.zeros(30)
    A = make_A()
    b = A.dot(x0)

    a0 = 0.1
    b0 = 0.5

    epz = 10 ** -10
    x = x0
    v = v0
    print("A:")
    print(A)
    print("--------------------------------------")
    print("x0:")
    print(x0)
    print("--------------------------------------")
    print("b:")
    print(b)
    print("--------------------------------------")
    print("从可行点出发的newton方法")
    Available_point_newton_method(x, A, a0, b0, epz)
    print("--------------------------------------")
    x = np.ones(100)
    print("从不可行点出发的newton方法")
    Unailable_point_newton_method(x, v, A, b, a0, b0, epz)
    print("--------------------------------------")
    print("使用无约束newton方法计算对偶问题")
    Dual_method_use_Newton(v, A, b, a0, b0, epz)
