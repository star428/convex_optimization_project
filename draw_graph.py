#!/usr/bin/env python
# encoding: utf-8
"""
@author: star428
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited 
@contact: yewang863@gmail.com
@software: pycharm
@file: draw_graph.py
@time: 2020/11/24 1:16
@desc:
"""
import pygal
import numpy as np
from Unconstrained_optimization_problem import Gradient_descent_method

line_chart = pygal.Line()
line_chart.title = 'the contant of the a and b'
line_chart.x_labels = map(str, range(0, 15))
x0 = np.array([1, 1])
for i in range(1, 10, 1):
    the_list = Gradient_descent_method(x0, 0.05 * i, 0.5, 0.001)
    line_chart.add(the_list[0], the_list[1])

line_chart.render_to_file('the available is a.svg')

line_chart_two = pygal.Line()
line_chart_two.title = "the contant of the a and b"
line_chart_two.x_labels = map(str, range(0, 25))
for i in range(1, 10, 1):
    the_list = Gradient_descent_method(x0, 0.1, 0.1 * i, 0.001)
    line_chart_two.add(the_list[0], the_list[1])

line_chart_two.render_to_file('the available is b.svg')
