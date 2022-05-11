#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Created by lq on 2022/5/11.
"""
import os
from datetime import datetime
from functools import wraps


def make_sure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def show_run_time(print_start_end_time=True,
                  print_run_time=True,
                  print_func_name=True,
                  time_formatter='%Y-%m-%d %H:%M:%S'):
    def decorator(func):
        func_name = 'func'
        if print_func_name:
            func_name += f' "{func.__name__}"'

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            if print_start_end_time:
                print(f"start {func_name} at: {start_time.strftime(time_formatter)}")
            result = func(*args, **kwargs)
            end_time = datetime.now()
            if print_start_end_time:
                print(f"end {func_name} at: {end_time.strftime(time_formatter)}")
            if print_run_time:
                print(f"run {func_name} time: {(end_time - start_time).seconds}s")
            return result
        return wrapper
    return decorator
