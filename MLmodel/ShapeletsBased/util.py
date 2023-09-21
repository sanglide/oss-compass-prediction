import datetime
import os
import sys

import numpy as np


def list_to_str(list_of_int):
    sss = ""
    for it in list_of_int:
        sss += str(it)
    return sss

def get_shapelet_file_name(label_period_months, forecast_gap_months, data_period_months,
                           n_shapelets, window_size, window_step, n_channels, data_path, evolution_event_selection):

    # record mining results
    if "index_productivity_32" in data_path:
        filename = "d_32"
    else:
        filename = "d_696"
    return f"l_{label_period_months}_f_{forecast_gap_months}_d_{data_period_months}_ns_{n_shapelets}_wsz_{window_size}_wsp_{window_step}_nc_{n_channels}_{filename}_select_{list_to_str(evolution_event_selection)}.json"


def get_result_root_dir():
    if 'linux' in sys.platform or 'Linux' in sys.platform:
        root_dir_path = "/home/wangliang/workspace/result/oss_community_evolution_shapelets/"
    else:
        root_dir_path = "../000_shapelet_results/result/"
    return root_dir_path


def get_data_dir():
    if 'linux' in sys.platform or 'Linux' in sys.platform:
        data_dir_path = "/home/wangliang/workspace/data/wl_shapelet_data/"
    else:
        data_dir_path = "./data/"
    return data_dir_path


def default_dump(obj):
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def test_kill_and_exit():
    if os.path.exists("../kill"):
        os.remove("../kill")
        f = open("../killed", "w")
        f.write(str(datetime.datetime.now()))
        f.close()
        sys.exit(-1)
