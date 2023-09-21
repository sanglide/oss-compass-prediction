import json

import numpy as np
from joblib import delayed, Parallel
from numba import njit, prange
from numba.typed import List
from numpy.lib.stride_tricks import as_strided
from tqdm import tqdm
import pandas as pd

import util

from fastdtw import fastdtw


def log_mined_shapelets(shapelets, scores, info_gains, start_idx, end_idx, series_idx, repo_name_list,
                        repo_label_list, n_shapelets, window_size, window_step, n_channels, label_period_months,
                        forecast_gap_months, data_period_months, data_path, result_file, evolution_event_selection):
    n_shape = len(shapelets[0])  # the real number of shapes from channel 0

    if n_shapelets != n_shape:
        print(f"warning : n_shapelets!=n_shape {n_shapelets} {n_shape}")

    doc = {"n_shapelets": n_shapelets, "window_size": window_size,
           "window_step": window_step, "n_channels": n_channels, "data_path": data_path,
           "label_period_months": label_period_months, "forecast_gap_months": forecast_gap_months,
           "data_period_months": data_period_months,
           "evolution_event_selection": np.array(evolution_event_selection).tolist(), "shapelets": []}
    for i in range(n_shape):
        shape_doc = {"score": scores[i], "info_gain": info_gains[i], "start_idx": start_idx[i], "end_idx": end_idx[i],
                     "from_repo_idx": series_idx[i], "from_repo_name": repo_name_list[int(series_idx[i])],
                     "from_repo_label": repo_label_list[int(series_idx[i])],
                     "shape_split": np.array(shapelets[0][i]).tolist(),
                     "shape_shrink": np.array(shapelets[1][i]).tolist(),
                     "shape_merge": np.array(shapelets[2][i]).tolist(),
                     "shape_expand": np.array(shapelets[3][i]).tolist()}
        doc["shapelets"].append(shape_doc)
    json_doc = json.dumps(doc, default=util.default_dump)
    # print(f"{json_doc[:250]} ...")
    with open(result_file, "w") as f:
        f.write(json_doc)


def load_mined_shapelets(shapelet_file_path):
    with open(shapelet_file_path, "r") as f:
        loaded_shape_dict = json.load(f)
    return loaded_shape_dict


def extract_shapelets_from_dict(loaded_shape_dict):
    n_shapelets = loaded_shape_dict["n_shapelets"]
    n_channels = loaded_shape_dict["n_channels"]
    ret_shapelets = [[] for ch in range(n_channels)]
    for idx, shape_dict in enumerate(loaded_shape_dict["shapelets"]):
        ret_shapelets[0].append(np.array(shape_dict["shape_split"]))
        ret_shapelets[1].append(np.array(shape_dict["shape_shrink"]))
        ret_shapelets[2].append(np.array(shape_dict["shape_merge"]))
        ret_shapelets[3].append(np.array(shape_dict["shape_expand"]))
    # assert n_shapelets == len(ret_shapelets[0]) == len(ret_shapelets[1]) == len(ret_shapelets[2]) == len(ret_shapelets[3])
    assert len(ret_shapelets[0]) == len(ret_shapelets[1]) == len(ret_shapelets[2]) == len(ret_shapelets[3])
    # print(ret_shapelets)
    return np.array(ret_shapelets), len(ret_shapelets[0])


@njit()
def _extract_channel_shapelets(x_channel, window_size, window_step, x_seq_length):
    overlap = window_size - window_step
    shape_new = (
        (x_seq_length - overlap) // window_step, window_size // 1)  # shape_new = (number of windows, window size)
    # print("------line 99------")
    strides = x_channel.strides[0]  # bytes of a single element in the array
    strides_new = (window_step * strides, strides)
    # print("------line 102------")
    # Derive strided view of x_channel
    x_strided = as_strided(x_channel, shape=shape_new, strides=strides_new)
    # print("------line 105------")
    # print(x_strided)
    # x_strided=np.nan_to_num(x_strided)
    # print(x_strided)
    # print("------line 106------")
    shapelets = np.copy(x_strided)  # shapelets
    # print("------line 107------")
    start_idx = np.arange(0, x_seq_length - window_size + 1, window_step)  # start index of shapelets (included)
    end_idx = np.arange(window_size, x_seq_length + 1, window_step)  # end index of shapelets (excluded)
    return shapelets, start_idx, end_idx


@njit()
def _extract_all_shapelets_mv(x, window_size, window_step, n_channels, x_seq_length):
    def scale_to_max(data_seq, max_value, ratio):
        scale_seq = []
        for d in data_seq:
            scale_seq.append(ratio * d / max_value if max_value > 0 else 0)
        return np.array(scale_seq)

    """Extract all the shapelets from a single time series."""
    # x[channel_idx][data_point_idx]
    # print("---------------Enter extract one --------------------")
    shapelets = List()  # shapelets[channel][shapelet_idx][data_point_idx]
    shapelets_scaled = List()
    for channel_index in prange(n_channels):
        # 这里的覆盖很正常
        x_temp, start_idx, end_idx = _extract_channel_shapelets(x[channel_index], window_size,
                                                                window_step, x_seq_length)
        shapelets.append(x_temp)
    # perform local scale to the candidate shapelets
    if global_settings.DO_LOCAL_SCALE:
        n_candidates = len(shapelets[0])
        for channel_index in prange(n_channels):
            x_temp_scaled = np.zeros((n_candidates, window_size))
            for shape_idx in range(n_candidates):
                max_value = max([max(shapelets[ch][shape_idx]) for ch in range(n_channels)])
                x_temp_scaled[shape_idx] = scale_to_max(shapelets[channel_index][shape_idx], max_value, 10.)
            shapelets_scaled.append(x_temp_scaled)
        return shapelets_scaled, start_idx, end_idx
    else:
        return shapelets, start_idx, end_idx


@njit()
def dtw_distance(sequence1, sequence2):
    """
    This function calculates the Dynamic Time Warping (DTW) distance between two sequences.

    :param sequence1: First sequence of numbers
    :param sequence2: Second sequence of numbers
    :return: DTW distance between the two sequences
    """
    return fastdtw(np.array(sequence1).tolist(), np.array(sequence2).tolist())


@njit()
def _distance_func_squared(window, shapelet, n_channels, relative_position, evolution_event_selection):
    # window[channel][data_point_idx]
    # shapelet[channel][data_point_idx]
    dist = 0
    for i in range(n_channels):
        if not i in evolution_event_selection:
            continue
        if global_settings.USE_ABS_DIST:
            temp_dist = 0
            for idx, _ in enumerate(window[i]):
                temp_dist += np.abs(window[i][idx] - shapelet[i][idx])
            temp_dist = temp_dist / len(window[i])
            dist += temp_dist
        else:
            # mean squared distance is used here
            dist = dist + np.mean(np.sqrt((window[i] - shapelet[i]) ** 2))
    if global_settings.DO_POSITION_PENALTY:
        p_ratio = 1 + (1 - relative_position) * (global_settings.MAX_POSITION_PENALTY_RATE - 1)
        return dist * p_ratio
    else:
        return dist


@njit()
def _distances_single_mv(x, window_size, shapelets, n_channels, x_seq_length, evolution_event_selection):
    # the distance of a single multivarient time series with all shapelets
    # x[channel][data_point_idx]
    # shapelets[channel][shape_idx]
    assert (window_size == len(shapelets[0][0]))
    assert (window_size <= x_seq_length)
    # extract all windows in x

    windows, start_idx, end_idx = _extract_all_shapelets_mv(x, window_size, 1, n_channels, x_seq_length)
    n_windows = len(windows[0])  # count of windows in channel 0
    n_shapelets = len(shapelets[0])  # count of shapelets in channel 0
    dist_all_shapelets = []
    for shape_idx in prange(n_shapelets):
        # distance of a single shapelet
        shapelet = [shapelets[ch][shape_idx] for ch in prange(n_channels)]
        dist_shapelet = []
        for win_idx in prange(n_windows):
            win = [windows[ch][win_idx] for ch in prange(n_channels)]
            relative_position = start_idx[win_idx] / (x_seq_length - window_size) if x_seq_length > window_size else 0
            # the relative position increases from 0 to 1 with increasing start idx, a relative position of 1 means
            # the window is the latest one
            dist_shapelet.append(
                _distance_func_squared(win, shapelet, n_channels, relative_position, evolution_event_selection))
        min_dist = np.min(np.asarray(dist_shapelet))
        dist_all_shapelets.append(min_dist)
    return dist_all_shapelets


@njit()
def _distances_all_mv(X, window_size, shapelets, n_channels, X_valid_length_list, evolution_event_selection):
    # X[channel][seq_idx][data_point_idx]
    # len(X[0]) for the number of multivariate series in channel 0 of X
    # len(shapelets[0]) for the number of shapelets in channel 0
    n_series = len(X[0])
    dist_matrix = []
    for i in prange(n_series):
        series = [X[ch][i] for ch in prange(n_channels)]
        # dist_vector[shapelet_idx]
        dist_vector = _distances_single_mv(series, window_size, shapelets, n_channels, X_valid_length_list[i],
                                           evolution_event_selection)
        dist_matrix.append(dist_vector)
    # dist_matrix[series_idx][shapelet_idx]
    return dist_matrix


@njit()
def _remove_overlap_shapelets(scores, info_gains, start_idx, end_idx):
    """Remove self-similar shapelets."""
    # todo:改成从中间remove
    kept_idx = []
    remaining_idx = np.full(info_gains.size, True)

    # Sort the indices by scores
    argsorted_scores = np.argsort(info_gains)
    sorted_start_idx = start_idx[argsorted_scores]
    sorted_end_idx = end_idx[argsorted_scores]

    # While there are non-similar shapelets remaining
    while np.any(remaining_idx):
        idx = argsorted_scores[remaining_idx][-1]  # Find the best shapelet
        kept_idx.append(idx)

        start, end = start_idx[idx], end_idx[idx]  # Find start and end indices

        # Find non-similar shapelets
        remaining_idx = np.logical_and(
            np.logical_or(sorted_start_idx >= end, sorted_end_idx <= start),
            remaining_idx
        )

    return np.array(kept_idx)


def _keep_top_shapelets_idx(ratios, info_gains, n_shapelets):
    '''
     - 从两端选取（active和inactive各选择n_shapelets/2个）
     - 如果某一类的shapelets数量不够，则最终返回小于n_shapelets个shapelets
     - 约束：以1为分界线，选出的shapelets不能交错，不能越过1的界限
     - 约束：在最外层选择shapelets处，若无足够的shapelets，则报错
    '''
    # Sort the indices by scores
    # choose shapelets using distance algorithm
    # [2 3 1 0.1 0.2]-->[0.1 0.2 1 2 3] arrIndex=[3 4 2 1 0]
    arrIndex = np.array(ratios).argsort()
    arr_small, arr_big = [], []
    info_gains_small, info_gains_big = [], []
    arr_small_idx, arr_big_idx = [], []

    ratios = ratios[arrIndex]  # todo:test this line. correct?
    assert len(ratios) == len(info_gains)
    info_gains = info_gains[arrIndex]

    for i in range(len(ratios)):
        if ratios[i] > 1:
            arr_big.append(ratios[i])
            arr_big_idx.append(arrIndex[i])
            info_gains_big.append(info_gains[i])
        elif ratios[i] < 1:
            arr_small.append(ratios[i])
            arr_small_idx.append(arrIndex[i])
            info_gains_small.append(info_gains[i])
        else:
            continue
    # [0.1 0.2] [3 4] and [2 3] [1 0]
    num = int(n_shapelets / 2)

    # todo: 使用information gain作为选择最优shapelet的标准，可以加一个参数标志位进行方法选择

    # arr_big_idx_re = arr_big_idx[len(arr_big_idx) - num:len(arr_big_idx)] if len(arr_big_idx) >= num else arr_big_idx
    # arr_small_idx_re = arr_small_idx[:num] if len(arr_small_idx) >= num else arr_small_idx
    # arr_small_idx_re.extend(arr_big_idx_re)
    # assert len(arr_small_idx_re)>0
    # assert len(arr_big_idx_re)>0

    index_small = np.argsort(np.array(info_gains_small))
    index_big = np.argsort(np.array(info_gains_big))

    try:
        arr_small_idx_re = np.array(arr_small_idx)[index_small[len(index_small) - num:len(index_small)]] if len(
            index_small) >= num else np.array(arr_small_idx)
        arr_small_idx_re = arr_small_idx_re.tolist()
    except:
        print(f"arr_small_idx {arr_small_idx}")
        print(f"index_small {index_small}")
        print(
            f"index_small[len(index_small) - num:len(index_small)] {index_small[len(index_small) - num:len(index_small)]}")
        assert False
    arr_big_idx_re = np.array(arr_big_idx)[index_big[len(index_big) - num:len(index_big)]] if len(
        index_big) >= num else np.array(arr_big_idx)
    arr_big_idx_re = arr_big_idx_re.tolist()
    count_small = len(arr_small_idx_re)
    count_big = len(arr_big_idx_re)
    arr_small_idx_re.extend(arr_big_idx_re)
    return arr_small_idx_re, count_small, count_big

    # argsorted_scores = np.argsort(scores)
    # return argsorted_scores[-n_shapelets:]


# 定义C4.5算法所需的函数
# 定义C4.5算法所需的函数
def calc_entropy(data):
    """计算信息熵"""
    counts = data.value_counts()  # 统计各类别样本的数量
    probs = counts / len(data)  # 计算各类别样本的概率
    entropy = -sum(probs * np.log2(probs))  # 根据公式计算信息熵
    return entropy


def calc_conditional_entropy(data, feature, threshold):
    """计算条件熵"""
    coll = data.columns
    low_subset = data[data[feature] < threshold][coll[-1]]  # 获取label小于阈值的目标变量列
    print(low_subset)
    high_subset = data[data[feature] >= threshold][coll[-1]]  # 获取label大于等于阈值的目标变量列
    prob_low = len(low_subset) / len(data)  # 计算前一部分的出现概率
    prob_high = len(high_subset) / len(data)  # 计算后一部分的出现概率
    entropy = prob_low * calc_entropy(low_subset) + prob_high * calc_entropy(high_subset)  # 计算条件熵
    return entropy


def calc_information_gain(data, feature):
    """计算信息增益"""
    coll = data.columns
    base_entropy = calc_entropy(data[coll[-1]])  # 计算全局信息熵
    sorted_values = sorted(data[feature].unique())  # 对连续属性的取值进行排序
    thresholds = [(sorted_values[i] + sorted_values[i + 1]) / 2 for i in range(len(sorted_values) - 1)]
    # 计算各个分割点的阈值，例如200个数据点共有199个分割点
    info_gains = []
    for threshold in thresholds:
        cond_entropy = calc_conditional_entropy(data, feature, threshold)
        info_gain = base_entropy - cond_entropy
        info_gains.append(info_gain)
    best_threshold_index = np.argmax(info_gains)  # 找到信息增益最大的分割点
    best_threshold = thresholds[best_threshold_index]  # 找到对应的阈值
    return info_gains[best_threshold_index], best_threshold


def select_best_feature(data, features):
    """选择最佳特征"""
    info_gains = [calc_information_gain(data, feature) for feature in features]  # 计算每个特征的信息增益和最佳分割点
    best_feature_index = np.argmax([gain for gain, threshold in info_gains])  # 找到信息增益最大的特征
    best_feature = features[best_feature_index]  # 找到对应的属性名
    best_info_gain_and_threshold = info_gains[best_feature_index]
    # best_threshold = info_gains[best_feature_index][1]  # 找到对应的最佳分割点
    return best_feature, best_info_gain_and_threshold, info_gains


def _compute_info_gain(dist_matrix, y):
    df = pd.DataFrame(dist_matrix)
    clm = ["label"]
    y_df = pd.DataFrame(y, columns=clm)
    dataset = pd.concat([df, y_df], axis=1)
    best_feature, est_info_gain_and_threshold, info_gains = select_best_feature(dataset, df.columns[1:])
    print(best_feature)
    print(est_info_gain_and_threshold)
    scores = [info_gains[i][0] for i in range(len(info_gains))]
    return scores


@njit()
def calculate_information_gain_2(feature, labels):
    """
    This function calculates the information gain given a numerical feature and the corresponding labels.
    """

    if global_settings.DO_REMOVE_ZERO_DIST:
        labels = labels[feature != 0]
        feature = feature[feature != 0]

    # Calculate the entropy of the original dataset
    total_count = len(labels)
    label_counts_dict = {}
    for label in labels:
        if label in label_counts_dict:
            label_counts_dict[label] += 1
        else:
            label_counts_dict[label] = 1
    unique_labels = list(label_counts_dict.keys())
    label_counts = list(label_counts_dict.values())

    entropy = 0
    for count in label_counts:
        probability = count / total_count
        entropy -= probability * np.log2(probability)

    # Sort the feature values in ascending order
    sorted_feature = np.sort(feature)

    # Initialize variables to keep track of the best split and its information gain
    best_split = None
    best_gain = 0

    # Iterate through all possible split points
    for i in range(1, len(sorted_feature)):
        # Calculate the information gain for this split
        split_value = (sorted_feature[i - 1] + sorted_feature[i]) / 2
        left_labels = labels[feature <= split_value]
        right_labels = labels[feature > split_value]
        left_count = len(left_labels)
        right_count = len(right_labels)
        if right_count == 0:
            break
        left_entropy = 0
        right_entropy = 0
        for label in unique_labels:
            left_probability = np.sum(left_labels == label) / left_count
            right_probability = np.sum(right_labels == label) / right_count
            if left_probability > 0:
                left_entropy -= left_probability * np.log2(left_probability)
            if right_probability > 0:
                right_entropy -= right_probability * np.log2(right_probability)
        information_gain = entropy - (left_count / total_count * left_entropy) - (
                right_count / total_count * right_entropy)

        # Update the best split and its information gain if necessary
        if information_gain > best_gain:
            best_split = split_value
            best_gain = information_gain

    return best_gain


@njit()
def _compute_info_gain_2(dist_matrix, y):
    n_shapelets = len(dist_matrix[0])
    n_series = len(y)
    scores = []
    for idx_shape in range(n_shapelets):
        features = []
        for idx_series in range(n_series):
            features.append(dist_matrix[idx_series][idx_shape])
        best_gain = calculate_information_gain_2(np.array(features), y)
        scores.append(best_gain)
    return scores


@njit()
def _compute_scores(dist_matrix, y):
    n_shapelets = len(dist_matrix[0])
    scores = []
    n_labels = [np.sum(y == 0), np.sum(y == 1)]
    shape_labels = [-1 for i in range(n_shapelets)]
    dist_shape = np.zeros((n_shapelets, 2))
    for idx_series in range(len(y)):
        y_label = y[idx_series]
        assert y_label == 0 or y_label == 1
        for idx_shape in range(n_shapelets):
            # 每一个仓库，每一个shapelets，对某个标签进行距离求和
            dist_shape[idx_shape][y_label] += dist_matrix[idx_series][idx_shape]
            if global_settings.DO_REMOVE_ZERO_DIST:
                if dist_matrix[idx_series][idx_shape] < 0.0000000000001:
                    if shape_labels[idx_shape] == -1:
                        shape_labels[idx_shape] = y_label
                    else:
                        print(dist_matrix)
                        assert False
    assert n_labels[0] != 0 and n_labels[1] != 0
    dist_ratio_shape = [0.0 for i in range(n_shapelets)]

    for idx_shape in range(n_shapelets):
        if global_settings.DO_REMOVE_ZERO_DIST:
            dist_shape[idx_shape][0] = float(dist_shape[idx_shape][0]) / float(
                n_labels[0] - 1 if shape_labels[idx_shape] == 0 else n_labels[0])
            dist_shape[idx_shape][1] = float(dist_shape[idx_shape][1]) / float(
                n_labels[1] - 1 if shape_labels[idx_shape] == 1 else n_labels[1])
        else:
            dist_shape[idx_shape][0] = float(dist_shape[idx_shape][0]) / float(n_labels[0])
            dist_shape[idx_shape][1] = float(dist_shape[idx_shape][1]) / float(n_labels[1])
        assert dist_shape[idx_shape][0] != 0
        assert dist_shape[idx_shape][1] != 0
        if dist_shape[idx_shape][1] == 0:
            dist_ratio_shape[idx_shape] = 999999
        else:
            dist_ratio_shape[idx_shape] = float(dist_shape[idx_shape][0]) / float(dist_shape[idx_shape][1])
        scores.append(dist_ratio_shape[idx_shape])
    return scores


class ShapeletsMV():

    def __init__(self):
        self.verbose = True

    def fit_all(self, X, y, n_shapelets, window_size, window_step, n_channels, X_valid_length_list, n_jobs,
                evolution_event_selection, remove_overlap=False):
        # mine shapelets in the training data set
        n_series = len(X[0])  # number of multivariate time series from channel 0
        # mine all in parallel
        # i就是repo在变
        res = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(self._fit_one)(i, X, y, n_shapelets, window_size, window_step, n_channels, X_valid_length_list,
                                   evolution_event_selection, remove_overlap)
            for i in tqdm(range(n_series)))
        print("--------------------Finish _fit_one-------------------------")

        # unpack the results
        (shapelets, scores, info_gains, start_idx, end_idx, series_idx) = zip(*res)
        scores = np.concatenate(scores)
        shapelets = np.concatenate(shapelets, axis=1)
        if shapelets.ndim > 2:
            shapelets = [shapelets[channel_index].astype('float64') for channel_index in range(n_channels)]
        start_idx = np.concatenate(start_idx)
        end_idx = np.concatenate(end_idx)
        series_idx = np.concatenate(series_idx)
        info_gains = np.concatenate(info_gains)

        # select the top n_shapelets
        kept_idx, count_small, count_big = _keep_top_shapelets_idx(scores, info_gains, n_shapelets)
        print(f"\n@@@@ count_small {count_small}, count_big {count_big}, kept_idx {kept_idx}")
        shapelets = [shapelets[ch][kept_idx] for ch in range(n_channels)]
        scores = scores[kept_idx]
        start_idx = start_idx[kept_idx]
        end_idx = end_idx[kept_idx]
        series_idx = series_idx[kept_idx]
        return shapelets, scores, info_gains, start_idx, end_idx, series_idx

    def _fit_one(self, i, X, y, n_shapelets, window_size, window_step, n_channels, X_valid_length_list,
                 evolution_event_selection, remove_overlap=True):
        x = np.array([X[ch][i] for ch in prange(n_channels)])
        shapelets, start_idx, end_idx = _extract_all_shapelets_mv(x, window_size, window_step, n_channels,
                                                                  X_valid_length_list[i])


        dist_matrix = _distances_all_mv(X, window_size, shapelets, n_channels, X_valid_length_list,
                                        evolution_event_selection)

        dist_matrix = np.array(dist_matrix)

        ratios = _compute_scores(dist_matrix, y)


        # scores1, _ = f_classif(dist_matrix, y)
        ratios = np.nan_to_num(ratios)

        info_gains = _compute_info_gain_2(dist_matrix, y)
        info_gains = np.nan_to_num(info_gains)

        if remove_overlap:
            idx = _remove_overlap_shapelets(ratios.copy(), info_gains.copy(), start_idx, end_idx)
            ratios = ratios[idx]
            info_gains = info_gains[idx]
            shapelets = [shapelets[ch][idx] for ch in range(n_channels)]
            start_idx = start_idx[idx]
            end_idx = end_idx[idx]
            # dist_matrix = dist_matrix[:, idx]
        # print(shapelets)
        kept_idx, _, _ = _keep_top_shapelets_idx(ratios, info_gains, n_shapelets)
        try:
            shapelets = [shapelets[ch][kept_idx] for ch in range(n_channels)]
            print(kept_idx)
        except:
            assert False
        ratios = ratios[kept_idx]
        start_idx = start_idx[kept_idx]
        end_idx = end_idx[kept_idx]
        series_idx = np.array([i for k in range(len(kept_idx))])
        info_gains = info_gains[kept_idx]
        return shapelets, ratios, info_gains, start_idx, end_idx, series_idx


if __name__ == '__main__':
    x1 = [[i for i in range(10)], [j for j in range(20, 30)]]
    x2 = [[i for i in range(1, 11)], [j for j in range(21, 31)]]
    x3 = [[i for i in range(2, 12)], [j for j in range(22, 32)]]
    X = [[x1[0], x2[0], x3[0]], [x1[1], x2[1], x3[1]]]
    x1 = np.array(x1)
    x2 = np.array(x2)
    x3 = np.array(x3)
    X = np.array(X)

    sh = ShapeletsMV()
    shapelets, scores, start_idx, end_idx, series_idx = sh.fit_all(X, np.array([0, 1, 0]), 5, 10, 1, 2,
                                                                   np.array([100, 100, 100]), 4)
    print(shapelets)
    print(scores)
    print(start_idx)
    print(end_idx)
    print(series_idx)
