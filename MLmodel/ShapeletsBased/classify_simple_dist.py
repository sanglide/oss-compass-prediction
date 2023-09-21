import os
import pickle
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

import shapelets as sp
import numpy as np

import util
from plot_over import add_font_in_pic


class ClassifySimpleDist():

    def __init__(self):
        self.shapelet_file_path = None
        self.loaded_shape_dict = None
        self.shapelets = None
        self.n_shapelets = 0
        self.window_size = 0
        self.n_channels = 0
        self.plot_output_root_dir = None
        self.classify_threshold = 0

    def set_init_parameters(self, shapelet_file_path, plot_output_root_dir):
        self.shapelet_file_path = shapelet_file_path
        self.loaded_shape_dict = sp.load_mined_shapelets(shapelet_file_path)
        self.shapelets, self.n_shapelets = sp.extract_shapelets_from_dict(self.loaded_shape_dict)
        # self.n_shapelets = self.loaded_shape_dict["n_shapelets"]
        if self.n_shapelets < self.loaded_shape_dict["n_shapelets"]:
            print(
                f"warning : self.n_shapelets<self.loaded_shape_dict[\"n_shapelets\"] {self.n_shapelets} {self.loaded_shape_dict['n_shapelets']}")
        self.window_size = self.loaded_shape_dict["window_size"]
        self.n_channels = self.loaded_shape_dict["n_channels"]
        self.plot_output_root_dir = plot_output_root_dir
        self.evolution_event_selection = self.loaded_shape_dict["evolution_event_selection"]

    def log_label(self):
        result_png_root_path = f"{self.plot_output_root_dir}label/"
        if not os.path.exists(result_png_root_path):
            os.makedirs(result_png_root_path)

        n_shapelets = self.loaded_shape_dict["n_shapelets"]
        window_size = self.loaded_shape_dict["window_size"]
        window_step = self.loaded_shape_dict["window_step"]
        n_channels = self.loaded_shape_dict["n_channels"]
        data_path = self.loaded_shape_dict["data_path"]
        label_period_months = self.loaded_shape_dict["label_period_months"]
        forecast_gap_months = self.loaded_shape_dict["forecast_gap_months"]
        data_period_months = self.loaded_shape_dict["data_period_months"]

        png_filename_prefix = util.get_shapelet_file_name(label_period_months, forecast_gap_months, data_period_months,
                                                          n_shapelets, window_size, window_step, n_channels, data_path)

        for idx, shape_dict in enumerate(self.loaded_shape_dict["shapelets"]):
            content = f"label: {str(self.shapelet_labels[idx])} {self.dist_shape[idx]}"
            png_path = f"{self.plot_output_root_dir}{png_filename_prefix}_{idx}.png"
            result_png_path = f"{self.plot_output_root_dir}label/{png_filename_prefix}_{idx}.png"
            add_font_in_pic(png_path, result_png_path, content)

    def train(self, X, X_valid_length_list, y):
        # todo：目前丢掉了shapelet的判别能力强弱之分（将离散值二元化了）

        # get distance matrix, dist_matrix[series_idx][shapelet_idx], labels of the series are listed in y
        dist_matrix = sp._distances_all_mv(X, self.window_size, self.shapelets, self.n_channels, X_valid_length_list,
                                           self.evolution_event_selection)
        n_series = len(X_valid_length_list)
        # dist_shape[shape_idx][label=0/1]
        # 某一个shapelets和成功/失败的距离
        self.dist_shape = np.zeros((self.n_shapelets, 2))
        # active和inactive的比值
        self.dist_ratio_shape = [0 for i in range(self.n_shapelets)]
        # shapelets最终对应的label
        self.shapelet_labels = [-1 for i in range(self.n_shapelets)]
        # active和inactive的数量
        self.n_labels = [0, 0]
        self.n_shapelet_labels = [0, 0]
        assert len(X_valid_length_list) == len(y)
        for idx_series in range(n_series):
            y_label = y[idx_series]
            assert y_label == 0 or y_label == 1
            for idx_shape in range(self.n_shapelets):
                # 每一个仓库，每一个shapelets，对某个标签进行距离求和
                self.dist_shape[idx_shape][y_label] += dist_matrix[idx_series][idx_shape]
                self.n_labels[y_label] += 1
        assert self.n_labels[0] != 0 and self.n_labels[1] != 0
        for idx_shape in range(self.n_shapelets):
            self.dist_shape[idx_shape][0] /= self.n_labels[0]
            self.dist_shape[idx_shape][1] /= self.n_labels[1]
            assert self.dist_shape[idx_shape][0] != 0 and self.dist_shape[idx_shape][1] != 0
            # assert self.dist_shape[idx_shape][0] != self.dist_shape[idx_shape][1]
            if self.dist_shape[idx_shape][0] == self.dist_shape[idx_shape][1]:
                print(
                    f"======self.dist_shape[idx_shape][0] != self.dist_shape[idx_shape][1]======={self.dist_shape[idx_shape][0]} , {self.dist_shape[idx_shape][1]}")
                for ch in range(self.n_channels):
                    print(self.shapelets[ch][idx_shape])
            # ratio < 1 = inactive
            # ratio > 1 = active
            self.dist_ratio_shape[idx_shape] = self.dist_shape[idx_shape][0] / self.dist_shape[idx_shape][1]
            if self.dist_ratio_shape[idx_shape] < 1:
                self.shapelet_labels[idx_shape] = 0
                self.n_shapelet_labels[0] += 1
            else:
                self.shapelet_labels[idx_shape] = 1
                self.n_shapelet_labels[1] += 1

    def report(self, test_repos, y_true, y_pred, label_period_months, forecast_gap_months, data_period_months,
               result_folder_path, evolution_event_selection,
               report_file_name="prediction_report.csv", detailed_file_name="prediction_report_details.csv"):
        c = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True)
        re = [[label_period_months, forecast_gap_months, data_period_months, self.window_size, self.classify_threshold, evolution_event_selection,
               c[0][0], c[0][1], c[1][0], c[1][1],
               report["0"]["precision"], report["0"]["recall"], report["0"]["f1-score"], report["0"]["support"],
               report["1"]["precision"], report["1"]["recall"], report["1"]["f1-score"], report["1"]["support"],
               (c[0][0] + c[1][1]) / (c[0][0] + c[1][1] + c[1][0] + c[0][1])]]
        print(re)
        dd = pd.DataFrame(data=re, columns=["label_period_months", "forecast_gap_months",
                                            "data_period_months", "window_size", "classify_threshold", "evolution_event_selection", "cm0_0", "cm0_1",
                                            "cm1_0", "cm1_1",
                                            "cr0_precision", "cr0_recall", "cr0_f1", "cr0_support",
                                            "cr1_precision", "cr1_recall", "cr1_f1", "cr1_support", "accuracy"])
        re_path = result_folder_path + report_file_name
        if not os.path.exists(re_path):
            dd.to_csv(re_path, mode='a', index=False, index_label=False)
        else:
            dd.to_csv(re_path, mode='a', index=False, index_label=False, header=False)

        if not os.path.exists(result_folder_path + detailed_file_name):
            with open(result_folder_path + detailed_file_name, "w") as ff:
                ff.write("repo,label,true_cnt,false_cnt\n")
                ff.write("xxx,1,1,1\n")
        detailed_data = pd.read_csv(result_folder_path + detailed_file_name)
        for index, repo in enumerate(test_repos):
            label = y_true[index]
            predicted = y_pred[index]
            idx = 0
            for known_idx, known_repo in enumerate(detailed_data['repo']):
                if repo == known_repo:
                    idx = known_idx
                    break
            is_correct = label == predicted
            if idx == 0:
                detailed_data = detailed_data.append({'repo': repo, 'label': label, 'true_cnt': 1 if is_correct else 0,
                                                      'false_cnt': 0 if is_correct else 1}, ignore_index=True)
            else:
                if label != detailed_data.iloc[idx, 1]:
                    print(
                        f"XXXX {repo} {label} {detailed_data.iloc[idx, 1]} {label_period_months} {forecast_gap_months} {data_period_months}")
                    assert label == detailed_data.iloc[idx, 1]
                if is_correct:
                    detailed_data.iloc[idx, 2] += 1
                else:
                    detailed_data.iloc[idx, 3] += 1
        detailed_data.to_csv(result_folder_path + detailed_file_name, index=False)

    def get_distance_matrix(self, X, X_valid_length_list, evolution_event_selection):
        return sp._distances_all_mv(X, self.window_size, self.shapelets, self.n_channels, X_valid_length_list,
                                    evolution_event_selection)

    def get_window_size(self):
        return self.window_size

    def get_shapelet_label(self, idx_shape):
        return self.shapelet_labels[idx_shape]

    def classify(self, X, X_valid_length_list, y, threshold):
        # 分类函数，X与shapelets算距离，看看是离1还是0近
        # 如果挖出来的shapelets有成败、并且好解释，那么挖掘就是有意义的
        # 如果结果是能分类的（不追求分数很高，但是至少比胡猜好）
        # get distance matrix, dist_matrix[series_idx][shapelet_idx], labels of the series are listed in y
        # print(f"windowe size {self.window_size}")
        # print(f"n shapelets {self.n_shapelets}")
        # print(f"shapelets {self.shapelets}")
        #
        # print(X)
        # print(X_valid_length_list)

        self.classify_threshold = threshold

        dist_matrix = sp._distances_all_mv(X, self.window_size, self.shapelets, self.n_channels, X_valid_length_list)
        n_test_series = len(y)
        y_pred = [-1 for i in range(n_test_series)]
        for idx_series in range(n_test_series):
            dist_series_label = [0, 0]
            for idx_shape in range(self.n_shapelets):
                label_shape = self.shapelet_labels[idx_shape]
                # average distance to shapelets for active and inactive cases
                # large room for improvements
                dist_series_label[label_shape] += dist_matrix[idx_series][idx_shape]
            dist_series_label[0] /= self.n_shapelet_labels[0]
            dist_series_label[1] /= self.n_shapelet_labels[1]
            assert dist_series_label[0] != 0
            assert dist_series_label[1] != 0
            assert dist_series_label[0] != dist_series_label[1]
            y_pred[idx_series] = 0 if dist_series_label[0] / dist_series_label[
                1] < threshold else 1
        return y_pred

    def evaluate(self, y, y_pred):
        assert len(y) == len(y_pred)
        len_y = len(y)
        assert len_y > 0
        count_correct = 0
        confusion_matrix = np.zeros(2, 2)
        for i in range(len_y):
            if y[i] == y_pred[i]:
                count_correct += 1
            confusion_matrix[y[i]][y_pred[i]] += 1
        accuracy = count_correct / len_y

        print()

    def save(self, filepath):
        f = open(filepath, 'wb')
        v = {}
        v["shapelet_file_path"] = self.shapelet_file_path
        v["loaded_shape_dict"] = self.loaded_shape_dict
        v["shapelets"] = self.shapelets
        v["n_shapelets"] = self.shapelets
        v["plot_output_root_dir"] = self.plot_output_root_dir
        v["dist_shape"] = self.dist_shape
        v["dist_ratio_shape"] = self.dist_ratio_shape
        v["shapelet_labels"] = self.shapelet_labels
        v["n_labels"] = self.n_labels
        v["n_shapelet_labels"] = self.n_shapelet_labels
        v["evolution_event_selection"] = self.evolution_event_selection
        pickle.dump(v, f)
        f.close()

    def load(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                content = pickle.loads(f.read())
        else:
            print(f"Cannot find model file {filepath}")
            assert False

        # cs = ClassifySimpleDist()
        self.set_init_parameters(content["shapelet_file_path"], content["plot_output_root_dir"])
        self.dist_shape = content["dist_shape"]
        self.dist_ratio_shape = content["dist_ratio_shape"]
        self.shapelet_labels = content["shapelet_labels"]
        self.n_labels = content["n_labels"]
        self.n_shapelet_labels = content["n_shapelet_labels"]
        self.evolution_event_selection = content["evolution_event_selection"]
        f.close()
        # return cs
