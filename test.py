# -*- coding: UTF-8 -*-
"""
@Author: xueyang
@File Create: 20200611
@Last Modify: 20200611
@Function: Test Conv-LSTM model for stock prediction
"""

import pandas as pd
import numpy as np
import os
import sys
import time
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from model.model_pytorch import train, predict


class Config:
    # 数据参数
    # feature_columns = list(range(2, 9))     # 要作为feature的列，按原数据从0开始计算，也可以用list 如 [2,4,6,8] 设置
    # label_columns = [4, 5]                  # 要预测的列，按原数据从0开始计算, 如同时预测第四，五列 最低价和最高价
    # label_in_feature_index = [feature_columns.index(i) for i in label_columns]  # 这样写不行
    # label_in_feature_index = (lambda x,y: [x.index(i) for i in y])(feature_columns, label_columns)  # 因为feature不一定从0开始
    # 网络参数
    # input_size = len(feature_columns)
    # output_size = len(label_columns)

    time_step = 30                # 设置用前多少天的数据来预测，也是LSTM的time step数
    predict_day = 7               # 预测未来多少天的涨跌平
    input_channels = 1            # 输入的特征维数
    hidden_size = 32              # LSTM的隐藏层大小，也是输出大小
    lstm_layers = 3               # LSTM的堆叠层数
    conv_kernel = (3, 3)          # ConvLSTM卷积核的大小

    # 输出的维度由hidden_size到category_num的过程，使用1x1卷积进行降维
    conv_out_channel = 16         # 卷积中间层的输出channel数
    category_num = 3              # 输出的预测类别数量

    train_data_rate = 0.8         # 训练数据占总体数据比例，测试数据就是 1-train_data_rate
    valid_data_rate = 0.1         # 验证数据占训练数据比例，验证集在训练过程使用，为了做模型和参数选择

    batch_size = 64
    learning_rate = 0.0005
    epoch = 120                    # 整个训练集被训练多少遍，不考虑早停的前提下
    patience = 120                 # 训练多少epoch，验证集没提升就停掉
    random_seed = 42               # 随机种子，保证可复现

    # do_continue_train = False    # 每次训练把上一次的final_state作为下一次的init_state，仅用于RNN类型模型，目前仅支持pytorch
    # continue_flag = ""           # 但实际效果不佳，可能原因：仅能以 batch_size = 1 训练
    # if do_continue_train:
    #     shuffle_train_data = False
    #     batch_size = 1
    #     continue_flag = "continue_"

    # 训练模式
    debug_mode = False  # 调试模式下，是为了跑通代码，追求快
    debug_num = 500  # 仅用debug_num条数据来调试
    use_cuda = True  # 是否使用GPU训练

    experiment_name = 'ConvLSTM'
    model_postfix = ".pth"
    model_name = 'model_20200611_195548.pth'

    # 路径参数
    train_data_path = './data/Astock_hs300_no_nan.npy'
    model_save_path = "./checkpoint/" + experiment_name + "/"
    figure_save_path = "./figure/"
    log_save_path = "./log/"
    do_log_save = True                  # 是否将config和训练过程记录到log
    do_figure_save = False
    do_train_visualized = False         # 训练loss可视化，pytorch用visdom或tensorboardX


class Data:
    def __init__(self, config):
        self.config = config
        self.data = self.read_data()                                       # shape: (H, W, T)
        self.data_num = self.data.shape[2]                                 # data_num = T = total number of days
        self.train_num = int(self.data_num * self.config.train_data_rate)
        self.test_num = self.data_num - self.train_num

        self.mean = np.mean(self.data, axis=2)                             # 计算每只股票时序上的均值和方差
        self.std = np.std(self.data, axis=2)
        self.mean = self.mean[:, :, np.newaxis]
        self.std = self.std[:, :, np.newaxis]
        self.norm_data = (self.data - self.mean) / self.std                # 分别对每只股票的价格做归一化 shape: (H, W, T)

        # 构造标签
        self.label_data = np.zeros_like(self.norm_data, dtype='int64')     # 每只股票每天的label（根据前7天） shape: (H, W, T)
        for i in range(self.config.predict_day, self.norm_data.shape[2]):    # i in range [7, T)
            reference_data = self.norm_data[:, :, i - self.config.predict_day:i]    # 取出 range [i - 7, i)的数据
            difference = reference_data[:, :, -1] - reference_data[:, :, 0]    # 7天当中的最后一天股价减第一天股价，作为sort依据
            temp_frame = np.zeros_like(difference)         # 用于存储第i天各个股票的label shape: (H, W)
            sorted_diff = np.sort(difference.flatten())
            low, high = sorted_diff[int(len(sorted_diff) * 1 / 3)], sorted_diff[int(len(sorted_diff) * 2 / 3)]
            temp_frame[difference < low] = 1               # label: 0:平  1:跌  2:涨
            temp_frame[difference > high] = 2
            self.label_data[:, :, i] = temp_frame

    def read_data(self):                # 读取初始数据
        init_data = np.load(self.config.train_data_path)
        return init_data

    def get_train_and_valid_data(self):
        feature_data = self.norm_data[:, :, :self.train_num]       # 取出归一化后的数据的训练部分, (H, W, T)
        label_data = self.label_data[:, :, self.config.time_step: self.train_num]   # 将延后7天的数据作为label, (H, W, T)

        # 每time_step行数据会作为一个样本，两个样本错开一日，比如：1-30日，2-31日
        train_x = [feature_data[:, :, i:i+self.config.time_step] for i in range(self.train_num-self.config.time_step)]
        train_y = np.transpose(label_data, (2, 0, 1))     # (T, H, W)

        train_x = np.array(train_x)                              # b, h, w, t
        train_x = np.transpose(train_x, (0, 3, 1, 2))            # b, t, h, w
        train_x = train_x[:, :, np.newaxis, :, :]                # b, t, c, h, w

        # 划分训练和验证集，并打乱
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=self.config.valid_data_rate,
                                                              random_state=self.config.random_seed,
                                                              shuffle=self.config.shuffle_train_data)
        return train_x, valid_x, train_y, valid_y

    def get_test_data(self):
        feature_data = self.norm_data[:, :, self.train_num:]
        label_data = self.label_data[:, :, self.train_num + self.config.time_step:]

        # 在测试数据中，每两个样本错开一日，比如：1-30日，2-31日，到数据末尾
        test_x = [feature_data[:, :, i:i + self.config.time_step] for i in range(self.test_num - self.config.time_step)]
        test_y = np.transpose(label_data, (2, 0, 1))

        test_x = np.array(test_x)                             # b, h, w, t
        test_x = np.transpose(test_x, (0, 3, 1, 2))           # b, t, h, w
        test_x = test_x[:, :, np.newaxis, :, :]               # b, t, c, h, w

        return test_x, test_y


def load_logger(config):
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)

    # StreamHandler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level=logging.INFO)
    formatter = logging.Formatter(datefmt='%Y/%m/%d %H:%M:%S',
                                  fmt='[ %(asctime)s ] %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # FileHandler
    if config.do_log_save:
        file_handler = logging.FileHandler(config.log_save_path + "out.log")
        file_handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 把config信息也记录到log 文件中
        config_dict = {}
        for key in dir(config):
            if not key.startswith("_"):
                config_dict[key] = getattr(config, key)
        config_str = str(config_dict)
        config_list = config_str[1:-1].split(", '")
        config_save_str = "\nConfig:\n" + "\n'".join(config_list)
        logger.info(config_save_str)

    return logger


def main(config):
    logger = load_logger(config)
    try:
        np.random.seed(config.random_seed)  # 设置随机种子，保证可复现
        data_gainer = Data(config)

        test_X, test_Y = data_gainer.get_test_data()
        pred_ys, real_ys = predict(config, [test_X, test_Y])
        target_names = ['class flat', 'class down', 'class rise']  # label: 0:平  1:跌  2:涨
        print('Classification table for test set:')
        print(classification_report(real_ys, pred_ys, target_names=target_names))
        # draw(config, data_gainer, logger, pred_result)

    except Exception:
        logger.error("Run Error", exc_info=True)


if __name__=="__main__":
    import argparse
    # argparse方便于命令行下输入参数，可以根据需要增加更多
    parser = argparse.ArgumentParser()
    # parser.add_argument("-t", "--do_train", default=False, type=bool, help="whether to train")
    # parser.add_argument("-p", "--do_predict", default=True, type=bool, help="whether to train")
    # parser.add_argument("-b", "--batch_size", default=64, type=int, help="batch size")
    # parser.add_argument("-e", "--epoch", default=20, type=int, help="epochs num")
    args = parser.parse_args()

    con = Config()
    for key in dir(args):               # dir(args) 函数获得args所有的属性
        if not key.startswith("_"):     # 去掉 args 自带属性，比如__name__等
            setattr(con, key, getattr(args, key))   # 将属性值赋给Config

    main(con)