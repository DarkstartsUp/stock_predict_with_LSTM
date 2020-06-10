# -*- coding: utf-8 -*-
"""
@Author: xueyang
@File Create: 20200606
@Last Modify: 20200610
@Function: temporary codebase for debugging any code here
"""
import time
import numpy as np
from sklearn.model_selection import train_test_split
from train import Config

import torch
from model.conv_lstm import ConvLSTM, CrossEntropy2d
from torch.nn import Softmax2d

class Data:
    def __init__(self, config):
        self.config = config
        self.data = self.read_data()

        self.data_num = self.data.shape[2]
        self.train_num = int(self.data_num * self.config.train_data_rate)
        self.test_num = self.data_num - self.train_num

        self.mean = np.mean(self.data, axis=2)                                            # 计算每只股票时序上的均值和方差
        self.std = np.std(self.data, axis=2)
        self.mean = np.reshape(self.mean, (self.mean.shape[0], self.mean.shape[1], 1))
        self.std = np.reshape(self.std, (self.std.shape[0], self.std.shape[1], 1))
        self.norm_data = (self.data - self.mean) / self.std                               # 对每只股票的价格归一化

        # 构造标签
        self.label_data = np.zeros_like(self.norm_data, dtype='int64')
        for i in range(self.config.predict_day + 1, self.norm_data.shape[2]):
            reference_data = self.norm_data[:, :, i - self.config.predict_day - 1:i]
            difference = reference_data[:, :, -1] - reference_data[:, :, 0]
            temp_frame = np.zeros_like(difference)
            sorted_diff = np.sort(difference.flatten())
            low, high = sorted_diff[int(len(sorted_diff) * 1 / 3)], sorted_diff[int(len(sorted_diff) * 2 / 3)]
            temp_frame[difference < low] = 1
            temp_frame[difference > high] = 2
            self.label_data[:, :, i] = temp_frame

        self.start_num_in_test = 0      # 测试集中前几天的数据会被删掉，因为它不够一个time_step

    def read_data(self):                # 读取初始数据
        init_data = np.load(self.config.train_data_path)
        return init_data

    def get_train_and_valid_data(self):
        feature_data = self.norm_data[:, :, :self.train_num]
        # 将延后几天的数据作为label
        label_data = self.label_data[:, :, self.config.time_step + 1: self.train_num + 1]

        # 每time_step行数据会作为一个样本，两个样本错开一日，比如：1-30日，2-31日
        train_x = [feature_data[:, :, i:i+self.config.time_step] for i in range(self.train_num-self.config.time_step)]
        train_y = np.transpose(np.transpose(label_data, (2, 1, 0)), (0, 2, 1))

        train_x, train_y = np.array(train_x), np.array(train_y)
        b, h, w, t = train_x.shape
        train_x = train_x.reshape((b, t, 1, h, w))
        # 划分训练和验证集，并打乱
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=self.config.valid_data_rate,
                                                              random_state=self.config.random_seed,
                                                              shuffle=self.config.shuffle_train_data)

        return train_x, valid_x, train_y, valid_y

    def get_test_data(self, return_label_data=False):
        feature_data = self.norm_data[:, :, self.train_num:]
        label_data = self.label_data[:, :, self.train_num + self.config.time_step:]

        # 在测试数据中，每两个样本错开一日，比如：1-30日，2-31日，到数据末尾
        test_x = [feature_data[:, :, i:i + self.config.time_step] for i in range(self.test_num - self.config.time_step)]
        test_y = np.transpose(np.transpose(label_data, (2, 1, 0)), (0, 2, 1))
        test_x, test_y = np.array(test_x), np.array(test_y)
        b, h, w, t = test_x.shape
        test_x = test_x.reshape((b, t, 1, h, w))

        return test_x, test_y


if __name__ == '__main__':

    # for i in range(len(stock_data)):
    #     for j in range(len(stock_data[i])):
    #         for t in range(len(stock_data[i][j])):
    #             if np.isnan(stock_data[i][j][t]):
    #                 print(i, j, t)

    import argparse
    # argparse方便于命令行下输入参数，可以根据需要增加更多
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    config = Config()
    for key in dir(args):  # dir(args) 函数获得args所有的属性
        if not key.startswith("_"):  # 去掉 args 自带属性，比如__name__等
            setattr(config, key, getattr(args, key))  # 将属性值赋给Config

    data_gainer = Data(config)
    data_gainer.get_train_and_valid_data()






    '''
    x = torch.rand((2, 10, 1, 18, 18), requires_grad=True)
    label = torch.empty(2, 18, 18, dtype=torch.long).random_(3)
    print(label)

    convlstm = ConvLSTM(1, 3, (3, 3), 2, True, True, False)
    _, last_states = convlstm(x)
    h = last_states[0][0]  # torch.Size([2, 3, 18, 18])

    m = Softmax2d()
    # you softmax over the 2nd dimension
    output = m(h)

    loss = CrossEntropy2d()
    l = loss(output, label)
    print(l)
    l.backward()
    '''

    # h = last_states[0][0]  # 0 for layer index, 0 for h index
    # print(layer_output[0].size())
    # print(np.array(layer_output).shape)
    # print(last_states[0][0])
