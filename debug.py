# -*- coding: utf-8 -*-
"""
@Author: xueyang
@File Create: 20200606
@Last Modify: 20200627
@Function: temporary codebase for debugging any code here
"""
import time
import numpy as np
from sklearn.model_selection import train_test_split
from train import Config
import torch
from model.conv_lstm import ConvLSTM, CrossEntropy2d
from torch.nn import Softmax2d
import matplotlib.pyplot as plt

class Data:
    def __init__(self, config):
        self.config = config
        self.data = self.read_data()                                       # shape: (H, W, T, C)
        self.data_num = self.data.shape[2]                                 # data_num = T = total number of days
        self.train_num = int(self.data_num * self.config.train_data_rate)
        self.test_num = self.data_num - self.train_num

        # 对每只股票的每个channel在时序根据均值和方差进行归一化
        for channel in range(config.input_channels):
            temp_data = self.data[:, :, :, channel]                        # shape: (H, W, T, 1)
            temp_data = np.squeeze(temp_data)                              # shape: (H, W, T)
            mean = np.mean(temp_data, axis=2)[:, :, np.newaxis]
            std = np.std(temp_data, axis=2)[:, :, np.newaxis]
            normed_temp_data = (temp_data - mean) / std                    # shape: (H, W, T)
            if channel == 0:
                self.prize_data = normed_temp_data
                self.normed_data = normed_temp_data[:, :, :, np.newaxis]   # shape: (H, W, T, C)
            else:
                self.normed_data = np.concatenate((self.normed_data, normed_temp_data[:, :, :, np.newaxis]), axis=3)

        # 构造标签
        self.label_data = np.zeros_like(self.prize_data, dtype='int64')     # 每只股票每天的label（根据后7天） shape: (H, W, T)
        for i in range(self.prize_data.shape[2] - self.config.predict_day):    # i in range [0, T-7]
            reference_data = self.prize_data[:, :, i: i + self.config.predict_day]    # 取出 range [i, i + 6]的数据
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
        # 取出归一化后的数据的训练部分, (H, W, T, C)
        feature_data = self.normed_data[:, :, :self.train_num - self.config.predict_day, :]
        # 将延后7天的数据作为label, (H, W, T)
        label_data = self.label_data[:, :, self.config.time_step: self.train_num - self.config.predict_day]
        # 每time_step行数据会作为一个样本，两个样本错开一日，比如：1-30日，2-31日
        train_x = [feature_data[:, :, i:i+self.config.time_step, :] for i in range(self.train_num - self.config.time_step - self.config.predict_day)]
        train_y = np.transpose(label_data, (2, 0, 1))            # (T, H, W)

        train_x = np.array(train_x)                              # b, h, w, t, c
        train_x = np.transpose(train_x, (0, 3, 4, 1, 2))         # b, t, c, h, w

        # 划分训练和验证集，并打乱
        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=self.config.valid_data_rate,
                                                              random_state=self.config.random_seed,
                                                              shuffle=self.config.shuffle_train_data)
        return train_x, valid_x, train_y, valid_y

    def get_test_data(self):
        feature_data = self.normed_data[:, :, self.train_num: -self.config.predict_day, :]
        label_data = self.label_data[:, :, self.train_num + self.config.time_step: -self.config.predict_day]

        # 在测试数据中，每两个样本错开一日，比如：1-30日，2-31日，到数据末尾
        test_x = [feature_data[:, :, i:i + self.config.time_step, :] for i in range(self.test_num - self.config.time_step - self.config.predict_day)]
        test_y = np.transpose(label_data, (2, 0, 1))          # (T, H, W)

        test_x = np.array(test_x)                             # b, h, w, t
        test_x = np.transpose(test_x, (0, 3, 4, 1, 2))        # b, t, c, h, w

        return test_x, test_y


if __name__ == '__main__':
    prize_data = np.load('data/Astock_zz800_no_nan.npy')
    volume_data = np.load('data/data_zz800_vol.npy')
    prize_data = prize_data[:, :, :, np.newaxis]
    volume_data = volume_data[:, :, :, np.newaxis]
    stock_data = np.concatenate((prize_data, volume_data), axis=3)
    print(stock_data.shape)
    np.save('data/data_zz800_prize_and_volume.npy', stock_data)

    # for i in range(len(stock_data)):
    #     for j in range(len(stock_data[i])):
    #         for t in range(len(stock_data[i][j])):
    #             if np.isnan(stock_data[i][j][t]):
    #                 print(i, j, t)

    # import argparse
    # # argparse方便于命令行下输入参数，可以根据需要增加更多
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()
    #
    # config = Config()
    # for key in dir(args):  # dir(args) 函数获得args所有的属性
    #     if not key.startswith("_"):  # 去掉 args 自带属性，比如__name__等
    #         setattr(config, key, getattr(args, key))  # 将属性值赋给Config
    #
    # data_gainer = Data(config)
    # test_x, test_y = data_gainer.get_test_data()
    #
    # print(test_x[0,0,:,0,0])

    # x = torch.rand((2, 10, 1, 18, 18), requires_grad=True)
    # label = torch.empty(2, 18, 18, dtype=torch.long).random_(3)
    # # print(label)
    #
    # convlstm = ConvLSTM(1, 6, (3, 3), 2, True, True, False)
    # _, last_states = convlstm(x)
    # h = last_states[0][0]  # torch.Size([2, 3, 18, 18])
    # m = last_states[0][1]
    #
    # out = torch.cat((h, m), dim=1)
    # print(out.size())
    # print(out[0,0,:,:])


    #
    # m = Softmax2d()
    # # you softmax over the 2nd dimension
    # output = m(h)
    #
    # print(output.size())
    # print(output)
    #
    # mm = np.argmax(output.detach().numpy(), axis=1)
    #
    # print(mm.shape)
    # print(mm)

    # loss = CrossEntropy2d()
    # l = loss(output, label)
    # print(l)
    # l.backward()

    # h = last_states[0][0]  # 0 for layer index, 0 for h index
    # print(layer_output[0].size())
    # print(np.array(layer_output).shape)
    # print(last_states[0][0])

    # init_data = np.load('./data/Astock_center_zz800.npy')
    # save_data = init_data[:19, :19, 600:]
    # print(save_data.shape)
    # np.save('./data/Astock_center_zz800_19_19_600p.npy', save_data)
    # print(init_data.shape)
    # gegu = init_data[20, 20, :]
    # xzhou = [i for i in range(len(gegu))]
    #
    # plt.plot(xzhou, gegu)
    # plt.show()
