# -*- coding: UTF-8 -*-
"""
@Author: xueyang
@File Create: 20200609
@Last Modify: 20200627
@Function: Train Conv-LSTM for stock prediction, including config parsing, data processing and model training
"""


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
    # 网络参数
    time_step = 30                # 设置用前多少天的数据来预测，也是LSTM的time step数
    predict_day = 7               # 预测未来多少天的涨跌平
    input_channels = 2            # 输入的特征维数，默认价格在第0维
    hidden_size = 32              # LSTM的隐藏层大小，也是输出大小
    lstm_layers = 3               # LSTM的堆叠层数
    conv_kernel = (5, 5)          # ConvLSTM卷积核的大小

    # 输出的维度由hidden_size到category_num的过程，使用1x1卷积进行降维
    # 由于将convlstm输出的hidden和memory concatenate到一起输入1x1卷积层，因此convlstm的实际输出维度为2 * hidden_size
    conv_out_channel = 32         # 卷积中间层的输出channel数
    category_num = 3              # 输出的预测类别数量

    # dropout_rate = 0.2          # TODO：dropout概率

    # 训练参数
    do_train = True
    do_predict = True
    add_train = False             # 是否载入已有模型参数进行增量训练
    shuffle_train_and_val = False  # 是否对打乱混合训练集和验证集
    shuffle_train_data = True     # 是否对训练数据做shuffle
    use_cuda = True               # 是否使用GPU训练

    train_data_rate = 0.8         # 训练数据占总体数据比例，测试数据就是 1-train_data_rate
    valid_data_rate = 0.1         # 验证数据占训练数据比例，验证集在训练过程使用，为了做模型和参数选择

    batch_size = 48
    learning_rate = 0.0001         # TODO： lr梯度下降
    epoch = 30                    # 整个训练集被训练多少遍，不考虑早停的前提下
    lr_step_size = 30             # 学习率自动调整的步长
    patience = 500                # 训练多少epoch，验证集没提升就停掉
    random_seed = 42              # 随机种子，保证可复现

    # do_continue_train = False    # 每次训练把上一次的final_state作为下一次的init_state，仅用于RNN类型模型，目前仅支持pytorch
    # continue_flag = ""           # 但实际效果不佳，可能原因：仅能以 batch_size = 1 训练
    # if do_continue_train:
    #     shuffle_train_data = False
    #     batch_size = 1
    #     continue_flag = "continue_"

    # 训练模式
    debug_mode = False  # 调试模式下，是为了跑通代码，追求快
    debug_num = 500  # 仅用debug_num条数据来调试

    experiment_name = 'ConvLSTM'
    model_postfix = ".pth"
    model_name = "model_" + time.strftime("%Y%m%d_%H%M%S") + model_postfix

    # 路径参数
    train_data_path = './data/data_zz800_prize_and_volume.npy'
    model_save_path = "./checkpoint/" + experiment_name + "/"
    figure_save_path = "./figure/"
    log_save_path = "./log/"
    do_log_save = True                  # 是否将config和训练过程记录到log
    do_figure_save = False
    do_train_visualized = False         # 训练loss可视化，pytorch用visdom或tensorboardX

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)    # makedirs 递归创建目录
    if not os.path.exists(figure_save_path):
        os.mkdir(figure_save_path)
    if do_log_save or do_train_visualized:
        cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        log_save_path = log_save_path + cur_time + '_' + experiment_name + "/"
        os.makedirs(log_save_path)


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
                                                              shuffle=self.config.shuffle_train_and_val)
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


# 结果可视化模块  暂时没用到
def draw(config: Config, origin_data: Data, logger, predict_norm_data: np.ndarray):
    label_data = origin_data.data[origin_data.train_num + origin_data.start_num_in_test : ,
                                            config.label_in_feature_index]
    predict_data = predict_norm_data * origin_data.std[config.label_in_feature_index] + \
                   origin_data.mean[config.label_in_feature_index]   # 通过保存的均值和方差还原数据
    assert label_data.shape[0]==predict_data.shape[0], "The element number in origin and predicted data is different"

    label_name = [origin_data.data_column_name[i] for i in config.label_in_feature_index]
    label_column_num = len(config.label_columns)

    # label 和 predict 是错开config.predict_day天的数据的
    # 下面是两种norm后的loss的计算方式，结果是一样的，可以简单手推一下
    # label_norm_data = origin_data.norm_data[origin_data.train_num + origin_data.start_num_in_test:,
    #              config.label_in_feature_index]
    # loss_norm = np.mean((label_norm_data[config.predict_day:] - predict_norm_data[:-config.predict_day]) ** 2, axis=0)
    # logger.info("The mean squared error of stock {} is ".format(label_name) + str(loss_norm))

    loss = np.mean((label_data[config.predict_day:] - predict_data[:-config.predict_day] ) ** 2, axis=0)
    loss_norm = loss/(origin_data.std[config.label_in_feature_index] ** 2)
    logger.info("The mean squared error of stock {} is ".format(label_name) + str(loss_norm))

    label_X = range(origin_data.data_num - origin_data.train_num - origin_data.start_num_in_test)
    predict_X = [ x + config.predict_day for x in label_X]

    if True:  # not sys.platform.startswith('linux'):    # 无桌面的Linux下无法输出，如果是有桌面的Linux，如Ubuntu，可去掉这一行
        for i in range(label_column_num):
            plt.figure(i+1)                     # 预测数据绘制
            plt.plot(label_X, label_data[:, i], label='label')
            plt.plot(predict_X, predict_data[:, i], label='predict')
            plt.title("Predict stock {} price with {}".format(label_name[i], config.used_frame))
            logger.info("The predicted stock {} for the next {} day(s) is: ".format(label_name[i], config.predict_day) +
                  str(np.squeeze(predict_data[-config.predict_day:, i])))
            if config.do_figure_save:
                plt.savefig(config.figure_save_path+"{}predict_{}_with_{}.png".format(config.continue_flag, label_name[i], config.used_frame))

        plt.show()


def main(config):
    logger = load_logger(config)
    try:
        np.random.seed(config.random_seed)  # 设置随机种子，保证可复现
        data_gainer = Data(config)

        if config.do_train:
            train_X, valid_X, train_Y, valid_Y = data_gainer.get_train_and_valid_data()
            train(config, logger, [train_X, train_Y, valid_X, valid_Y])

        if config.do_predict:
            test_X, test_Y = data_gainer.get_test_data()
            pred_ys, real_ys, pred_ys_no_flat = predict(config, [test_X, test_Y])
            target_names = ['class flat', 'class down', 'class rise']    # label: 0:平  1:跌  2:涨
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
