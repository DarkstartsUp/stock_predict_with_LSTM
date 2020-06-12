# -*- coding: UTF-8 -*-
"""
@Author: xueyang
@File Create: 20200611
@Last Modify: 20200611
@Function: Test Conv-LSTM model for stock prediction
"""


import numpy as np
from train import Data, load_logger
from sklearn.metrics import classification_report
from model.model_pytorch import predict


class Config:
    time_step = 30                # 设置用前多少天的数据来预测，也是LSTM的time step数
    predict_day = 7               # 预测未来多少天的涨跌平
    input_channels = 1            # 输入的特征维数
    hidden_size = 32              # LSTM的隐藏层大小，也是输出大小
    lstm_layers = 3               # LSTM的堆叠层数
    conv_kernel = (3, 3)          # ConvLSTM卷积核的大小

    # 输出的维度由hidden_size到category_num的过程，使用1x1卷积进行降维
    conv_out_channel = 16         # 卷积中间层的输出channel数
    category_num = 3              # 输出的预测类别数量

    train_data_rate = 0.1         # 训练数据占总体数据比例，测试数据就是 1-train_data_rate
    valid_data_rate = 0.1         # 验证数据占训练数据比例，验证集在训练过程使用，为了做模型和参数选择
    random_seed = 42               # 随机种子，保证可复现
    use_cuda = True  # 是否使用GPU训练

    experiment_name = 'ConvLSTM'
    model_postfix = ".pth"
    model_name = 'model_20200612_224227.pth'

    # 路径参数
    train_data_path = './data/Astock_center_zz800.npy'
    model_save_path = "./checkpoint/" + experiment_name + "/"
    figure_save_path = "./figure/"
    log_save_path = "./log/"
    do_log_save = True                  # 是否将config和训练过程记录到log
    do_figure_save = False
    do_train_visualized = False         # 训练loss可视化，pytorch用visdom或tensorboardX


def main(config):
    logger = load_logger(config)
    try:
        np.random.seed(config.random_seed)  # 设置随机种子，保证可复现
        data_gainer = Data(config)

        test_X, test_Y = data_gainer.get_test_data()
        pred_ys, real_ys, pred_ys_no_flat = predict(config, [test_X, test_Y])
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
    args = parser.parse_args()

    con = Config()
    for key in dir(args):               # dir(args) 函数获得args所有的属性
        if not key.startswith("_"):     # 去掉 args 自带属性，比如__name__等
            setattr(con, key, getattr(args, key))   # 将属性值赋给Config
    main(con)