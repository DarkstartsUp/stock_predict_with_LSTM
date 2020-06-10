# -*- coding: UTF-8 -*-
"""
@author: hichenway
@contact: lyshello123@163.com
@time: 2020/5/9 17:00
@license: Apache
pytorch 模型
"""

import torch
from .conv_lstm import ConvLSTM, CrossEntropy2d
from torch.nn import Module, LSTM, Linear, Softmax2d, Conv2d
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import numpy as np


class Net(Module):
    '''
    pytorch预测模型，包括ConvLSTM时序预测层和用于输出降维的全连接层（Softmax分类输出层）
    '''
    def __init__(self, config):
        super(Net, self).__init__()
        # self.lstm = LSTM(input_size=config.input_size, hidden_size=config.hidden_size,
        #                  num_layers=config.lstm_layers, batch_first=True, dropout=config.dropout_rate)
        # self.linear = Linear(in_features=config.hidden_size, out_features=config.output_size)

        # use convlstm:
        self.convlstm = ConvLSTM(input_dim=config.input_channels, hidden_dim=config.hidden_size,
                                 kernel_size=config.conv_kernel, num_layers=config.lstm_layers, batch_first=True)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=config.hidden_size, out_channels=config.conv_out_channel, kernel_size=1, padding=0),
            nn.BatchNorm2d(config.conv_out_channel),
            nn.ReLU())  # 32, 16, 16

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=config.conv_out_channel, out_channels=config.category_num, kernel_size=1, padding=0),
            nn.BatchNorm2d(config.category_num),
            nn.ReLU())  # 16, 3, 3

        self.softmax = Softmax2d()

    def forward(self, x):
        # , hidden=None
        # lstm_out, hidden = self.lstm(x, hidden)
        # linear_out = self.linear(lstm_out)

        _, last_states = self.convlstm(x)
        h = last_states[0][0]  # torch.Size([batch_size, 3, 18, 18])

        c1 = self.conv1(h)
        c2 = self.conv2(c1)

        # Softmax over the 2nd dimension
        softmax_output = self.softmax(c2)

        return softmax_output         # linear_out, hidden


def train(config, logger, train_and_valid_data):
    if config.do_train_visualized:
        import visdom
        vis = visdom.Visdom(env='model_pytorch')

    train_X, train_Y, valid_X, valid_Y = train_and_valid_data
    train_X, train_Y = torch.from_numpy(train_X).float(), torch.from_numpy(train_Y).long()     # 先转为Tensor
    train_loader = DataLoader(TensorDataset(train_X, train_Y), batch_size=config.batch_size)    # DataLoader可自动生成可训练的batch数据

    valid_X, valid_Y = torch.from_numpy(valid_X).float(), torch.from_numpy(valid_Y).long()
    valid_loader = DataLoader(TensorDataset(valid_X, valid_Y), batch_size=config.batch_size)

    device = torch.device("cuda:0" if config.use_cuda and torch.cuda.is_available() else "cpu") # CPU训练还是GPU
    model = Net(config).to(device)      # 如果是GPU训练， .to(device) 会把模型/数据复制到GPU显存中
    if config.add_train:                # 如果是增量训练，会先加载原模型参数
        model.load_state_dict(torch.load(config.model_save_path + config.model_name))

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = CrossEntropy2d()      # 这两句是定义优化器和loss

    valid_loss_min = float("inf")
    bad_epoch = 0
    global_step = 0
    for epoch in range(config.epoch):
        logger.info("Epoch {}/{}".format(epoch, config.epoch))
        model.train()                   # pytorch中，训练时要转换成训练模式
        train_loss_array = []
        for i, _data in enumerate(train_loader):
            _train_X, _train_Y = _data[0].to(device),_data[1].to(device)
            optimizer.zero_grad()               # 训练前要将梯度信息置 0
            pred_Y = model(_train_X)    # 这里走的就是前向计算forward函数

            loss = criterion(pred_Y, _train_Y)  # 计算loss
            loss.backward()                     # 将loss反向传播
            optimizer.step()                    # 用优化器更新参数
            train_loss_array.append(loss.item())
            global_step += 1
            if config.do_train_visualized and global_step % 100 == 0:   # 每一百步显示一次
                vis.line(X=np.array([global_step]), Y=np.array([loss.item()]), win='Train_Loss',
                         update='append' if global_step > 0 else None, name='Train', opts=dict(showlegend=True))

        # 以下为早停机制，当模型训练连续config.patience个epoch都没有使验证集预测效果提升时，就停止，防止过拟合
        model.eval()                    # pytorch中，预测时要转换成预测模式
        valid_loss_array = []
        for _valid_X, _valid_Y in valid_loader:
            _valid_X, _valid_Y = _valid_X.to(device), _valid_Y.to(device)
            pred_Y = model(_valid_X)
            loss = criterion(pred_Y, _valid_Y)  # 验证过程只有前向计算，无反向传播过程
            valid_loss_array.append(loss.item())

        train_loss_cur = np.mean(train_loss_array)
        valid_loss_cur = np.mean(valid_loss_array)
        logger.info("The train loss is {:.6f}. ".format(train_loss_cur) +
              "The valid loss is {:.6f}.".format(valid_loss_cur))
        if config.do_train_visualized:      # 第一个train_loss_cur太大，导致没有显示在visdom中
            vis.line(X=np.array([epoch]), Y=np.array([train_loss_cur]), win='Epoch_Loss',
                     update='append' if epoch > 0 else None, name='Train', opts=dict(showlegend=True))
            vis.line(X=np.array([epoch]), Y=np.array([valid_loss_cur]), win='Epoch_Loss',
                     update='append' if epoch > 0 else None, name='Eval', opts=dict(showlegend=True))

        if valid_loss_cur < valid_loss_min:
            valid_loss_min = valid_loss_cur
            bad_epoch = 0
            torch.save(model.state_dict(), config.model_save_path + config.model_name)  # 模型保存
        else:
            bad_epoch += 1
            if bad_epoch >= config.patience:    # 如果验证集指标连续patience个epoch没有提升，就停掉训练
                logger.info(" The training stops early in epoch {}".format(epoch))
                break


def predict(config, test_X):
    # 获取测试数据
    test_X = torch.from_numpy(test_X).float()
    test_set = TensorDataset(test_X)
    test_loader = DataLoader(test_set, batch_size=1)

    # 加载模型
    device = torch.device("cuda:0" if config.use_cuda and torch.cuda.is_available() else "cpu")
    model = Net(config).to(device)
    model.load_state_dict(torch.load(config.model_save_path + config.model_name))   # 加载模型参数

    # 先定义一个tensor保存预测结果
    result = torch.Tensor().to(device)

    # 预测过程
    model.eval()
    for _data in test_loader:
        data_X = _data[0].to(device)
        pred_X = model(data_X)
        cur_pred = torch.squeeze(pred_X, dim=0)
        result = torch.cat((result, cur_pred), dim=0)

    return result.detach().cpu().numpy()    # 先去梯度信息，如果在gpu要转到cpu，最后要返回numpy数据
