# -*- coding: utf-8 -*-
"""
@Author: xueyang
@File Create: 20200606
@Last Modify: 20200606
@Function: temporary codebase for debugging any code here
"""

import numpy as np


if __name__ == '__main__':
    stock_file_path = './data/Astock_hs300.npy'
    stock_data = np.load(stock_file_path)
    # shape: (18, 18, 975)
    # (stock, stock, timestamp)
    # stock sort: rop & eps
