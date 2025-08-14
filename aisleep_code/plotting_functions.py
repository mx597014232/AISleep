"""
画图函数模块 - 将算法和画图函数分离
包含所有与数据可视化相关的函数
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from sklearn.neighbors import KernelDensity
from scipy.stats import norm
import matplotlib as mpl
from aisleep_code.data_processor import DatasetCreate

# 设置matplotlib参数
ft18 = 18
color = {0: '#317EC2',  # Wake 蓝色
         1: '#F2B342',  # N1 粉色
         2: '#5AAA46',  # N2 绿色
         3: '#C03830',  # N3 红色
         4: '#825CA6',  # REM 紫色
         5: '#C43D96',  # Wake(Open eye) 黄橙色
         6: '#8D7136'}  # NREM

stage = ['Wake', 'N1', 'N2', 'N3', 'REM']

# 设置全局字体
mpl.rcParams.update({'font.size': ft18})
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['svg.fonttype'] = 'none'


def fig_2a_plot_gamma_hist(self=None, ax=None):
    if self is None:
        self = DatasetCreate(0)
    psd, y = np.log(self.psd), self.y
    # scaler = StandardScaler()  # 标准化处理
    # psd = scaler.fit_transform(psd)  # 不需要做标准化处理了
    mean = psd[:, np.logical_and(25 < self.freqs, self.freqs < 50)].mean(axis=1)  # 16至50Hz的归一化均值， 25-50的脑电波更好

    x, loss_list = mean, []
    num_all, start, end = x.shape[0], x.min(), x.max()
    # print(start, end)
    threshold, loss, interval = start, 0, (end - start) / 100  # 最大化loss(类间方差), 搜索步长间隔为1%
    bins = np.arange(start + interval, end - interval, interval)
    # 阈值从start到end遍历搜索
    for th in np.arange(start + interval, end - interval, interval):
        c0, c1 = x[x <= th], x[x > th]  # 根据阈值划分两个类别的点
        w0, w1 = c0.shape[0] / num_all, c1.shape[0] / num_all  # 权重
        u0, u1 = c0.mean(), c1.mean()  # 均值
        temp_loss = w0 * w1 * (u0 - u1) * (u0 - u1)  # 当前阈值对应的类间方差
        loss_list.append(temp_loss)
        if loss < temp_loss:
            # print(th, temp_loss)
            threshold, loss = th, temp_loss
    loss_list = np.array(loss_list)
    th_idx = np.argmax(loss_list)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    bins_ = np.linspace(mean.min() - 0.1, mean.max() + 0.1, 40)
    ax.hist(mean[mean <= threshold], bins=bins_, label='Low gamma', alpha=0.5, color='gray')
    ax.hist(mean[mean >= threshold], bins=bins_, label='High gamma', alpha=0.5, color=color[0])
    ax.plot([bins[th_idx], bins[th_idx]], [0, 93], c='red', linestyle='--', label='Threshold')
    ax.set_ylabel('Bin counts', fontdict={'size': ft18})
    ax.set_xlabel('Gamma power (dB)', fontdict={'size': ft18})
    ax.tick_params(labelsize=ft18)  # 坐标轴字体大小
    # ax2 = ax.twinx()  # 双y轴显示
    # ax2.plot(np.arange(start + interval, end - interval, interval), loss_list, label='Loss', c='black')
    # ax2.set_ylabel('Otsu loss value', fontdict={'size': ft18})
    # ax2.set_ylim(0, loss_list.max() * 2)
    # ax2.scatter([bins[th_idx]], [loss_list.max()], marker='x', c='red', s=100, label='Maximum')
    legend = ax.legend(framealpha=0, bbox_to_anchor=(0.75, 0.85), loc='center', bbox_transform=ax.transAxes,
                       prop={'size': ft18})
    legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0
    # fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes,
    #            prop={'size': ft18})
    ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
    # ax.set_ylim(0, 2)
    # ax.set_yticks([0, 1, 2], [0, 1, 2])
    ax.yaxis.set_label_coords(-0.11, 0.5)  # 固定y坐标轴label的位置
    # ax2.spines['top'].set_visible(False)
    # ax2.tick_params(labelsize=ft18)  # 坐标轴字体大小

    if fig:
        plt.subplots_adjust(left=0.12, bottom=0.15, right=0.88, top=1)
        plt.savefig('./paper_figure/Wake_Otsu.svg')
        plt.show()


