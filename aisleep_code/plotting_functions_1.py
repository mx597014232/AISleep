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
from sklearn.metrics.pairwise import pairwise_distances
from scipy.stats import norm
import matplotlib as mpl
from aisleep_code.data_processor import DatasetCreate, ostu, sleep_stage_all
from aisleep_code.stats_tools import paired_samples_test, convert_pvalue_to_asterisks
import yasa
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks,  medfilt


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


# gamma波的直方图分布
def fig_2a_plot_gamma_hist(self=None, ax=None):
    if self is None:
        self = DatasetCreate()
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


# 绘制清醒期的umap分布
def fig_2_b_plot_wake(self=None, ax=None):
    if self is None:
        self = DatasetCreate()
    psd, y = np.log(self.psd), self.y
    # scaler = StandardScaler()  # 标准化处理
    # psd = scaler.fit_transform(psd)
    mean = psd[:, np.logical_and(25 < self.freqs, self.freqs < 50)].mean(axis=1)  # 16至50Hz的归一化均值， 25-50的脑电波更好

    # 阈值分割
    label, proba = ostu(mean)
    wake_percent = proba
    # wake_percent = (proba - 0.5) * 2  # 16-50占比，也可以用作样本权重
    # wake_percent[wake_percent < 0] = 0  # 只考虑16-50部分的权重
    kde = KernelDensity(kernel="gaussian", bandwidth="scott").fit(self.psd_umap, sample_weight=wake_percent)
    x, y = self.psd_umap[:, 0], self.psd_umap[:, 1]
    nx, ny, dx, dy = 100, 100, (x.max() - x.min()) / 98, (y.max() - y.min()) / 98  # 100*100像素
    x, y = np.linspace(x.min() - dx, x.max() + dx, nx), np.linspace(y.min() - dy, y.max() + dx, ny)
    xv, yv = np.meshgrid(x, y)
    xy = np.vstack([xv.ravel(), yv.ravel()]).T  # 需要计算的每一个点的坐标
    p = np.exp(kde.score_samples(xy).reshape(100, 100))  # 概率密度

    psd_umap_p = np.exp(kde.score_samples(self.psd_umap))  # 概率密度
    levels = np.linspace(psd_umap_p.min(), psd_umap_p.max(), 10)  # 概率密度10等分
    self.wake = psd_umap_p > levels[1]

    from matplotlib.cm import get_cmap
    from matplotlib.colors import LinearSegmentedColormap
    cmap1 = get_cmap('Blues')
    colors1 = cmap1(np.linspace(0, 1, 10))
    cmap2 = LinearSegmentedColormap.from_list('my_cmap', ['white', 'white'], N=10)
    colors2 = cmap2(np.linspace(0, 1, 2))
    # 使用 concatenate() 函数将两个颜色列表连接一起
    colors = np.concatenate([colors2, colors1[1:]])
    # 使用 LinearSegmentedColormap 函数创建新的自定义线性颜色映射
    my_cmap = LinearSegmentedColormap.from_list('my_cmap', colors)

    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.8, 4))
    cb = ax.contourf(x, y, p, levels=levels, cmap=my_cmap)  # cmap='Blues'  # 将 cmap 参数设置为新的自定义线性颜色映射：
    # ax.scatter(self.psd_umap[self.y == 0, 0], self.psd_umap[self.y == 0, 1], c='black', s=4, label='Wake\n(open)')
    # ax.scatter(self.psd_umap[self.y != 0, 0], self.psd_umap[self.y != 0, 1], c='gray', s=4, label='Others')
    ax.scatter(self.psd_umap[self.wake, 0], self.psd_umap[self.wake, 1], c='black', s=4, label='Wake$_{\mathrm{open}}$')
    ax.scatter(self.psd_umap[~self.wake, 0], self.psd_umap[~self.wake, 1], c='gray', s=4, label='Others')
    ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
    ax.set_xlabel('UMAP 1', fontdict={'size': ft18}), ax.set_ylabel('UMAP 2', fontdict={'size': ft18})
    ax.set_xticks([]), ax.set_yticks([])
    ax.tick_params(labelsize=ft18)  # 坐标轴字体大小
    cb.set_clim(0, 0.3)
    cbar = plt.colorbar(cb, ax=ax)
    cbar.ax.set_yticklabels(['{:.2f}'.format(tick) for tick in np.arange(0, 0.3, 0.03)])
    cbar.ax.set_yticklabels(['0', '', '', '', '', '', '', '', '', '0.3'])
    # cbar.ax.set_title('Kernel Density Estimation', fontsize=12)
    cbar.ax.title.set_rotation(90)  # 将标题垂直显示
    cbar.set_label('Density', rotation=270, fontdict={'size': ft18}, labelpad=10)
    # 设置颜色条刻度标签的字体大小
    for label in cbar.ax.get_yticklabels():
        label.set_fontsize(ft18)
    legend = ax.legend(prop={'size': ft18}, handlelength=1, markerscale=4)
    legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0
    # plt.subplots_adjust(left=0.1, bottom=0.12, right=0.95, top=0.95)
    ax.xaxis.set_label_coords(0.5, -0.08), ax.yaxis.set_label_coords(-0.08, 0.5)
    if fig:
        plt.subplots_adjust(left=0.12, bottom=0.15, right=0.95, top=0.95)
        plt.savefig('./paper_figure/Wake_contourf.svg')
        plt.show()
    else:
        return cbar


# wake期 睁眼时的 gamma z-score
def fig_2_c_boxplot(sub_data=None, ax=None):
    if sub_data is None:
        sub_data = sleep_stage_all()  # 所有数据进行睡眠分期
    gamma_z = np.array([[self.gamma[self.wake == 1].mean(axis=0),
                         self.gamma[self.wake == 0].mean(axis=0)] for self in sub_data])

    # # 创建大图对象
    xticks, xlabel = [1, 2], ['Wake$_{\mathrm{open}}$', 'Others']
    xi, yi = [1, 2], np.array([gamma_z[:, i].mean() for i in range(2)])
    yerr = np.array([gamma_z[:, i].std() for i in range(2)])
    yi_test, h = yi.max() + yerr.max() + 2, 0.25
    bx = ax.boxplot([gamma_z[:, 0], gamma_z[:, 1]], notch=True, widths=0.5,
                    boxprops={'linewidth': 2, 'facecolor': color[0]}, patch_artist=True)
    # 男性与女性
    ax.plot([xi[0], xi[0], xi[1], xi[1]],
            [yi_test, yi_test + h, yi_test + h, yi_test], lw=1, c="k")
    p_value = paired_samples_test(gamma_z[:, 0], gamma_z[:, 1])  # 配对t检验(先检查是否正态分布)
    ax.text((xi[0] + xi[1]) * 0.5, yi_test + h, convert_pvalue_to_asterisks(p_value),
            ha='center', va='bottom', color="k", fontsize=ft18)
    ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
    ax.set_xticks(xticks, xlabel)
    ax.set_ylabel('Gamma power (dB)')
    ax.yaxis.set_label_coords(-0.18, 0.5)



def fig_3_plot_irasa_new(self=None, ax_hist=None, ax_umap=None):
    if self is None:
        self = DatasetCreate()
    print(self.no, 'IRASA')
    freqs, psd_aperiodic, psd_osc = yasa.irasa(self.x_signal, sf=int(self.sample_rate),
                                               hset=[1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4],
                                               band=(1, 25), win_sec=4, return_fit=False)
    psd = psd_aperiodic + psd_osc  # 功率谱密度 = 周期性功率谱密度 + 非周期性功率谱密度
    log_psd_osc = np.log(psd) - np.log(psd_aperiodic)  # 对数值相减，得到非周期性功率谱密度的log数量级

    log_psd_osc_filter_ = gaussian_filter(log_psd_osc, sigma=2)  # 高斯滤波去除噪声

    self.irasa['freqs'] = freqs
    self.irasa['psd'], self.irasa['aperiodic'], self.irasa['osc'] = psd, psd_aperiodic, psd_osc
    self.irasa['log_osc'], self.irasa['log_osc_filter'] = log_psd_osc, log_psd_osc_filter_

    time = np.linspace(0, psd.shape[0] / 120, psd.shape[0])

    # 峰值检测
    print(self.no, 'nrem')
    freqs = self.irasa['freqs']
    psd, psd_aperiodic, psd_osc = self.irasa['psd'], self.irasa['aperiodic'], self.irasa['osc']
    log_psd_osc, log_psd_osc_filter = self.irasa['log_osc'], self.irasa['log_osc_filter'].copy()
    log_psd_osc_filter[:, np.logical_or(freqs < 5, freqs > 20)] = 0  # 只关注5-20Hz内的周期信号
    # 峰值检测
    max_idx = np.argmax(log_psd_osc_filter, axis=1)
    max_peak, max_freq = np.max(log_psd_osc_filter, axis=1), freqs[max_idx]
    kde = KernelDensity(kernel="gaussian", bandwidth="scott").fit(max_freq[:, None])
    fast_sp_p0 = np.exp(kde.score_samples(freqs[:, None]))  # 估计最大峰在频率上的分布

    peak_idxs, properties = find_peaks(fast_sp_p0, prominence=0.05)  # 峰值对应的坐标,突出度最小为0.05
    sp_l_idx, sp_r_idx = np.where(freqs >= 11)[0][0], np.where(freqs <= 16)[0][-1]
    if len(peak_idxs):
        sp_idx = np.argmin(np.abs(freqs[peak_idxs] - 14))  # 快纺锤波12-16Hz，寻找与中心频率14Hz最接近的纺锤波段
        sp_freq = freqs[peak_idxs[sp_idx]]
        if 11 <= sp_freq <= 16:
            sp_l_idx, sp_r_idx = np.where(freqs >= sp_freq - 1)[0][0], np.where(freqs <= sp_freq + 1)[0][-1]

    sp_mean = np.mean(log_psd_osc_filter[:, sp_l_idx:sp_r_idx], axis=1)

    # 下一步的思路：
    # 理论上，0是区分NREM与REM的阈值
    # 实际上，该阈值应该随着分布稍作调整
    # 用高斯分布拟合 <0 和 >0 的两个分布
    # 然后用这两个分布的累积概率密度分配权重，然后KDE
    from scipy.stats import norm
    mean_0, mean_1 = np.mean(sp_mean[sp_mean <= 0]), np.mean(sp_mean[sp_mean >= 0])
    std_0, std_1 = np.std(sp_mean[sp_mean <= 0]), np.std(sp_mean[sp_mean >= 0])
    x_sp = np.linspace(np.min(sp_mean), np.max(sp_mean), 100)
    pdf_0, pdf_1 = norm.pdf(x_sp, loc=mean_0, scale=std_0), norm.pdf(x_sp, loc=mean_1, scale=std_1)
    pdf_0[x_sp < mean_0], pdf_1[x_sp > mean_1] = pdf_0.max(), pdf_1.max()
    w0, w1 = pdf_0 / (pdf_0 + pdf_1), pdf_1 / (pdf_0 + pdf_1)
    # 根据概率密度函数分配权重
    weight_0, weight_1 = norm.pdf(sp_mean, loc=mean_0, scale=std_0), norm.pdf(sp_mean, loc=mean_1, scale=std_1)
    weight_0[sp_mean < mean_0], weight_1[sp_mean > mean_1] = weight_0.max(), weight_1.max()
    weight_0_norm, weight_1_norm = weight_0 / (weight_0 + weight_1), weight_1 / (weight_0 + weight_1)

    kde0 = KernelDensity(kernel="gaussian", bandwidth="scott").fit(self.psd_umap, sample_weight=weight_0_norm)
    kde1 = KernelDensity(kernel="gaussian", bandwidth="scott").fit(self.psd_umap, sample_weight=weight_1_norm)
    p0 = np.exp(kde0.score_samples(self.psd_umap))  # 概率密度
    p1 = np.exp(kde1.score_samples(self.psd_umap))  # 概率密度
    self.n2n3 = p0 < p1  # NREM期对应的区域

    import matplotlib
    matplotlib.rcParams.update({'font.size': 18})

    # Histogram of fast spindle power 快纺锤波直方图
    # fig = plt.figure(layout="constrained", figsize=(4, 3.5))
    # gs = GridSpec(6, 2, figure=fig)

    bins = np.linspace(sp_mean.min() - 0.01, sp_mean.max() + 0.3, 40)
    if ax_hist is None:
        fig, ax_hist = plt.subplots(figsize=(6, 5))
    ax_hist.hist(sp_mean[sp_mean > 0],
                 bins=bins, label='N2N3', alpha=0.4, color='green')
    ax_hist.hist(sp_mean[sp_mean <= 0],
                 bins=bins, label='Others', alpha=0.4, color='gray')
    ax_hist.plot([0, 0], [0, 60], c='red', linestyle='--', label='Threshold')
    # ax_hist.set_xlim(0, 5)
    pdf0, pdf1 = norm.pdf(x_sp, loc=mean_0, scale=std_0), norm.pdf(x_sp, loc=mean_1, scale=std_1)
    # ax_hist.plot(x_sp, pdf1, c='green', alpha=1), ax_hist.plot(x_sp, pdf0, c='gray', alpha=1)
    legend = ax_hist.legend(framealpha=1, loc='center', bbox_to_anchor=(0.75, 0.85))
    legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0
    ax_hist.set_ylabel('Bin counts', fontdict={'size': ft18})
    ax_hist.set_xlabel('Fast spindle power (dB)', fontdict={'size': ft18})
    # ax_hist.set_title('Histogram', fontsize=ft18)
    ax_hist.spines['top'].set_visible(False), ax_hist.spines['right'].set_visible(False)
    ax_hist.yaxis.set_label_coords(-0.11, 0.5)  # 固定y坐标轴label的位置

    from matplotlib.cm import get_cmap
    from matplotlib.colors import LinearSegmentedColormap

    levels = np.linspace(p1.min(), p1.max(), 10)  # 概率密度10等分
    x, y = self.psd_umap[:, 0], self.psd_umap[:, 1]
    nx, ny, dx, dy = 100, 100, (x.max() - x.min()) / 98, (y.max() - y.min()) / 98  # 100*100像素
    x, y = np.linspace(x.min() - dx, x.max() + dx, nx), np.linspace(y.min() - dy, y.max() + dx, ny)
    xv, yv = np.meshgrid(x, y)
    xy = np.vstack([xv.ravel(), yv.ravel()]).T  # 需要计算的每一个点的坐标
    p = np.exp(kde1.score_samples(xy).reshape(100, 100))  # 概率密度

    # NREM区域
    cmap1 = get_cmap('Greens')
    colors1 = cmap1(np.linspace(0, 1, 10))
    cmap2 = LinearSegmentedColormap.from_list('my_cmap', ['white', 'white'], N=10)
    colors2 = cmap2(np.linspace(0, 1, 2))
    # 使用 concatenate() 函数将两个颜色列表连接一起
    colors = np.concatenate([colors2, colors1[1:]])
    # 使用 LinearSegmentedColormap 函数创建新的自定义线性颜色映射
    my_cmap = LinearSegmentedColormap.from_list('my_cmap', colors)
    if ax_umap:
        ax = ax_umap
    else:
        plt.show()
        fig, ax = plt.subplots(figsize=(4.8, 4))
    cb = ax.contourf(x, y, p, levels=levels, cmap=my_cmap)  # cmap='Blues'  # 将 cmap 参数设置为新的自定义线性颜色映射：

    # ax.scatter(self.psd_umap[np.logical_or(self.y == 2, self.y == 3), 0],
    #            self.psd_umap[np.logical_or(self.y == 2, self.y == 3), 1], c='black', s=4, label='N2N3')
    # ax.scatter(self.psd_umap[~np.logical_or(self.y == 2, self.y == 3), 0],
    #            self.psd_umap[~np.logical_or(self.y == 2, self.y == 3), 1], c='gray', s=4, label='Others')
    ax.scatter(self.psd_umap[self.n2n3, 0], self.psd_umap[self.n2n3, 1], c='black', s=4, label='N2N3')
    ax.scatter(self.psd_umap[~self.n2n3, 0], self.psd_umap[~self.n2n3, 1], c='gray', s=4, label='Others')
    ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_xlabel('UMAP 1', fontdict={'size': ft18})
    ax.tick_params(labelsize=ft18)  # 坐标轴字体大小
    ax.set_ylabel('UMAP 2', fontdict={'size': ft18})
    cb.set_clim(0, 0.09)
    cbar = plt.colorbar(cb)
    cbar.ax.set_yticklabels(['{:.2f}'.format(tick) for tick in np.arange(0, 0.09, 0.009)])
    cbar.ax.set_yticklabels(['0', '', '', '', '', '', '', '', '', '0.09'])
    # cbar.ax.set_title('Kernel Density Estimation', fontsize=12)
    cbar.ax.title.set_rotation(90)  # 将标题垂直显示
    cbar.set_label('Density', rotation=270, fontdict={'size': ft18}, labelpad=-2)
    for label in cbar.ax.get_yticklabels():
        label.set_fontsize(ft18)
    legend = ax.legend(prop={'size': ft18}, handlelength=1, markerscale=4)
    legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0
    # plt.tight_layout()
    # plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.94)
    plt.subplots_adjust(left=0.12, bottom=0.15, right=0.95, top=0.95)
    ax.xaxis.set_label_coords(0.5, -0.08), ax.yaxis.set_label_coords(-0.08, 0.5)
    if ax_umap is None:
        plt.show()
    return cbar


# N2N3期 的 纺锤波功率差异
def fig_3_e_spindle_boxplot(sub_data=None, ax=None):
    if sub_data is None:
        sub_data = sleep_stage_all()  # 所有数据进行睡眠分期

    sp = []
    for self in sub_data:
        sp.append([self.fast_sp[self.n2n3].mean(), self.fast_sp[~self.n2n3].mean()])
    sp = np.array(sp)

    xticks, xlabel = [1, 2], ['N2N3', 'Others']
    xi, yi = [1, 2], np.array([sp[:, i].mean() for i in range(2)])
    yerr = np.array([sp[:, i].std() for i in range(2)])
    yi_test, h = yi.max() + yerr.max() + 0.8, 0.1
    # ax.bar(xi, yi, yerr=yerr, capsize=5, alpha=0.7, edgecolor='black', label='Age')
    bx = ax.boxplot([sp[:, 0], sp[:, 1]], notch=True, widths=0.5,
                    boxprops={'linewidth': 2, 'facecolor': color[2]}, patch_artist=True)
    # 纺锤波与非纺锤波部分
    ax.plot([xi[0], xi[0], xi[1], xi[1]],
            [yi_test, yi_test + h, yi_test + h, yi_test], lw=1, c="k")
    p_value = paired_samples_test(sp[:, 0], sp[:, 1])  # 配对检验(先检查是否正态分布)

    ax.text((xi[0] + xi[1]) * 0.5, yi_test + h, convert_pvalue_to_asterisks(p_value),
            ha='center', va='bottom', color="k", fontsize=ft18)
    ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
    ax.set_xticks(xticks, xlabel), ax.set_yticks([0, 1], [0, 1])
    ax.set_ylabel('Fast spindle power (dB)')
    ax.set_ylim(-0.5, 1.8)
    ax.yaxis.set_label_coords(-0.18, 0.5)



# N3期相关绘图
def fig_4_plot_n3(ax_hist=None, ax_umap=None, no=0):
    # 数据读取
    print(no)  # 104, 105一致
    # if no in [27, 73, 78, 79, 104, 136, 137, 138, 139, 156, 157, 158, 159]:
    #     return
    # 设置全局字体大小
    import matplotlib
    matplotlib.rcParams.update({'font.size': 18})
    self = DatasetCreate(no=no, show=False)
    self.local_cluster()  # 局部类区域划分
    self.clc_irasa()  # 1/f分形信号分解
    self.find_wake()  # W期判定
    self.find_nrem_irasa_max_freq()  # nrem期判定
    # n3期判定
    # n3期判定：考虑点到3个类（w,nrem,unknow）的平均距离
    print(self.no, 'n3')
    sw = yasa.sw_detect(self.x_signal.flatten() * (10 ** 4), sf=100, freq_sw=(0.5, 2.0),
                        dur_neg=(0.3, 1.5), dur_pos=(0.1, 1.0), amp_neg=(10, 500),
                        amp_pos=(10, 500), amp_ptp=(75, 350), coupling=False,
                        remove_outliers=False, verbose=False)
    sw_mask = sw.get_mask().reshape((-1, 3000))  # 慢波对应的区间,转换成睡眠帧相应的mask
    self.so_percent = sw_mask.sum(axis=1) / 3000  # 慢波时间占比
    self.so_percent = self.so_percent - 0.10
    self.so_percent[self.so_percent < 0] = 0
    self.so_percent[self.wake == 1] = 0  # W期对应的权重置零

    kde3 = KernelDensity(kernel="gaussian", bandwidth="scott").fit(self.psd_umap, sample_weight=self.so_percent)
    psd_umap_p = np.exp(kde3.score_samples(self.psd_umap))  # 概率密度
    levels = np.linspace(psd_umap_p.min(), psd_umap_p.max(), 10)  # 概率密度10等分
    self.n3 = psd_umap_p > levels[1]

    from matplotlib.cm import get_cmap
    from matplotlib.colors import LinearSegmentedColormap

    x, y = self.psd_umap[:, 0], self.psd_umap[:, 1]
    nx, ny, dx, dy = 100, 100, (x.max() - x.min()) / 98, (y.max() - y.min()) / 98  # 100*100像素
    x, y = np.linspace(x.min() - dx, x.max() + dx, nx), np.linspace(y.min() - dy, y.max() + dx, ny)
    xv, yv = np.meshgrid(x, y)
    xy = np.vstack([xv.ravel(), yv.ravel()]).T  # 需要计算的每一个点的坐标
    p = np.exp(kde3.score_samples(xy).reshape(100, 100))  # 概率密度

    cmap1 = get_cmap('Reds')
    colors1 = cmap1(np.linspace(0, 1, 10))
    cmap2 = LinearSegmentedColormap.from_list('my_cmap', ['white', 'white'], N=10)
    colors2 = cmap2(np.linspace(0, 1, 2))
    # 使用 concatenate() 函数将两个颜色列表连接一起
    colors = np.concatenate([colors2, colors1[1:]])
    # 使用 LinearSegmentedColormap 函数创建新的自定义线性颜色映射
    my_cmap = LinearSegmentedColormap.from_list('my_cmap', colors)

    if ax_umap is None:
        fig, ax = plt.subplots(figsize=(4.8, 4))
    else:
        ax = ax_umap
    cb = ax.contourf(x, y, p, levels=levels, cmap=my_cmap)  # cmap='Blues'  # 将 cmap 参数设置为新的自定义线性颜色映射：
    ax.scatter(self.psd_umap[self.n3, 0], self.psd_umap[self.n3, 1], c='black', s=4, label='N3')
    ax.scatter(self.psd_umap[~self.n3, 0], self.psd_umap[~self.n3, 1], c='gray', s=4, label='Others')
    ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
    ax.set_xlabel('UMAP 1', fontdict={'size': ft18}), ax.set_ylabel('UMAP 2', fontdict={'size': ft18})
    ax.set_xticks([]), ax.set_yticks([])
    cb.set_clim(0, 0.4)
    cbar = plt.colorbar(cb, ax=ax_umap)
    cbar.ax.set_yticklabels(['{:.2f}'.format(tick) for tick in np.arange(0, 0.4, 0.04)])
    cbar.ax.set_yticklabels(['0', '', '', '', '', '', '', '', '', '0.4'])
    # cbar.ax.set_title('Kernel Density Estimation', fontsize=12)
    cbar.ax.title.set_rotation(90)  # 将标题垂直显示
    cbar.set_label('Density', rotation=270, fontdict={'size': ft18}, labelpad=10)
    for label in cbar.ax.get_yticklabels():
        label.set_fontsize(ft18)
    legend = ax.legend(prop={'size': ft18}, handlelength=1, markerscale=4)
    legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0
    # plt.subplots_adjust(left=0.12, bottom=0.15, right=0.95, top=0.95)
    ax.xaxis.set_label_coords(0.5, -0.08), ax.yaxis.set_label_coords(-0.08, 0.5)
    if ax_umap:
        plt.savefig('./paper_figure/N3_contourf.svg', bbox_inches='tight')
        plt.show()

    # self.n3修正，由于W期和rem期有眼动，带来类似于慢波的干扰。N3检测需要check
    conflict_mask = np.logical_and(self.wake == 1, self.n2n3 == 1)  # 既有高频活动，也有纺锤波段信号的区域
    wake_mask = np.logical_and(self.wake == 1, self.n2n3 == 0)  # 有高频活动，无纺锤波段信号（可能是Wake区）
    nrem_mask = np.logical_and(self.wake == 0, self.n2n3 == 1)  # 有纺锤波段信号，无高频活动（可能是NREM区）
    unknow_mask = np.logical_and(self.wake == 0, self.n2n3 == 0)  # 既没有高频活动，也没有纺锤波段信号（可能是REM区）

    # 计算每个N3期的点到wake区，nrem区，unknow区的距离，conflict_mask 不考虑

    label = wake_mask * 0 + nrem_mask * 2 + unknow_mask * 4
    for i, label_i in enumerate(label):
        # i表示要计算距离的点，xx_mask是一组点的坐标列表
        dis_w = np.mean(pairwise_distances(self.psd_umap[None, i, :],
                                           self.psd_umap[wake_mask], metric='euclidean'))
        dis_nrem = np.mean(pairwise_distances(self.psd_umap[None, i, :],
                                              self.psd_umap[nrem_mask], metric='euclidean'))
        dis_rem = np.mean(pairwise_distances(self.psd_umap[None, i, :],
                                             self.psd_umap[unknow_mask], metric='euclidean'))
        if conflict_mask[i]:  # 该点存在W期与N2期冲突的猜测
            label[i] = 0 if dis_w < dis_nrem else 2
        elif self.n3[i]:  # 该点没有冲突情况，但有n3期的判定
            min_dis = np.min([dis_w, dis_nrem, dis_rem])  # 最小平均距离
            if dis_w == min_dis:
                label[i] = 0  # 离W较近，判为W期
            elif dis_rem == min_dis:
                label[i] = 4  # 离rem期较近，判为rem期
            elif dis_nrem == min_dis:
                label[i] = 3  # 离nrem较近，判为n3期
    self.label = label

    # 绘制慢波占比
    n2, n3, so_percent = 71, 200, sw_mask.sum(axis=1) / 3000  # 慢波时间占比

    # ax_so = ax_hist
    if ax_hist is None:
        show = True
        fig, ax_hist = plt.subplots(figsize=(6, 5))
    else:
        show = False
    bins = np.linspace(so_percent.min() - 0.05, so_percent.max() + 0.1, 40)
    ax_hist.hist(so_percent[so_percent > 0.1],
                 bins=bins, label='N3', alpha=0.4, color='C3')
    ax_hist.hist(so_percent[so_percent <= 0.1],
                 bins=bins, label='Others', alpha=0.4, color='gray')
    ax_hist.plot([0.1, 0.1], [0, 285], c='red', linestyle='--', label='Cutoff (10%)')
    legend = ax_hist.legend(framealpha=1)
    legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0
    ax_hist.set_ylabel('Bin counts', fontdict={'size': ft18})
    ax_hist.set_xlabel(f"SO percentage", fontdict={'size': ft18})
    ax_hist.set_xticks([0, 0.2, 0.4, 0.6, 0.8], ['0%', '20%', '40%', '60%', '80%'], fontsize=ft18)
    # ax_hist.set_yticks([0, 100, 200, 300], [0, 100, 200, 300], fontsize=ft18)
    ax_hist.spines['top'].set_visible(False), ax_hist.spines['right'].set_visible(False)
    ax_hist.set_xlim(-0.05, so_percent.max() + 0.05)
    ax_hist.yaxis.set_label_coords(-0.11, 0.5)  # 固定y坐标轴label的位置
    # ax_hist.set_ylim(0, 0.8)
    legend = ax_hist.legend(framealpha=0, loc='center', bbox_to_anchor=(0.75, 0.85), bbox_transform=ax_hist.transAxes,
                            prop={'size': ft18}, columnspacing=0.5, ncol=1)
    legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0
    if show:
        plt.savefig('./paper_figure/so_percentage_n2_n3_sleep_stage.svg', bbox_inches='tight')
        plt.show()

    return cbar


def fig_4_so_boxplot(sub_data=None, ax=None):
    so = []
    for self in sub_data:
        if self.so[self.n3].shape[0] != 0:
            so.append([self.so[self.n3].mean(), self.so[~self.n3].mean()])
    so = np.array(so)
    so = np.nan_to_num(so)

    xticks, xlabel = [1, 2], ['N3', 'Others']
    xi, yi = [1, 2], np.array([so[:, i].mean() for i in range(2)])
    yi_test, h = so.max() + 0.05, 0.02
    bx = ax.boxplot([so[:, 0], so[:, 1]], notch=True, widths=0.5,
                    boxprops={'linewidth': 2, 'facecolor': color[3]}, patch_artist=True)
    # N3期与N3期以外的睡眠期
    ax.plot([xi[0], xi[0], xi[1], xi[1]],
            [yi_test, yi_test + h, yi_test + h, yi_test], lw=1, c="k")
    p_value = paired_samples_test(so[:, 0], so[:, 1])  # 配对检验(先检查是否正态分布)

    ax.text((xi[0] + xi[1]) * 0.5, yi_test + h, convert_pvalue_to_asterisks(p_value),
            ha='center', va='bottom', color="k", fontsize=ft18)
    ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
    ax.set_xticks(xticks, xlabel)
    ax.set_yticks([0, 0.2, 0.4, 0.6], ['0%', '20%', '40%', '60%'])
    ax.set_ylabel('SO percentage')
    ax.set_ylim(-0.01, 0.7)
    ax.yaxis.set_label_coords(-0.18, 0.5)



# 绘制α波较强的区域
def fig_5_plot_alpha(ax_hist=None, ax_umap=None, no=0):
    # 数据读取
    print(no)  # 104, 105一致
    self = DatasetCreate(no=no, show=False)
    self.local_cluster()  # 局部类区域划分
    self.clc_irasa()  # 1/f分形信号分解
    self.find_wake()  # W期判定
    self.find_nrem_irasa_max_freq()  # nrem期判定
    self.find_n3_yasa_check_mean_dis()  # n3期判定

    for c in np.unique(self.local_class):  # 为每一个局部类分配一个标签，按照数量的多少来确定
        c_mask = (self.local_class == c)
        wake_num = np.sum(np.logical_and(c_mask, self.label == 0))  # 局部类中W期的数量
        n2n3_num = np.sum(np.logical_and(c_mask, np.logical_or(self.label == 2, self.label == 3)))
        # 局部类中N2N3期的数量
        n1rem_num = np.sum(np.logical_and(c_mask, self.label == 4))  # 局部类中N1&REM期的数量
        # 预分类中，W期与N2不冲突，n1rem与W,N2N3不冲突
        # 优先级： W期 > N2N3期 = N1REM期  # W期数量大于0，且REM期比NREM期多，则判定为W期
        if wake_num > 0 and n1rem_num > n2n3_num:
            self.label[c_mask] = 0  # W偏向低估，所以W优先级高一些

    # 中值滤波去除毛刺
    self.label = medfilt(self.label, kernel_size=5)  # 3:0.7582646740 5:0.75839245

    # 峰值检测
    print(self.no, 'nrem+alpha')
    freqs = self.irasa['freqs']
    psd, psd_aperiodic, psd_osc = self.irasa['psd'], self.irasa['aperiodic'], self.irasa['osc']
    log_psd_osc, log_psd_osc_filter = self.irasa['log_osc'], self.irasa['log_osc_filter'].copy()
    log_psd_osc_filter[:, np.logical_or(freqs < 5, freqs > 20)] = 0  # 只关注5-20Hz内的周期信号

    # 峰值检测
    osc_std = np.std(log_psd_osc_filter, axis=1)

    # 下一步的思路：
    # 理论上，0是区分NREM与REM的阈值
    # 实际上，该阈值应该随着分布稍作调整
    # 用高斯分布拟合 <0 和 >0 的两个分布
    # 然后用这两个分布的累积概率密度分配权重，然后KDE
    th = log_psd_osc_filter.std()
    mean_0, mean_1 = np.mean(osc_std[osc_std <= th]), np.mean(osc_std[osc_std >= th])
    std_0, std_1 = np.std(osc_std[osc_std <= th]), np.std(osc_std[osc_std >= th])
    from scipy.stats import norm
    x = np.linspace(osc_std.min(), osc_std.max(), 100)
    pdf_0, pdf_1 = norm.pdf(x, loc=mean_0, scale=std_0), norm.pdf(x, loc=mean_1, scale=std_1)
    pdf_0[x < mean_0], pdf_1[x > mean_1] = pdf_0.max(), pdf_1.max()
    w0, w1 = pdf_0 / (pdf_0 + pdf_1), pdf_1 / (pdf_0 + pdf_1)
    # 根据概率密度函数分配权重
    weight_0, weight_1 = norm.pdf(osc_std, loc=mean_0, scale=std_0), norm.pdf(osc_std, loc=mean_1, scale=std_1)
    weight_0[osc_std < mean_0], weight_1[osc_std > mean_1] = weight_0.max(), weight_1.max()
    weight_0_norm, weight_1_norm = weight_0 / (weight_0 + weight_1), weight_1 / (weight_0 + weight_1)

    kde0 = KernelDensity(kernel="gaussian", bandwidth="scott").fit(self.psd_umap, sample_weight=weight_0_norm)
    kde1 = KernelDensity(kernel="gaussian", bandwidth="scott").fit(self.psd_umap, sample_weight=weight_1_norm)
    p0 = np.exp(kde0.score_samples(self.psd_umap))  # 概率密度
    p1 = np.exp(kde1.score_samples(self.psd_umap))  # 概率密度
    self.osc_mask = p0 < p1  # 周期性信号对应的区域

    new_label = self.label.copy()
    for i, label in enumerate(self.label):
        if self.osc_mask[i] == 1 and label == 4:
            new_label[i] = 0  # 阿尔法波

    from matplotlib.cm import get_cmap
    from matplotlib.colors import LinearSegmentedColormap

    x, y = self.psd_umap[:, 0], self.psd_umap[:, 1]
    nx, ny, dx, dy = 100, 100, (x.max() - x.min()) / 98, (y.max() - y.min()) / 98  # 100*100像素
    x, y = np.linspace(x.min() - dx, x.max() + dx, nx), np.linspace(y.min() - dy, y.max() + dx, ny)
    xv, yv = np.meshgrid(x, y)
    xy = np.vstack([xv.ravel(), yv.ravel()]).T  # 需要计算的每一个点的坐标
    p = np.exp(kde1.score_samples(xy).reshape(100, 100))  # 概率密度
    #
    colors = ['white', color[5]]
    my_cmap = LinearSegmentedColormap.from_list('my_cmap', colors, N=10)
    #
    levels = np.linspace(p1.min(), p1.max(), 10)  # 概率密度10等分

    if ax_umap is None:
        fig, ax_umap = plt.subplots(figsize=(4.8, 4))
        show=True
    else:
        show=False
    cb = ax_umap.contourf(x, y, p, levels=levels, cmap=my_cmap, norm=plt.Normalize(vmin=0, vmax=0.09))
    cb.set_clim(0, 0.09)
    # cmap='Blues'  # 将 cmap 参数设置为新的自定义线性颜色映射：
    # mask = np.logical_and(self.osc_mask, ~np.logical_or(self.y == 2, self.y == 3))
    wake_close = np.logical_and(self.osc_mask, ~self.n2n3)
    ax_umap.scatter(self.psd_umap[:, 0], self.psd_umap[:, 1], c='gray', s=4, label='Others')
    ax_umap.scatter(self.psd_umap[self.n2n3, 0], self.psd_umap[self.n2n3, 1], c=color[2], s=4, label='N2N3', alpha=0.5)
    # ax.scatter(self.psd_umap[mask, 0], self.psd_umap[mask, 1], c=color[0], s=4, label='Osc ∩ ~NREM')
    ax_umap.scatter(self.psd_umap[wake_close, 0], self.psd_umap[wake_close, 1], c=color[0], s=4,
               label='Wake$_{\mathrm{close}}$')
    ax_umap.spines['top'].set_visible(False), ax_umap.spines['right'].set_visible(False)
    ax_umap.set_xlabel('UMAP 1', fontdict={'size': ft18}), ax_umap.set_ylabel('UMAP 2', fontdict={'size': ft18})
    ax_umap.set_xticks([]), ax_umap.set_yticks([])
    cbar = plt.colorbar(cb, ax=ax_umap)
    cbar.ax.set_yticklabels(['{:.2f}'.format(tick) for tick in np.arange(0, 0.09, 0.009)])
    cbar.ax.set_yticklabels(['0', '', '', '', '', '', '', '', '', '0.09'])
    # cbar.ax.set_title('Kernel Density Estimation', fontsize=12)
    cbar.ax.title.set_rotation(90)  # 将标题垂直显示
    cbar.set_label('Density', rotation=270, fontdict={'size': ft18}, labelpad=0)
    for label in cbar.ax.get_yticklabels():
        label.set_fontsize(ft18)
    from matplotlib.collections import PathCollection
    from matplotlib.legend_handler import HandlerPathCollection
    def update_scatter(handle, orig):
        handle.update_from(orig)
        handle.set_sizes([64])

    legend = ax_umap.legend(handler_map={PathCollection: HandlerPathCollection(update_func=update_scatter)},
                            loc='center',
                            framealpha=0,
                            bbox_to_anchor=(0.25, 0.8), prop={'size': ft18}, handlelength=1, markerscale=4,
                            bbox_transform=ax_umap.transAxes)
    legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0
    ax_umap.xaxis.set_label_coords(0.5, -0.08), ax_umap.yaxis.set_label_coords(-0.08, 0.5)
    if ax_umap is None:
        plt.subplots_adjust(left=0.12, bottom=0.15, right=0.95, top=0.95)
        plt.savefig('./paper_figure/alpha_ax.contourf.svg', bbox_inches='tight')
        plt.show()

    bins = np.linspace(osc_std.min() - 0.01, osc_std.max() + 0.01, 40)

    if ax_hist is None:
        fig, ax_hist = plt.subplots(figsize=(4.8, 4))
        show = True
    else:
        show = False
    ax_hist.hist(osc_std[osc_std > th],
                 bins=bins, label='Osc', alpha=0.4, color=color[5])
    ax_hist.hist(osc_std[osc_std <= th],
                 bins=bins, label='Others', alpha=0.4, color='gray')
    ax_hist.plot([th, th], [0, 45], c='red', linestyle='--', label='Threshold')
    # ax_hist.set_xlim(0, 5)
    # pdf0, pdf1 = norm.pdf(x_sp, loc=mean_0, scale=std_0), norm.pdf(x_sp, loc=mean_1, scale=std_1)
    # ax_hist.plot(x_sp, pdf1, c='green', alpha=1), ax_hist.plot(x_sp, pdf0, c='gray', alpha=1)
    legend = ax_hist.legend(framealpha=0, loc='center', bbox_to_anchor=(0.75, 0.85),
                            bbox_transform=ax_hist.transAxes)
    legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0
    ax_hist.set_ylabel('Bin counts', fontdict={'size': ft18})
    ax_hist.set_xlabel('Std$_{\mathrm{osc}}$ (dB)', fontdict={'size': ft18})
    ax_hist.set_ylim(0, 60)
    # ax_hist.set_yticks([0, 15, 30, 45, 60], [0, 15, 30, 45, 60])
    # ax_hist.set_title('Histogram', fontsize=ft18)
    ax_hist.spines['top'].set_visible(False), ax_hist.spines['right'].set_visible(False)
    ax_hist.yaxis.set_label_coords(-0.11, 0.5)  # 固定y坐标轴label的位置

    if show:
        plt.show()

    return cbar


def fig_5d_osc_boxplot(sub_data=None, ax=None):
    osc_std = []
    for self in sub_data:
        if self.posc_std[self.wake_close].shape[0] != 0:
            osc_std.append([self.posc_std[self.wake].mean(), self.posc_std[self.wake_close].mean()])

    osc_std = np.array(osc_std)

    xticks, xlabel = [1, 2], ['Wake$_{\mathrm{open}}$', 'Wake$_{\mathrm{close}}$']
    xi, yi = [1, 2], np.array([osc_std[:, i].mean() for i in range(2)])
    yi_test, h = osc_std.max() + 0.05, 0.02
    bx = ax.boxplot([osc_std[:, 0], osc_std[:, 1]], notch=True, widths=0.5,
                    boxprops={'linewidth': 2, 'facecolor': color[5]}, patch_artist=True)
    # 设置箱线图框的填充颜色透明度为0.5
    bx['boxes'][0].set_facecolor(color[0])
    bx['boxes'][1].set_facecolor(color[0] + '55')
    # N3期与N3期以外的睡眠期
    ax.plot([xi[0], xi[0], xi[1], xi[1]],
            [yi_test, yi_test + h, yi_test + h, yi_test], lw=1, c="k")
    # stat, p_value = ttest_ind(osc_std_0, osc_std_1, equal_var=False)
    # stat, p_value = ttest_rel(osc_std_0, osc_std_1, alternative='less')  # 成对t检验
    p_value = paired_samples_test(osc_std[:, 0], osc_std[:, 1])  # 配对检验(先检查是否正态分布)
    ax.text((xi[0] + xi[1]) * 0.5, yi_test + h, convert_pvalue_to_asterisks(p_value),
            ha='center', va='bottom', color="k", fontsize=ft18)
    ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
    ax.set_xticks(xticks, xlabel)
    # ax.set_yticks(np.arange(0, 1.01, 0.2), [f'{i}%' for i in range(0, 101, 20)])
    ax.set_ylabel('Std$_{\mathrm{osc}}$ (dB)')
    ax.set_ylim(-0.05, 0.7)
    ax.yaxis.set_label_coords(-0.18, 0.5)


if __name__ == "__main__":
    fig_2a_plot_gamma_hist()
    fig_2_b_plot_wake()
    fig_2_c_boxplot()
    fig_3_plot_irasa_new()
    fig_4_plot_n3()
    fig_4_so_boxplot()
    fig_5_plot_alpha()