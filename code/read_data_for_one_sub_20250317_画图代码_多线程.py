"""
"""
import os
import mne
import umap
import yasa
import pickle
import numpy as np
from scipy import signal
import matplotlib.cm as cm
import scipy.stats as stats
from scipy.stats import norm, ttest_ind, ttest_rel
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from lspopt import spectrogram_lspopt
from matplotlib.colors import Normalize
from scipy.ndimage import gaussian_filter
from skimage.segmentation import watershed  # 分水岭分割算法
from skimage.feature import peak_local_max
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_distances
from scipy.signal import medfilt, find_peaks, savgol_filter
from joblib import Parallel, delayed
import sklearn
from sklearn.metrics import classification_report
import pandas as pd
from collections import Counter

# 输出正确率
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

# 第一步，先运行1_20提取原始信号.py 在目录下生成包含153次睡眠实验预处理后的信号的npz文件
npz_root = 'D:\\code\\实验室代码\\data\\eeg_fpz_cz\\'
ft18 = 18
import matplotlib as mpl

mpl.rcParams.update({'font.size': ft18})
mpl.rcParams['font.family'] = 'Arial'
mpl.rcParams['svg.fonttype'] = 'none'  # 生成的字体不再以svg的形式保存，这样可以在inkspace中开始编辑文字

color = {0: '#317EC2',  # Wake 蓝色
         1: '#F2B342',  # N1 粉色
         2: '#5AAA46',  # N2 绿色
         3: '#C03830',  # N3 红色
         4: '#825CA6',  # REM 紫色
         5: '#C43D96',  # Wake(Open eye) 黄橙色
         6: '#8D7136'}  # NREM

stage = ['Wake', 'N1', 'N2', 'N3', 'REM']


# 1.2 读取单个npz文件
def load_npz_file(sub=0, day='1'):
    """
    :param sub: 受试者编号
    :param day: 实验日期
    :return: 数据，标签，采样率
    """
    sub = str(sub)
    if len(sub) == 1:  # 0-9需要补0
        sub = '0' + sub
    if os.path.exists(npz_root + 'SC4' + sub + day + 'E0.npz'):
        npz_path = npz_root + 'SC4' + sub + day + 'E0.npz'
        print(npz_path)
    elif os.path.exists(npz_root + 'SC4' + sub + day + 'F0.npz'):
        npz_path = npz_root + 'SC4' + sub + day + 'F0.npz'
        print(npz_path)
    elif os.path.exists(npz_root + 'SC4' + sub + day + 'G0.npz'):
        npz_path = npz_root + 'SC4' + sub + day + 'G0.npz'
        print(npz_path)
    else:
        print(sub, day, 'path not found！')
        return None, None, None
    # 从路径中读取npz文件
    with np.load(npz_path) as f:
        data = f["x"]
        labels = f["y"]
        sampling_rate = f["fs"]
    return data, labels, sampling_rate


# 1.4 导入若干个受试者的数据
def get_train_test_loader(sub_no):
    data, label, sr = [], [], 0
    for s in range(sub_no // 2 + 1):
        for day in ['1', '2']:
            if len(data) == sub_no:
                x, y, sr = load_npz_file(sub=s, day=day)
                if x is not None:
                    data.append(x[:, :, 0])
                    label.append(y)
            else:
                data.append([])
                label.append([])
    return data, label, sr


# 特征降维处理
def get_umap(feature):
    scaler = StandardScaler()  # 标准化处理
    scaler.fit(feature)
    reducer = umap.UMAP(random_state=42, n_components=2)  # umap降维处理
    return reducer.fit_transform(feature)


# 直方图阈值分割
def ostu(x, return_threshold=False):
    num_all, start, end = x.shape[0], x.min(), x.max()
    # print(start, end)
    threshold, loss, interval = start, 0, (end - start) / 100  # 最大化loss(类间方差), 搜索步长间隔为1%
    # 阈值从start到end遍历搜索
    for th in np.arange(start + interval, end - interval, interval):
        c0, c1 = x[x <= th], x[x > th]  # 根据阈值划分两个类别的点
        w0, w1 = c0.shape[0] / num_all, c1.shape[0] / num_all  # 权重
        u0, u1 = c0.mean(), c1.mean()  # 均值
        temp_loss = w0 * w1 * (u0 - u1) * (u0 - u1)  # 当前阈值对应的类间方差
        if loss < temp_loss:
            # print(th, temp_loss)
            threshold, loss = th, temp_loss
    label = np.copy(x)
    label = np.zeros_like(x) + (label > threshold)  # 二值化处理
    if return_threshold:
        return label, threshold
    proba, u0, u1 = np.copy(x), x[label == 0].mean(), x[label == 1].mean()
    num_u0, num_u1 = np.sum(np.logical_and(u0 < x, x < threshold)), np.sum(np.logical_and(threshold < x, x < u1))
    for i, p in enumerate(proba):
        if p < u0:
            proba[i] = 0
        elif p > u1:
            proba[i] = 1
        elif p < threshold:
            proba[i] = 0.5 * np.sum(np.logical_and(u0 <= x, x <= p)) / num_u0
        else:
            proba[i] = 0.5 + 0.5 * np.sum(np.logical_and(threshold <= x, x <= p)) / num_u1

    # sigmoid f(x) = 1 / (1 + e^(a*(x-b)))
    # b = threshold
    # a = np.log(0.01) / (b-u1)
    # c = np.linspace(x.min(), x.max(), 100)
    # f_x = [1/ (1 + np.exp(-a*(i-b))) for i in c]
    # f_x = [2*(fx-0.5) if fx >= 0.5 else 0 for fx in f_x]
    # plt.hist(x, bins=100, density=True)
    # plt.plot(c, f_x)
    # plt.show()
    # proba = np.array([1 / (1 + np.exp(-a*(p-b))) for p in proba])
    proba = (proba - 0.5) * 2  # 25-50占比，也可以用作样本权重
    proba[proba < 0] = 0  # 只考虑25-50部分的权重
    return label, proba


# 非参数核密度估计
def kernel_density_estimation(data, weight):
    # kernel ["gaussian", "tophat", "epanechnikov", "exponential", "linear", "cosine"]
    # bandwidth=“scott”，“silverman”
    kde = KernelDensity(kernel="gaussian", bandwidth="scott").fit(data, sample_weight=weight)
    x, y = data[:, 0], data[:, 1]
    nx, ny, dx, dy = 100, 100, (x.max() - x.min()) / 98, (y.max() - y.min()) / 98  # 100*100像素
    x, y = np.linspace(x.min() - dx, x.max() + dx, nx), np.linspace(y.min() - dy, y.max() + dx, ny)
    xv, yv = np.meshgrid(x, y)
    xy = np.vstack([xv.ravel(), yv.ravel()]).T  # 需要计算的每一个点的坐标
    p = np.exp(kde.score_samples(xy).reshape(100, 100))  # 概率密度
    return xv, yv, p, kde


class DatasetCreate:
    def __init__(self, no=0, show=False):
        self.no = no  # 编号
        self.sample_rate = 100
        self.nperseg = int(30 * 100)
        self.x_signal, self.y, self.label = None, None, None
        self.f, self.t, self.Sxx = None, None, None
        self.psd, self.freqs = None, None
        self.Sxx_umap, self.psd_umap = None, None
        self.wake, self.n2n3, self.n3, self.n1rem = None, None, None, None
        self.local_class = None
        self.acc = None
        self.show = show
        self.fea = None
        self.irasa = {}
        self.osc_mask = None
        self.gamma = None  # 每个睡眠帧的gamma
        self.fast_sp = None  # 每个睡眠帧的快纺锤波功率
        self.so_percent = None  # 每个睡眠帧的慢波占比(经过-10%的裁减)
        self.so = None  # 20%
        self.posc_std = None  # 每个睡眠帧的震荡
        self.sp_11_16 = None
        self.wake_close = None
        self.gamma_z = None

        if os.path.exists(f'./data/class_data_sub_{self.no}.pkl'):
            f = open(f'./data/class_data_sub_{self.no}.pkl', 'rb')
            self.__dict__.update(pickle.load(f))
            print("Successfully read!")
        else:
            self.read_all_data_per_person()  # 读取单次实验的信号
            self.compute_sxx()  # 计算信号特征
            self.compute_psd()  # 计算功率谱密度特征
            self.save()

    def plot_psd_scaler(self):
        fig, (ax_psd, ax_psd_norm) = plt.subplots(ncols=2, figsize=(8, 4))
        psd, y = np.log(self.psd), self.y
        for i in range(5):
            mean, std = psd[y == i].mean(axis=0), psd[y == i].std(axis=0)
            ax_psd.plot(self.freqs, mean)
            ax_psd.fill_between(self.freqs, mean - std, mean + std, alpha=0.2)
        scaler = StandardScaler()  # 标准化处理
        psd_norm = scaler.fit_transform(psd)
        for i in range(5):
            mean, std = psd_norm[y == i].mean(axis=0), psd_norm[y == i].std(axis=0)
            ax_psd_norm.plot(self.freqs, mean)
            ax_psd_norm.fill_between(self.freqs, mean - std, mean + std, alpha=0.2)
        plt.tight_layout()
        plt.title(self.no)
        plt.savefig(f'./figure/no_{self.no + 1000}_psd.png', dpi=300)
        plt.close()
        # plt.show()

    def clc_irasa(self):
        print(self.no, 'IRASA')
        freqs, psd_aperiodic, psd_osc, fit_ans = yasa.irasa(self.x_signal, sf=int(self.sample_rate),
                                                            hset=[1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4],
                                                            band=(1, 25), win_sec=4, return_fit=True)
        psd = psd_aperiodic + psd_osc  # 功率谱密度 = 周期性功率谱密度 + 非周期性功率谱密度
        log_psd_osc = np.log(psd) - np.log(psd_aperiodic)  # 对数值相减，得到非周期性功率谱密度的log数量级

        log_psd_osc_filter = gaussian_filter(log_psd_osc, sigma=2)  # 高斯滤波去除噪声
        self.irasa['freqs'] = freqs
        self.irasa['psd'], self.irasa['aperiodic'], self.irasa['osc'] = psd, psd_aperiodic, psd_osc
        self.irasa['log_osc'], self.irasa['log_osc_filter'] = log_psd_osc, log_psd_osc_filter
        self.irasa['fit'] = fit_ans

    def sleep_stage(self):
        print(self.no)
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

        # 周期性信号-纺锤波信号 = α波信号
        self.find_w_n1_rem_new()

        # 区分N1REM期
        self.old_label = self.label.copy()  # 保存区分N1和REM之前的结果
        # self.label = distinguish_n1_rem_new(self)
        self.label = distinguish_n1_rem(self)

        # self.label = distinguish_n1_rem_old(self)  # 通过低通滤波后的N1&REM关系，对睡眠期做出判断
        print('acc:', np.sum((self.label == self.y)) / self.label.shape[0])

        # 预测结果
        fig, ((ax_y, ax_w, ax_nrem), (ax_n3, ax_osc, ax_pred)) = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
        for i in range(5):
            ax_y.scatter(self.psd_umap[self.y == i, 0], self.psd_umap[self.y == i, 1])
        ax_y.set_title(f'{self.no}')
        ax_w.scatter(self.psd_umap[self.wake == 0, 0], self.psd_umap[self.wake == 0, 1], c='black')
        ax_w.scatter(self.psd_umap[self.wake == 1, 0], self.psd_umap[self.wake == 1, 1], c='C0')
        ax_w.set_title(f'blue:Wake')
        ax_nrem.scatter(self.psd_umap[~self.n2n3, 0], self.psd_umap[~self.n2n3, 1], c='black')
        ax_nrem.scatter(self.psd_umap[self.n2n3, 0], self.psd_umap[self.n2n3, 1], c='C2')
        ax_nrem.set_title(f'green:NREM')
        ax_n3.scatter(self.psd_umap[~self.n3, 0], self.psd_umap[~self.n3, 1], c='black')
        ax_n3.scatter(self.psd_umap[self.n3, 0], self.psd_umap[self.n3, 1], c='C3')
        ax_n3.set_title(f'red:n3')
        ax_osc.scatter(self.psd_umap[~self.osc_mask, 0], self.psd_umap[~self.osc_mask, 1], c='black')
        ax_osc.scatter(self.psd_umap[self.osc_mask, 0], self.psd_umap[self.osc_mask, 1], c='C5')
        ax_osc.set_title(f'n1+nrem')
        for i in range(5):
            ax_pred.scatter(self.psd_umap[self.label == i, 0],
                            self.psd_umap[self.label == i, 1], c=f'C{i}', label=f'C{i}')
        ax_pred.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'./figure/no_{self.no + 1000}_sleep_stage.png', dpi=300)
        plt.close()

    def read_all_data_per_person(self):
        X_person, y_person, self.sample_rate = get_train_test_loader(self.no)
        self.x_signal, self.y = X_person[self.no], y_person[self.no]

    def compute_sxx(self):  # 计算频谱特征
        self.f, self.t, self.Sxx = spectrogram_lspopt(self.x_signal.flatten(),
                                                      self.sample_rate,
                                                      nperseg=self.nperseg,
                                                      noverlap=0)
        print(f'no.{self.no}, signal:{self.x_signal.shape}, Sxx:{self.Sxx.T.shape}')
        good_freqs = np.logical_and(self.f >= 0.2, self.f <= 30)
        Sxx = np.log(self.Sxx[good_freqs, :].T)
        self.Sxx_umap = get_umap(Sxx)

    def compute_psd(self):  # 计算功率谱密度
        self.freqs, self.psd = signal.welch(x=self.x_signal, fs=self.sample_rate, nperseg=256)
        print(f'psd {self.no}')
        good_freqs = np.logical_and(self.freqs >= 0.2, self.freqs <= 30)
        psd = np.log(self.psd[:, good_freqs])
        self.psd_umap = get_umap(psd)

    def save(self):
        f = open(f'./data/class_data_sub_{self.no}.pkl', 'wb')
        pickle.dump(self.__dict__, f)
        print('Successfully save!')

    def plot_sxx(self):
        f, t, Sxx = self.f, self.t, np.log(self.Sxx)
        trimperc = 2.5  # 百分比范围2.5%——97.5%
        v_min, v_max = np.percentile(Sxx, [0 + trimperc, 100 - trimperc])
        v_norm = Normalize(vmin=v_min, vmax=v_max)
        # 画图
        fig, ax = plt.subplots(nrows=2, figsize=(12, 8))
        im = ax[0].pcolormesh(t, f, Sxx, norm=v_norm, antialiased=True, shading="auto", cmap='RdBu_r')
        ax[0].set_xlim(0, t.max())
        ax[0].set_xticks(np.arange(0, t.max(), 3600), np.arange(0, t.max() / 3600, 1))
        ax[0].set_ylabel('Frequency (Hz)')
        ax[0].set_xlabel('Time (h)')
        ax[0].set_title(f'no_{self.no + 1000}')
        ax[1].plot(self.y)
        ax[1].set_xlim(0, len(self.y))
        ax[1].set_xticks(np.arange(0, len(self.y), 120), np.arange(0, len(self.y) / 120, 1))
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'./figure/no_{self.no + 1000}.png', dpi=300)
        plt.close()

    def plot_psd(self):
        t, f, psd = np.arange(0, self.psd.shape[0], 1), self.freqs, np.log(self.psd)
        trimperc = 2.5  # 百分比范围2.5%——97.5%
        v_min, v_max = np.percentile(psd, [0 + trimperc, 100 - trimperc])
        norm = Normalize(vmin=v_min, vmax=v_max)
        fig, ax = plt.subplots(nrows=2, figsize=(12, 8))
        im = ax[0].pcolormesh(t, f, psd.T, norm=norm, antialiased=True, shading="auto", cmap='RdBu_r')
        ax[0].set_xlim(0, t.max())
        ax[0].set_xticks(np.arange(0, t.max(), 120), np.arange(0, t.max() / 120, 1))
        ax[0].set_ylabel('Frequency (Hz)')
        ax[0].set_xlabel('Time (h)')
        ax[0].set_title(f'no_{self.no + 1000}')
        ax[1].plot(self.y)
        ax[1].set_xlim(0, len(self.y))
        ax[1].set_xticks(np.arange(0, len(self.y), 120), np.arange(0, len(self.y) / 120, 1))
        plt.tight_layout()
        plt.show()
        # plt.savefig(f'./figure/no_{self.no+1000}_psd.png', dpi=300)
        # plt.close()

    def find_wake(self):
        # gamma功率
        psd, y = np.log(self.psd), self.y
        # scaler = StandardScaler()  # 标准化处理 （不需要标准化处理，z-score差别不大）
        # psd = scaler.fit_transform(psd)
        mean = psd[:, np.logical_and(25 < self.freqs, self.freqs < 50)].mean(axis=1)  # 16至50Hz的归一化均值， 25-50的脑电波更好

        self.gamma = mean  # mean对应gamma功率
        # 阈值分割
        label, proba = ostu(mean)

        wake_percent = proba

        kde = KernelDensity(kernel="gaussian", bandwidth="scott").fit(self.psd_umap, sample_weight=wake_percent)
        psd_umap_p = np.exp(kde.score_samples(self.psd_umap))  # 概率密度
        levels = np.linspace(psd_umap_p.min(), psd_umap_p.max(), 10)  # 概率密度10等分
        self.wake = psd_umap_p > levels[1]
        self.gamma_z = mean

        bins = np.linspace(mean.min(), mean.max(), 100)  # 直方图bins范围
        fig, (ax_line, ax_bins, ax_scatter) = plt.subplots(ncols=3, figsize=(15, 5))
        ax_line.plot(mean / mean.max())
        ax_line.plot(self.y / self.y.max() + 1)
        _ = ax_bins.hist(mean[label == 0], bins=bins, alpha=0.6, label='low power')
        _ = ax_bins.hist(mean[label == 1], bins=bins, alpha=0.6, label='high power')
        ax_bins.set_title(f'bins: {16}-{50}hz')
        ax_bins.legend(loc='upper right')
        ax_scatter.scatter(self.psd_umap[self.wake != 1, 0], self.psd_umap[self.wake != 1, 1])
        ax_scatter.scatter(self.psd_umap[self.wake == 1, 0], self.psd_umap[self.wake == 1, 1])
        ax_scatter.set_title(f'no_{self.no + 1000}')
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'./figure/no_{self.no + 1000}_wake.png', dpi=300)
        plt.close()

    def find_nrem_irasa_max_freq(self):
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
        p0 = np.exp(kde.score_samples(freqs[:, None]))  # 估计最大峰在频率上的分布

        peak_idxs, properties = find_peaks(p0, prominence=0.05)  # 峰值对应的坐标,突出度最小为0.05
        sp_l_idx, sp_r_idx = np.where(freqs >= 11)[0][0], np.where(freqs <= 16)[0][-1]
        sp_mean_11_16 = np.mean(log_psd_osc_filter[:, sp_l_idx:sp_r_idx], axis=1)
        self.sp_11_16 = sp_mean_11_16
        if len(peak_idxs):
            sp_idx = np.argmin(np.abs(freqs[peak_idxs] - 14))  # 快纺锤波12-16Hz，寻找与中心频率14Hz最接近的纺锤波段
            sp_freq = freqs[peak_idxs[sp_idx]]
            if 11 <= sp_freq <= 16:
                sp_l_idx, sp_r_idx = np.where(freqs >= sp_freq - 1)[0][0], np.where(freqs <= sp_freq + 1)[0][-1]

        sp_mean = np.mean(log_psd_osc_filter[:, sp_l_idx:sp_r_idx], axis=1)
        self.fast_sp = sp_mean  # 快纺锤波功率

        # 下一步的思路：
        # 理论上，0是区分NREM与REM的阈值
        # 实际上，该阈值应该随着分布稍作调整
        # 用高斯分布拟合 <0 和 >0 的两个分布
        # 然后用这两个分布的累积概率密度分配权重，然后KDE
        th = 0
        mean_0, mean_1 = np.mean(sp_mean[sp_mean <= th]), np.mean(sp_mean[sp_mean >= th])
        std_0, std_1 = np.std(sp_mean[sp_mean <= th]), np.std(sp_mean[sp_mean >= th])
        x = np.linspace(np.min(sp_mean), np.max(sp_mean), 100)
        pdf_0, pdf_1 = norm.pdf(x, loc=mean_0, scale=std_0), norm.pdf(x, loc=mean_1, scale=std_1)
        pdf_0[x < mean_0], pdf_1[x > mean_1] = pdf_0.max(), pdf_1.max()
        w0, w1 = pdf_0 / (pdf_0 + pdf_1), pdf_1 / (pdf_0 + pdf_1)
        # 根据概率密度函数分配权重
        weight_0, weight_1 = norm.pdf(sp_mean, loc=mean_0, scale=std_0), norm.pdf(sp_mean, loc=mean_1, scale=std_1)
        weight_0[sp_mean < mean_0], weight_1[sp_mean > mean_1] = weight_0.max(), weight_1.max()
        weight_0_norm, weight_1_norm = weight_0 / (weight_0 + weight_1), weight_1 / (weight_0 + weight_1)

        fig, ((ax_psd, ax_psd_aperiodic),
              (ax_psd_osc, ax_psd_osc_filter),
              (ax_psd_mean, ax_sp),
              (ax_sp_hist, ax_y)) \
            = plt.subplots(nrows=4, ncols=2, figsize=(16, 9))
        ax_psd.imshow(np.log(psd).T, aspect='auto')
        ax_psd_aperiodic.imshow(np.log(psd_aperiodic).T, aspect='auto')
        ax_psd_osc.imshow(log_psd_osc.T, aspect='auto')
        ax_psd_osc_filter.imshow(log_psd_osc_filter.T, aspect='auto')
        ax_y.plot(self.y)
        ax_y.set_xlim(0, len(self.y))
        ax_psd_mean.plot(freqs, p0)
        ax_psd_mean.vlines(x=freqs[peak_idxs], ymin=p0[peak_idxs] - properties["prominences"],
                           ymax=p0[peak_idxs], color="C1")
        ax_psd_mean.plot([freqs[sp_l_idx], freqs[sp_r_idx]], [0, 0], c='red')
        ax_sp.plot(sp_mean)
        ax_sp.plot([0, len(self.y)], [0, 0])
        ax_sp.set_xlim(0, len(self.y))
        ax_sp_hist.hist(sp_mean, bins=100, density=True)
        ax_sp_hist.plot(x, w0)
        ax_sp_hist.plot(x, w1)
        plt.title(self.no)
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'./figure/no_{self.no + 1000}_nrem.png', dpi=300)
        plt.close()

        kde0 = KernelDensity(kernel="gaussian", bandwidth="scott").fit(self.psd_umap, sample_weight=weight_0_norm)
        kde1 = KernelDensity(kernel="gaussian", bandwidth="scott").fit(self.psd_umap, sample_weight=weight_1_norm)
        p0 = np.exp(kde0.score_samples(self.psd_umap))  # 概率密度
        p1 = np.exp(kde1.score_samples(self.psd_umap))  # 概率密度
        self.n2n3 = p0 < p1  # NREM期对应的区域

    def find_n3_yasa_check_mean_dis(self):
        # n3期判定：考虑点到3个类（w,nrem,unknow）的平均距离
        print(self.no, 'n3')
        sw = yasa.sw_detect(self.x_signal.flatten() * (10 ** 4), sf=100, freq_sw=(0.5, 2.0),
                            dur_neg=(0.3, 1.5), dur_pos=(0.1, 1), amp_neg=(10, 500),
                            amp_pos=(10, 500), amp_ptp=(75, 350), coupling=False,
                            remove_outliers=False, verbose=False)
        sw_mask = sw.get_mask().reshape((-1, 3000))  # 慢波对应的区间,转换成睡眠帧相应的mask
        self.so = sw_mask.sum(axis=1) / 3000  # 慢波时间占比

        self.so_percent = self.so - 0.10  # 减去0.1，对应的正确率是79.83%，不减去0.15，正确率是：76.81%；-0.10，对应79.87%
        self.so_percent[self.so_percent < 0] = 0
        self.so_percent[self.wake == 1] = 0  # W期对应的权重置零

        kde3 = KernelDensity(kernel="gaussian", bandwidth="scott").fit(self.psd_umap, sample_weight=self.so_percent)
        psd_umap_p = np.exp(kde3.score_samples(self.psd_umap))  # 概率密度
        levels = np.linspace(psd_umap_p.min(), psd_umap_p.max(), 10)  # 概率密度10等分
        self.n3 = psd_umap_p > levels[1]

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

        fig, (ax_line, ax_scatter, ax_fix, ax_y) = plt.subplots(ncols=4, figsize=(20, 5))
        ax_line.plot(sw_mask.sum(axis=1) / 3000)  # 原始的慢波百分比曲线
        ax_line.plot(self.so_percent + 1)
        ax_line.plot(self.y / self.y.max() + 2)
        ax_scatter.scatter(self.psd_umap[self.n3 != 1, 0], self.psd_umap[self.n3 != 1, 1])
        ax_scatter.scatter(self.psd_umap[self.n3 == 1, 0], self.psd_umap[self.n3 == 1, 1], c='C3')
        ax_scatter.set_title(f'no_{self.no + 1000}_so')
        for i in range(5):
            ax_fix.scatter(self.psd_umap[label == i, 0], self.psd_umap[label == i, 1], c=f'C{i}')
        ax_fix.set_title(f'label_n3_fix')
        for i in range(5):
            ax_y.scatter(self.psd_umap[self.y == i, 0], self.psd_umap[self.y == i, 1])
        ax_y.set_title(f'real label')
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'./figure/no_{self.no + 1000}_n3.png', dpi=300)
        plt.close()

    def find_w_n1_rem_new(self):
        # 峰值检测
        print(self.no, 'nrem+alpha')
        freqs = self.irasa['freqs']
        psd, psd_aperiodic, psd_osc = self.irasa['psd'], self.irasa['aperiodic'], self.irasa['osc']
        log_psd_osc, log_psd_osc_filter = self.irasa['log_osc'], self.irasa['log_osc_filter'].copy()
        log_psd_osc_filter[:, np.logical_or(freqs < 5, freqs > 20)] = 0  # 只关注5-20Hz内的周期信号

        # 峰值检测
        osc_std = np.std(log_psd_osc_filter, axis=1)
        self.posc_std = osc_std  # 睡眠帧的震荡信号的功率标准差

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

        fig, ((ax_psd, ax_psd_osc), (ax_psd_osc_filter, ax_sp), (ax_sp_hist, ax_y)) = \
            plt.subplots(nrows=3, ncols=2, figsize=(16, 8))
        ax_psd.imshow(np.log(psd).T, aspect='auto')
        ax_psd_osc.imshow(log_psd_osc.T, aspect='auto')
        ax_psd_osc_filter.imshow(self.irasa['log_osc_filter'].copy().T, aspect='auto')
        ax_y.plot(self.y)
        ax_y.set_xlim(0, len(self.y))
        ax_sp.plot(osc_std)
        ax_sp.plot([0, len(self.y)], [th, th])
        ax_sp.set_xlim(0, len(self.y))
        ax_sp_hist.hist(osc_std, bins=100, density=True)
        ax_sp_hist.plot(x, w0)
        ax_sp_hist.plot(x, w1)
        plt.title(self.no)
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'./figure/no_{self.no + 1000}_osc_power.png', dpi=300)
        plt.close()

        kde0 = KernelDensity(kernel="gaussian", bandwidth="scott").fit(self.psd_umap, sample_weight=weight_0_norm)
        kde1 = KernelDensity(kernel="gaussian", bandwidth="scott").fit(self.psd_umap, sample_weight=weight_1_norm)
        p0 = np.exp(kde0.score_samples(self.psd_umap))  # 概率密度
        p1 = np.exp(kde1.score_samples(self.psd_umap))  # 概率密度
        self.osc_mask = p0 < p1  # 周期性信号对应的区域
        self.wake_close = np.logical_and(self.label == 4, self.osc_mask)

        new_label = self.label.copy()
        for i, label in enumerate(self.label):
            if self.osc_mask[i] == 1 and label == 4:
                new_label[i] = 0  # 阿尔法波

        fig, (ax_scatter, ax_l_old, ax_l_new, ax_y) = plt.subplots(ncols=4, figsize=(20, 5))
        ax_scatter.scatter(self.psd_umap[~self.osc_mask, 0], self.psd_umap[~self.osc_mask, 1])
        ax_scatter.scatter(self.psd_umap[self.osc_mask, 0], self.psd_umap[self.osc_mask, 1])
        for i in range(5):
            ax_l_old.scatter(self.psd_umap[self.label == i, 0], self.psd_umap[self.label == i, 1])
        for i in range(5):
            ax_l_new.scatter(self.psd_umap[new_label == i, 0], self.psd_umap[new_label == i, 1])
        for i in range(5):
            ax_y.scatter(self.psd_umap[self.y == i, 0], self.psd_umap[self.y == i, 1])
        ax_l_old.set_title(f'old:{np.sum(self.label == self.y) / len(self.y)}')
        ax_l_new.set_title(f'new:{np.sum(new_label == self.y) / len(self.y)}')
        ax_y.set_title(f'{self.no}')
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'./figure/no_{self.no + 1000}_nrem_n1.png', dpi=300)
        plt.close()

        self.label = new_label

    def local_cluster(self):
        x_umap = self.psd_umap  # 降维后的PSD分布
        x, y, z, kde = kernel_density_estimation(data=x_umap, weight=None)  # 核密度概率估计
        levels = np.linspace(z.min(), z.max(), 10)  # 概率密度10等分
        local_max = peak_local_max(z, footprint=np.ones((5, 5)), threshold_abs=levels[1])  # 寻找局部最大值，排除过低的概率密度
        peak_point = np.zeros(z.shape, dtype=bool)
        peak_point[tuple(local_max.T)] = True  # 峰值点标记
        markers, _ = ndi.label(peak_point)  # 给峰值点打上序号
        image_mask = np.zeros(z.shape)  # 排除过低的概率密度后的图像mask区域
        image_mask[z > levels[1]] = 1
        labels = watershed(-z, markers)  # 分水岭分割算法

        fig, ax = plt.subplots(ncols=3, figsize=(24, 8))
        ax[0].contourf(x, y, z, levels=levels, cmap='Reds')  # 概率密度等高线图
        ax[0].scatter(x_umap[:, 0], x_umap[:, 1], c='black', s=2)  # 分布散点图
        ax[0].scatter(x[tuple(local_max.T)], y[tuple(local_max.T)], c='blue', s=16)  # 局部最大值
        ax[0].set_title(f'no_{self.no + 1000}')
        ax[1].pcolormesh(x, y, labels, alpha=image_mask, cmap=cm.tab20)
        ax[1].scatter(x_umap[:, 0], x_umap[:, 1], c='black', s=2)
        ax[1].scatter(x[tuple(local_max.T)], y[tuple(local_max.T)], c='blue', s=8)
        # ax[2].pcolormesh(x, y, mask_so)
        x, y = x_umap[:, 0], x_umap[:, 1]  # 按照位置映射关系给每个局部簇打上标签
        nx, ny, dx, dy = 100, 100, (x.max() - x.min()) / 98, (y.max() - y.min()) / 98  # 100*100像素
        x, y = np.linspace(x.min() - dx, x.max() + dx, nx), np.linspace(y.min() - dy, y.max() + dx, ny)
        px, py = np.copy(x_umap[:, 0]), np.copy(x_umap[:, 1])
        px, py = (px - x.min()) / (x.max() - x.min()), (py - y.min()) / (y.max() - y.min())
        px, py = np.around(px * 100), np.around(py * 100)
        self.local_class = labels[py.astype('int'), px.astype('int')]  # 每个点所属的局部类别
        ax[2].scatter(x_umap[:, 0], x_umap[:, 1], c=self.local_class, cmap=cm.tab20)
        ax[2].set_xlim(x.min(), x.max())
        ax[2].set_ylim(y.min(), y.max())
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'./figure/no_{self.no + 1000}_local_cluster.png', dpi=300)
        plt.close()

    def plot_y_pred_y_real(self):
        fig, (ax_l_new, ax_y) = plt.subplots(nrows=2, figsize=(8, 6))
        ax_l_new.plot(self.label)
        ax_l_new.set_title(f'{self.no}y pred')
        ax_l_new.set_xlim(0, len(self.y))
        ax_y.plot(self.y)
        ax_y.set_title('y_real')
        ax_y.set_xlim(0, len(self.y))
        plt.tight_layout()
        plt.savefig(f'./figure/no_{self.no + 1000}_y_pred.png', dpi=300)
        plt.close()
        # plt.show()


def distinguish_n1_rem(self, win=20, return_curve=False):
    new_label = self.label.copy()
    # 状态平滑过渡，由于把N1当做REM，N1处与平滑后的曲线差异较大。
    new_label_sg = np.convolve(self.label, np.ones(win) / win, mode='same')
    # 对于每个REM期，如果与1更近，则为N1期，如果与4更近，则为REM
    for i, l in enumerate(self.label):
        if l == 4:
            if new_label_sg[i] < 2.5:
                new_label[i] = 1
                new_label_sg = np.convolve(new_label, np.ones(win) / win, mode='same')
            else:
                new_label[i] = 4
    # wake进入N2期时,一般不直接发生转换
    # new_label[np.logical_and(new_label == 2, new_label_sg < 1.5)] = 1
    # REM结束后一般是N1期

    if return_curve:
        return new_label, new_label_sg
    else:
        return new_label


def clc_acc(sub_data, max_no=40):
    y_pred, y_real = [], []
    for no, self in enumerate(sub_data):
        if no >= max_no:
            break
        print(self.no, np.sum(self.label == self.y) / len(self.y))
        y_pred.append(self.label)
        y_real.append(self.y)
    y_pred, y_real = np.concatenate(y_pred), np.concatenate(y_real)
    return np.sum(y_pred == y_real) / y_real.shape[0]


def plot_confusion_matrix(y_p=None, y_t=None, c_m=None, name=None, title=None,
                          xlabel='Predicted label', ylabel='True label', fmt=".1f"):
    fontsize = 18
    import matplotlib
    matplotlib.rcParams.update({'font.size': fontsize})
    if c_m is None:
        print(classification_report(y_t, y_p))
        c_m = sklearn.metrics.confusion_matrix(y_t, y_p)
        c_m = c_m.astype("float") / c_m.sum(axis=1)[:, np.newaxis]
        c_m = c_m * 100
    # 输出混淆矩阵
    print(c_m)
    # 混淆矩阵
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.grid(False)
    im = ax.imshow(c_m, interpolation="nearest", cmap=plt.cm.Blues)
    # We want to show all ticks...
    if title:
        plt.title(title, fontdict={'fontsize': fontsize})
    plt.xlabel(xlabel, fontdict={'fontsize': fontsize})
    plt.ylabel(ylabel, fontdict={'fontsize': fontsize})
    plt.xticks([0, 1, 2, 3, 4], ['Wake  ', 'N1 ', 'N2 ', 'N3 ', 'REM'], fontsize=fontsize - 1)
    plt.yticks([0, 1, 2, 3, 4], stage, fontsize=fontsize)

    # Loop over data dimensions and create text annotations.
    thresh = c_m.max() / 2.0
    for i in range(c_m.shape[0]):
        for j in range(c_m.shape[1]):
            ax.text(j, i, format(c_m[i, j], fmt), ha="center", va="center",
                    color="white" if c_m[i, j] > thresh else "black", size=ft18)
    ax.set_position([0.23, 0.18, 0.75, 0.65])
    ax.yaxis.set_label_coords(-0.25, 0.5)
    plt.savefig(f'./paper_figure/cm_{name}.svg')
    plt.show()


# P值与星号的转化关系
def convert_pvalue_to_asterisks(pvalue):
    # if pvalue <= 0.0001:
    #     return "****"
    if pvalue <= 0.001:
        return "***"
    elif pvalue <= 0.01:
        return "**"
    elif pvalue <= 0.05:
        return "*"
    return "n.s."


def sleep_stage_no(no):
    self = DatasetCreate(no=no, show=False)
    self.sleep_stage()  # 睡眠分期
    return self


def sleep_stage_all():
    no_list = []
    for no in range(166):
        if no in [27, 73, 78, 79, 104, 136, 137, 138, 139, 156, 157, 158, 159]:
            continue
        no_list.append(no)
    # 多进程并行计算
    results_self = Parallel(n_jobs=12)(delayed(sleep_stage_no)(no) for no in no_list)
    return results_self


# gamma波的直方图分布
def fig_2a_plot_gamma_hist(self, ax=None):
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
def fig_2_b_plot_wake(self, ax=None):
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


def fig_3_plot_irasa_new(self, ax_hist=None, ax_umap=None):
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
    # plt.savefig('./paper_figure/N3_contourf.svg', bbox_inches='tight')
    # plt.show()

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
    # plt.savefig('./paper_figure/so_percentage_n2_n3_sleep_stage.svg', bbox_inches='tight')
    # plt.show()

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
def fig_5_plot_alpha(ax_hist, ax_umap, no=0):
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
        fig, ax = plt.subplots(figsize=(4.8, 4))
    else:
        ax = ax_umap
    cb = ax.contourf(x, y, p, levels=levels, cmap=my_cmap, norm=plt.Normalize(vmin=0, vmax=0.09))
    cb.set_clim(0, 0.09)
    # cmap='Blues'  # 将 cmap 参数设置为新的自定义线性颜色映射：
    # mask = np.logical_and(self.osc_mask, ~np.logical_or(self.y == 2, self.y == 3))
    wake_close = np.logical_and(self.osc_mask, ~self.n2n3)
    ax.scatter(self.psd_umap[:, 0], self.psd_umap[:, 1], c='gray', s=4, label='Others')
    ax.scatter(self.psd_umap[self.n2n3, 0], self.psd_umap[self.n2n3, 1], c=color[2], s=4, label='N2N3', alpha=0.5)
    # ax.scatter(self.psd_umap[mask, 0], self.psd_umap[mask, 1], c=color[0], s=4, label='Osc ∩ ~NREM')
    ax.scatter(self.psd_umap[wake_close, 0], self.psd_umap[wake_close, 1], c=color[0], s=4,
               label='Wake$_{\mathrm{close}}$')
    ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
    ax.set_xlabel('UMAP 1', fontdict={'size': ft18}), ax.set_ylabel('UMAP 2', fontdict={'size': ft18})
    ax.set_xticks([]), ax.set_yticks([])
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
                            bbox_transform=ax.transAxes)
    legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0
    ax.xaxis.set_label_coords(0.5, -0.08), ax.yaxis.set_label_coords(-0.08, 0.5)
    # plt.subplots_adjust(left=0.12, bottom=0.15, right=0.95, top=0.95)
    # plt.savefig('./paper_figure/alpha_ax.contourf.svg', bbox_inches='tight')
    # plt.show()

    bins = np.linspace(osc_std.min() - 0.01, osc_std.max() + 0.01, 40)
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


# 其他睡眠分期算法的性能
def fig_7_plot_acc_time():
    from datetime import datetime, timedelta
    from matplotlib.font_manager import FontProperties
    font = FontProperties()
    font.set_weight('bold')
    date = np.array([[datetime.strptime('20210331', "%Y%m%d"), timedelta(days=-50)],  # 'XSleepNet'
                     [datetime.strptime('20200827', "%Y%m%d"), timedelta(days=-365)],  # 'TinySleepNet'
                     [datetime.strptime('20190131', "%Y%m%d"), timedelta(days=-400)],  # 'SeqSleepNet'
                     # [datetime.strptime('20190507', "%Y%m%d"), timedelta(days=-700)],  # SleepEEGNet
                     # [datetime.strptime('20211014', "%Y%m%d"), timedelta(days=-300)],  # 'DeepSleepNet-Lite'
                     # [datetime.strptime('20200622', "%Y%m%d"), timedelta(days=-180)],  # 'IITNet'
                     [datetime.strptime('20170628', "%Y%m%d"), timedelta(days=0)],  # 'DeepSleepNet'
                     # [datetime.strptime('20220131', "%Y%m%d"), timedelta(days=-900)],  # 'SleepTransformer'
                     [datetime.strptime('20211014', "%Y%m%d"), timedelta(days=-365)],  # YASA
                     [datetime.strptime('20230901', "%Y%m%d"), timedelta(days=-500)]])  # 'UK-Sleep'
    date[:, 1] = date.sum(axis=1)

    acc = np.array([[86.3, 84, -0.9, -1.2],  # 'XSleepNet'
                    [85.4, 83.1, -0.9, -1.2],  # 'TinySleepNet'
                    [85.2, 82.6, -0.9, -1.2],  # 'SeqSleepNet'
                    # [84.3, 80.0, -0.2, -0.2],  # SleepEEGNet
                    # [84, 80.3, -1, -1],  # 'DeepSleepNet-Lite'
                    # [83.9, 0, -1, 0],  # 'IITNet'
                    [82, 77.1, -1, -1.2],  # 'DeepSleepNet'
                    # [0, 81.4, 0, -0.2],  # 'SleepTransformer'
                    [77.0, 70.6, 0, 0],  # YASA
                    [81.9, 76.1, 0, 0],  # 'UK-Sleep'
                    ])
    xy_text = acc[:, 0:2] + acc[:, 2:]
    name = ['XSleepNet',
            'TinySleepNet',
            'SeqSleepNet',
            # 'SleepEEGNet',
            # 'DeepSleepNet-Lite',
            # 'IITNet',
            'DeepSleepNet',
            # 'SleepTransformer',
            'YASA',
            'UK-Sleep']

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 4))
    ax1.spines['right'].set_visible(False), ax1.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False), ax2.spines['top'].set_visible(False)
    ax1.set_ylabel('Accuracy (%)', fontdict={'size': ft18}), ax2.set_ylabel('Accuracy (%)', fontdict={'size': ft18})
    ax1.set_title('SleepEDF-20', fontdict={'size': ft18}, y=1)
    ax2.set_title('SleepEDF-78', fontdict={'size': ft18}, y=1)
    ax1.set_xticks([datetime.strptime('20180101', "%Y%m%d"),
                    datetime.strptime('20200101', "%Y%m%d"),
                    datetime.strptime('20220101', "%Y%m%d")], [2018, 2020, 2022])
    ax2.set_xticks([datetime.strptime('20180101', "%Y%m%d"),
                    datetime.strptime('20200101', "%Y%m%d"),
                    datetime.strptime('20220101', "%Y%m%d")], [2018, 2020, 2022])
    ax1.set_yticks([78, 82, 86], [78, 82, 86]), ax2.set_yticks([72, 76, 80, 84], [72, 76, 80, 84])
    ax1.set_xlim(datetime.strptime('20170101', "%Y%m%d"), datetime.strptime('20240101', "%Y%m%d"))
    ax2.set_xlim(datetime.strptime('20170101', "%Y%m%d"), datetime.strptime('20240101', "%Y%m%d"))
    ax1.scatter(date[:-1, 0], acc[:-1, 0], s=ft18 * 8, c='C0', marker='v', label='Supervised')
    ax2.scatter(date[:-1, 0], acc[:-1, 1], s=ft18 * 8, c='C0', marker='v', label='Supervised')
    ax1.scatter(date[-1, 0], acc[-1, 0], s=ft18 * 8, c='red', marker='v', label='Unsupervised')
    ax2.scatter(date[-1, 0], acc[-1, 1], s=ft18 * 8, c='red', marker='v', label='Unsupervised')
    for i, (xy20, xy78) in enumerate(xy_text):
        ax1.text(x=date[i, 1], y=xy20, s=name[i])
        ax2.text(x=date[i, 1], y=xy78, s=name[i])
        if i == -1:
            ax1.text(x=date[i, 1], y=xy20, s=name[i], fontproperties=font)
            ax2.text(x=date[i, 1], y=xy78, s=name[i], fontproperties=font)
    # legend = ax1.legend(loc='center', handlelength=1, markerscale=1,
    #                     ncol=2, bbox_to_anchor=(0.5, 1.05), framealpha=0,
    #                     bbox_transform=ax1.transAxes, prop={'size': ft18})
    # legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0
    # legend = ax2.legend(loc='center', handlelength=1, markerscale=1,
    #                     ncol=2, bbox_to_anchor=(0.5, 1.05), framealpha=0,
    #                     bbox_transform=ax2.transAxes, prop={'size': ft18})
    # legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0
    ax1.grid(linestyle='--'), ax2.grid(linestyle='--')
    ax1.set_position([0.06, 0.1, 0.42, 0.80])
    ax2.set_position([0.56, 0.1, 0.42, 0.80])
    # ax1.set_ylim(75, 87), ax2.set_ylim(70, 85)

    plt.savefig(f'./paper_figure/acc_compare.png', dpi=800)
    plt.show()


def fig_7_b_acc_uksleep_yasa():
    sub_data = sleep_stage_all()

    y_p, y_t = [], []
    for no, self in enumerate(sub_data):
        if self.no < 40:
            # y_p_i = distinguish_n1_rem_new(self, win=20)
            y_p_i = self.label
            y_p.append(y_p_i)
            y_t.append(self.y)
    y_p, y_t = np.concatenate(y_p), np.concatenate(y_t)
    acc_20 = accuracy_score(y_p, y_t)
    k_20 = cohen_kappa_score(y_p, y_t)
    mf1_20 = f1_score(y_p, y_t, average='macro')
    plot_confusion_matrix(y_p, y_t, name='edf_20_uk_sleep', title=f'UK-Sleep\nSleepEDF-20')
    print('uk-sleep edf-20')
    print(f'acc:{acc_20:.4f} k:{k_20:.4f} mf1{mf1_20:.4f}')
    for i in range(5):
        precision = np.sum(np.logical_and(y_p == i, y_t == i)) / np.sum(y_t == i)
        recall = np.sum(np.logical_and(y_p == i, y_t == i)) / np.sum(y_p == i)
        print(f'{stage[i]}, p:{precision:.4f}, r:{recall:.4f}, '
              f'f1:{2 * precision * recall / (precision + recall):.4f}')

    y_p, y_t = [], []
    for no, self in enumerate(sub_data):
        # y_p_i = distinguish_n1_rem_new(self, win=20)
        y_p_i = self.label
        y_p.append(y_p_i)
        y_t.append(self.y)

    y_p, y_t = np.concatenate(y_p), np.concatenate(y_t)
    acc_78 = accuracy_score(y_p, y_t)
    k_78 = cohen_kappa_score(y_p, y_t)
    mf1_78 = f1_score(y_p, y_t, average='macro')
    plot_confusion_matrix(y_p, y_t, name='edf_78_uk_sleep', title=f'UK-Sleep\nSleepEDF-78')
    print('uk-sleep edf-78')
    print(f'acc:{acc_78:.4f} k:{k_78:.4f} mf1{mf1_78:.4f}')
    for i in range(5):
        precision = np.sum(np.logical_and(y_p == i, y_t == i)) / np.sum(y_t == i)
        recall = np.sum(np.logical_and(y_p == i, y_t == i)) / np.sum(y_p == i)
        print(f'{stage[i]}, p:{precision:.4f}, r:{recall:.4f}, '
              f'f1:{2 * precision * recall / (precision + recall):.4f}')

    # YASA 睡眠分期
    y_p, y_t = [], []
    for self in sub_data:
        if self.no >= 40:
            break
        info = mne.create_info(ch_names=["EEG 1"], sfreq=100, ch_types=["eeg"])  # 创建info对象
        raw = mne.io.RawArray(self.x_signal.flatten()[np.newaxis, :], info)  # 利用mne.io.RawArray创建raw对象
        sls = yasa.SleepStaging(raw, eeg_name="EEG 1")
        y_p_i = sls.predict()
        sleep_map = {'N1': 1, 'N2': 2, 'N3': 3, 'R': 4, 'W': 0}
        y_p_i = [sleep_map[i] for i in y_p_i]
        y_p.append(np.array(y_p_i))
        y_t.append(self.y)
    y_p, y_t = np.concatenate(y_p), np.concatenate(y_t)
    acc_20 = accuracy_score(y_p, y_t)
    k_20 = cohen_kappa_score(y_p, y_t)
    mf1_20 = f1_score(y_p, y_t, average='macro')
    plot_confusion_matrix(y_p, y_t, name='edf_20_yasa', title=f'YASA\nSleepEDF-20')
    print('yasa edf-20')
    print(f'acc:{acc_20:.4f} k:{k_20:.4f} mf1{mf1_20:.4f}')
    for i in range(5):
        precision = np.sum(np.logical_and(y_p == i, y_t == i)) / np.sum(y_t == i)
        recall = np.sum(np.logical_and(y_p == i, y_t == i)) / np.sum(y_p == i)
        print(f'{stage[i]}, p:{precision:.4f}, r:{recall:.4f}, '
              f'f1:{2 * precision * recall / (precision + recall):.4f}')

    y_p, y_t = [], []
    for self in sub_data:
        info = mne.create_info(ch_names=["EEG 1"], sfreq=100, ch_types=["eeg"])  # 创建info对象
        raw = mne.io.RawArray(self.x_signal.flatten()[np.newaxis, :], info)  # 利用mne.io.RawArray创建raw对象
        sls = yasa.SleepStaging(raw, eeg_name="EEG 1")
        y_p_i = sls.predict()
        sleep_map = {'N1': 1, 'N2': 2, 'N3': 3, 'R': 4, 'W': 0}
        y_p_i = [sleep_map[i] for i in y_p_i]
        y_p.append(np.array(y_p_i))
        y_t.append(self.y)

    # 保存数组到文件中
    with open('YASA.pkl', 'wb') as f:
        pickle.dump((y_p, y_t), f)

    # 从文件中加载数组
    with open('YASA.pkl', 'rb') as f:
        y_p_loaded, y_t_loaded = pickle.load(f)

    y_p, y_t = np.concatenate(y_p), np.concatenate(y_t)
    acc_78 = accuracy_score(y_p, y_t)
    k_78 = cohen_kappa_score(y_p, y_t)
    mf1_78 = f1_score(y_p, y_t, average='macro')
    plot_confusion_matrix(y_p, y_t, name='edf_78_yasa', title=f'YASA\nSleepEDF-78')
    print('yasa edf-78')
    print(f'acc:{acc_78:.4f} k:{k_78:.4f} mf1{mf1_78:.4f}')
    for i in range(5):
        precision = np.sum(np.logical_and(y_p == i, y_t == i)) / np.sum(y_t == i)
        recall = np.sum(np.logical_and(y_p == i, y_t == i)) / np.sum(y_p == i)
        print(f'{stage[i]}, p:{precision:.4f}, r:{recall:.4f}, '
              f'f1:{2 * precision * recall / (precision + recall):.4f}')


from matplotlib.ticker import FormatStrFormatter, PercentFormatter


def fig_8_plot_regression(age_dis, y_label, filename, y_lim=None, c='black',
                          yticks_format=FormatStrFormatter('%.1f'), labelpad=1):
    import statsmodels.api as sm
    import seaborn as sns

    x, y = age_dis[:, 0], age_dis[:, 1]
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    print(model.summary())
    print('stats.shapiro(age)', stats.shapiro(x), '\nspearmanr', stats.spearmanr(x, y))

    fig, ax = plt.subplots(figsize=(5, 4))
    df = pd.DataFrame(age_dis, columns=['Age', y_label])
    sns.regplot(x='Age', y=y_label, data=df, ax=ax, label=y_label, color=c)
    ax.set_ylabel(y_label, labelpad=labelpad)
    if y_lim:
        ax.set_ylim(*y_lim)
    ax.yaxis.set_major_formatter(yticks_format)
    ax.plot([30], [y.min()], label=f'y = {model.params[1]:.4f}x + {model.params[0]:.4f}', c=c)
    ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
    # legend = plt.legend(loc='center', bbox_to_anchor=(0.6, 0.9), markerscale=1.5, framealpha=0)
    # legend.get_frame().set_linewidth(0)
    ax.set_position([0.2, 0.16, 0.80, 0.80])
    plt.savefig(f'./paper_figure/{filename}.svg', bbox_inches='tight')
    plt.show()


# 探究年龄与irasa平均差值,uksleep准确率的关系
def fig_8_age_line():
    sub_data = sleep_stage_all()  # 所有数据进行睡眠分期

    import pandas as pd
    df = pd.read_excel('E:\\data\\sleepedf\\SC-subjects.xls')
    df_np = np.array(df)
    no_age, no_sex = {}, {}
    for no in range(166):
        if no in [27, 73, 78, 79, 104, 136, 137, 138, 139, 156, 157, 158, 159]:
            continue
        idx = np.where(df_np[:, 0] == no // 2)[0][0]
        no_age[no], no_sex[no] = df_np[idx, 2], df_np[idx, 3]
    print(no_age)

    age_sp_dis, sex_sp_dis, age_sp_11_16_dis = [], [], []
    age_so_dis, age_gamma_dis, age_osc_dis, age_acc = [], [], [], []
    for i, self in enumerate(sub_data):
        sex, age = no_sex[self.no], no_age[self.no]
        sp = self.fast_sp[np.logical_or(self.label == 2, self.label == 3)].mean()  # 快纺锤波功率
        sp_11_16 = self.sp_11_16[np.logical_or(self.label == 2, self.label == 3)].mean()  # 纺锤波功率
        non_sp = self.fast_sp[~np.logical_or(self.label == 2, self.label == 3)].mean()  # NREM期以外的功率
        sp_dis, sp_11_16_dis = sp - non_sp, sp_11_16 - non_sp
        age_sp_dis.append((age, sp_dis))  # 年龄与快纺锤波功率差值
        age_sp_11_16_dis.append((age, sp_11_16_dis))  # 年龄与纺锤波功率差值

        if sum(self.label == 3):
            so_dis = self.so_percent[self.label == 3].mean() - self.so_percent[self.label != 3].mean()  # 慢波占比的差值
            age_so_dis.append((age, so_dis))  # 年龄与慢波占比差值

        gamma_dis = self.gamma[self.label == 0].mean() - self.gamma[self.label != 0].mean()  # gamma功率的差值
        age_gamma_dis.append((age, gamma_dis))  # 年龄与gamma功率差值

        osc_dis = self.posc_std[self.osc_mask].mean() - self.posc_std[~self.osc_mask].mean()
        age_osc_dis.append((age, osc_dis))  # 年龄与osc_std功率差值

        age_acc.append((age, accuracy_score(self.y, self.label)))

    # fig8a年龄与gamma功率差值，绘图
    age_gamma_dis = np.array(age_gamma_dis)
    fig_8_plot_regression(age_gamma_dis, y_label='$\Delta$ Gamma power (dB)',
                          filename='gamma_age', y_lim=(0.5, 4.5), c=color[0], labelpad=10)

    # 年龄与纺锤波功率差值，绘图
    # age_sp_11_16_dis = np.array(age_sp_11_16_dis)
    # fig_8_plot_regression(age_sp_11_16_dis, y_label='Spindle$_{\mathrm{11-16Hz}}$ diff',
    #                       filename='spindle_age', y_lim=None, c='green')

    # fig8b年龄与快纺锤波功率差值，绘图
    age_sp_dis = np.array(age_sp_dis)
    fig_8_plot_regression(age_sp_dis, y_label='$\Delta$ Fast spindle power (dB)',
                          filename='fast_spindle_age', y_lim=None, c=color[2], labelpad=10)

    # fig8c年龄与慢波占比功率差值，绘图
    age_so_dis = np.array(age_so_dis)
    fig_8_plot_regression(age_so_dis, y_label='$\Delta$ SO percentage',
                          yticks_format=PercentFormatter(xmax=1, decimals=0),
                          filename='so_age', y_lim=None, c=color[3], labelpad=5)

    # fig8d年龄与osc_std差值，绘图
    age_osc_dis = np.array(age_osc_dis)
    fig_8_plot_regression(age_osc_dis, y_label='$\Delta$ Std$_{\mathrm{osc}}$ (dB)',
                          filename='osc_age', y_lim=None, c=color[5], labelpad=10)

    # fig8e年龄与准确率
    age_acc = np.array(age_acc)
    fig_8_plot_regression(age_acc, y_label='Accuracy (%)', yticks_format=PercentFormatter(xmax=1, decimals=0),
                          filename='acc_age', y_lim=[0.54, 1.05], c='black', labelpad=-5)

    # 年龄与纺锤波功率差值，绘图
    age_sp_dis = np.array(age_sp_dis)
    age_sp_11_16_dis = np.array(age_sp_11_16_dis)

    import statsmodels.api as sm
    # 定义年龄和功率
    x, y_sp, y_sp_11_16 = age_sp_dis[:, 0], age_sp_dis[:, 1], age_sp_11_16_dis[:, 1]
    # 建立线性回归模型
    X = sm.add_constant(x)
    model = sm.OLS(y_sp, X).fit()
    print(model.summary(), 'r_Pearson', np.corrcoef([x, y_sp])[0, 1])
    X = sm.add_constant(x)
    model = sm.OLS(y_sp_11_16, X).fit()
    print(model.summary(), 'r_Pearson', np.corrcoef([x, y_sp_11_16])[0, 1])
    import seaborn
    fig, ax = plt.subplots(figsize=(7, 4.5))
    df = pd.DataFrame(age_sp_dis, columns=['Age', 'Fast spindle power'])
    seaborn.regplot(x='Age', y='Fast spindle power', data=df, ax=ax, label='Fast spindle (f$_{c}$ ± 1Hz)',
                    color='red')  # 绘制散点图
    df = pd.DataFrame(age_sp_11_16_dis, columns=['Age', 'Power difference (db)'])
    seaborn.regplot(x='Age', y='Power difference (db)', data=df, ax=ax, label='Spindle (11-16Hz)',
                    color='black')  # 绘制散点图
    ax.set_ylabel('$\Delta$ Power (dB)')
    ax.plot([30], [0.5], label='y = -0.0102x + 1.1133', c='red')
    ax.plot([30], [0.5], label='y = -0.0057x + 0.6297', c='black')
    # ax.set_title('Regression with Confidence Intervals', fontsize=ft18)
    # ax.text(0.25, 0.85, "R-squared: 0.541, k: -0.0097\n"  # R^2=1 拟合程度好
    #                     "P(F-statistic) < 0.001",  # P=2.76*10-27<0.001 统计上有意义
    #         transform=plt.gca().transAxes, fontsize=ft18, c='C0')
    # ax.text(0.25, 0.7, "R-squared: 0.519, k: -0.0056\n"  # R^2=1 拟合程度好
    #                    "P(F-statistic) < 0.001",  # P=8.31e-26<0.001 统计上有意义
    #         transform=plt.gca().transAxes, fontsize=ft18, c='C1')
    ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
    legend = plt.legend(markerscale=1.5)
    legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0
    ax.set_position([0.15, 0.15, 0.80, 0.80])
    plt.savefig('./paper_figure/spindle_age.svg', bbox_inches='tight')
    plt.show()

    # 准确率与年龄组
    age_acc = [[], [], []]  # Y,M,O
    sex_acc = [[], []]  # male, female

    for self in sub_data:
        if no_age[self.no] <= 45:
            idx = 0
        elif no_age[self.no] <= 69:
            idx = 1
        else:
            idx = 2
        age_acc[idx].append(accuracy_score(self.y, self.label))
        # 性别分组
        if no_sex[self.no] == 1:  # 1对应女性
            idx = 1
        else:  # 2对应男性
            idx = 0
        # 性别与准确率的关系
        sex_acc[idx].append(accuracy_score(self.y, self.label))

    # 多组之间两两进行差异对比，先执行anova方差分析，确定有差异之后，再执行两两统计检验
    f_value, p_value = stats.f_oneway(age_acc[0], age_acc[1], age_acc[2])
    print(f'ANOVA:{p_value}')
    stat, p_y_m = ttest_ind(age_acc[0], age_acc[1], equal_var=False)  # 年轻人与中年人
    stat, p_m_o = ttest_ind(age_acc[1], age_acc[2], equal_var=False)  # 中年人与老年人
    stat, p_y_o = ttest_ind(age_acc[0], age_acc[2], equal_var=False)  # 年轻人与老年人
    from statsmodels.stats.multitest import multipletests
    # 校正p值
    reject, p_value, _, alphacBonf = multipletests([p_y_m, p_m_o, p_y_o], method='bonferroni')
    print(p_value)

    # # 创建大图对象
    fig = plt.figure(figsize=(4, 4))
    ax = plt.subplot2grid(shape=(10, 10), loc=(0, 0))
    xticks, xlabel = [1, 2, 3], ['YA', 'MA', 'EA']

    xi, yi = [1, 2, 3], np.array([np.mean(age_acc[i]) for i in range(3)])
    yerr = np.array([np.std(age_acc[i]) for i in range(3)])
    yi_test, h = yi.max() + yerr.max() + 0.02, 0.05
    # ax.bar(xi, yi, yerr=yerr, capsize=5, alpha=0.7, edgecolor='black', label='Age')
    bx = ax.boxplot([age_acc[i] for i in range(3)], notch=True, widths=0.5,
                    boxprops={'linewidth': 2, 'facecolor': 'lightblue'}, patch_artist=True)
    # 年轻人与中年人
    ax.plot([xi[0], xi[0], xi[1] - 0.1, xi[1] - 0.1],
            [yi_test, yi_test + h, yi_test + h, yi_test], lw=1, c="k")
    ax.text((xi[0] + xi[1]) * 0.5, yi_test + h, convert_pvalue_to_asterisks(p_value[0]),
            ha='center', va='bottom', color="k", fontsize=ft18)
    # 中年人与老年人
    ax.plot([xi[1] + 0.1, xi[1] + 0.1, xi[2], xi[2]],
            [yi_test, yi_test + h, yi_test + h, yi_test], lw=1, c="k")
    ax.text((xi[1] + xi[2]) * 0.5, yi_test + h, convert_pvalue_to_asterisks(p_value[1]),
            ha='center', va='bottom', color="k", fontsize=ft18)
    # 年轻人与老年人
    ax.plot([xi[0], xi[0], xi[2], xi[2]],
            [yi_test + 0.1, yi_test + h + 0.1, yi_test + h + 0.1, yi_test + 0.1], lw=1, c="k")
    ax.text((xi[0] + xi[2]) * 0.5, yi_test + h + 0.1, convert_pvalue_to_asterisks(p_value[2]),
            ha='center', va='bottom', color="k", fontsize=ft18)
    ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)

    ax.set_xticks(xticks, xlabel)
    ax.set_yticks(np.arange(0, 1.01, 0.2), [f'{i}' for i in range(0, 101, 20)])
    ax.set_ylabel('Accuracy (%)')
    ax.yaxis.set_label_coords(-0.22, 0.5)
    ax.set_ylim(0.4, 1.1)
    ax.set_position([0.25, 0.15, 0.75, 0.8])
    plt.savefig('./paper_figure/acc_ya_ma_ea.svg')
    plt.show()

    # fig8f不同性别之间的对比，没有显著差异
    # # 创建大图对象
    fig = plt.figure(figsize=(4.5, 4))
    # ax = [plt.subplot2grid(shape=(10, 10), loc=(0, i)) for i in range(1, 5)]
    ax = plt.subplot2grid(shape=(10, 10), loc=(0, 0))
    xticks, xlabel = [1, 2], ['Male', 'Female']
    xi, yi = [1, 2], np.array([np.mean(sex_acc[i]) for i in range(2)])
    yerr = np.array([np.std(sex_acc[i]) for i in range(2)])
    yi_test, h = yi.max() + yerr.max() + 0.05, 0.05
    # ax.bar(xi, yi, yerr=yerr, capsize=5, alpha=0.7, edgecolor='black', label='Age')
    bx = ax.boxplot([sex_acc[i] for i in range(2)], notch=True, widths=0.5,
                    boxprops={'linewidth': 2, 'facecolor': 'lightblue'}, patch_artist=True)
    # 男性与女性
    ax.plot([xi[0], xi[0], xi[1], xi[1]],
            [yi_test, yi_test + h, yi_test + h, yi_test], lw=1, c="k")

    statistic, p_value = shapiro_test_and_ttest_u_test(sex_acc[0], sex_acc[1])
    ax.text((xi[0] + xi[1]) * 0.5, yi_test + h, convert_pvalue_to_asterisks(p_value),
            ha='center', va='bottom', color="k", fontsize=ft18)
    ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
    ax.set_xticks(xticks, xlabel)
    ax.set_yticks(np.arange(0, 1.01, 0.2), [f'{i}' for i in range(0, 101, 20)])
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0.4, 1.1)
    ax.set_position([0.2, 0.15, 0.8, 0.8])
    plt.savefig('./paper_figure/acc_female_male.svg')
    plt.show()


def fig_s1_c_microstate():
    import pandas as pd
    df = pd.read_excel('E:\\data\\sleepedf\\SC-subjects.xls')
    df_np = np.array(df)
    sub_data = sleep_stage_all()  # 所有数据进行睡眠分期
    no_age, no_sex = {}, {}

    for no in range(166):
        if no in [27, 73, 78, 79, 104, 136, 137, 138, 139, 156, 157, 158, 159]:
            continue
        idx = np.where(df_np[:, 0] == no // 2)[0][0]
        no_age[no], no_sex[no] = df_np[idx, 2], df_np[idx, 3]
    print(no_age)

    age_lc_y = [[], [], []]  # Y, M, O
    for self in sub_data:
        no = self.no
        # 年龄分组
        if no_age[no] <= 45:
            idx = 0
        elif no_age[no] <= 69:
            idx = 1
        else:
            idx = 2
        # 年龄与局部类的重合度
        lc, y = self.local_class, self.y
        lc_y_bincount = [np.bincount(y[lc == i]) for i in np.unique(lc)]  # 每个微状态中， 5个真实睡眠期的数量
        lc_y_acc = np.array([(np.argmax(i), np.max(i)) for i in lc_y_bincount])
        age_lc_y[idx].append(lc_y_acc[:, 1].sum() / y.shape[0])

    # 多组之间两两进行差异对比，先执行anova方差分析，确定有差异之后，再执行两两统计检验
    f_value, p_value = stats.f_oneway(age_lc_y[0], age_lc_y[1], age_lc_y[2])
    print(f'ANOVA:{p_value}')
    stat, p_y_m = ttest_ind(age_lc_y[0], age_lc_y[1], equal_var=False)  # 年轻人与中年人
    stat, p_m_o = ttest_ind(age_lc_y[1], age_lc_y[2], equal_var=False)  # 中年人与老年人
    stat, p_y_o = ttest_ind(age_lc_y[0], age_lc_y[2], equal_var=False)  # 年轻人与老年人
    from statsmodels.stats.multitest import multipletests
    # 校正p值
    reject, p_value, _, alphacBonf = multipletests([p_y_m, p_m_o, p_y_o], method='bonferroni')
    print(p_value)

    import matplotlib as mpl
    mpl.rcParams.update({'font.size': ft18})
    mpl.rcParams['font.family'] = 'Arial'
    # 局部类准确率，反映人类评分者对成分相似的信号之间的判断一致性
    fig = plt.figure(figsize=(5, 5))
    # ax = [plt.subplot2grid(shape=(10, 10), loc=(0, i)) for i in range(1, 5)]
    ax = plt.subplot2grid(shape=(10, 10), loc=(0, 0))
    xticks, xlabel = [1, 2, 3], ['YA', 'MA', 'EA']

    xi, yi = [1, 2, 3], np.array([np.mean(age_lc_y[i]) for i in range(3)])
    yerr = np.array([np.std(age_lc_y[i]) for i in range(3)])
    yi_test, h = yi.max() + yerr.max() + 0.08, 0.03
    # ax.bar(xi, yi, yerr=yerr, capsize=5, alpha=0.7, edgecolor='black', label='Age')
    bx = ax.boxplot([age_lc_y[i] for i in range(3)], notch=True, widths=0.5,
                    boxprops={'linewidth': 2, 'facecolor': 'white'}, patch_artist=True)
    # 年轻人与中年人
    ax.plot([xi[0], xi[0], xi[1] - 0.1, xi[1] - 0.1],
            [yi_test, yi_test + h, yi_test + h, yi_test], lw=1, c="k")
    # stat, p_value = ttest_ind(age_lc_y[0], age_lc_y[1], equal_var=False)
    ax.text((xi[0] + xi[1]) * 0.5, yi_test + h, convert_pvalue_to_asterisks(p_value[0]),
            ha='center', va='bottom', color="k", fontsize=ft18)
    # 中年人与老年人
    ax.plot([xi[1] + 0.1, xi[1] + 0.1, xi[2], xi[2]],
            [yi_test, yi_test + h, yi_test + h, yi_test], lw=1, c="k")
    # stat, p_value = ttest_ind(age_lc_y[1], age_lc_y[2], equal_var=False)
    ax.text((xi[1] + xi[2]) * 0.5, yi_test + h, convert_pvalue_to_asterisks(p_value[1]),
            ha='center', va='bottom', color="k", fontsize=ft18)
    # 年轻人与老年人
    ax.plot([xi[0], xi[0], xi[2], xi[2]],
            [yi_test + 0.05, yi_test + h + 0.05, yi_test + h + 0.05, yi_test + 0.05], lw=1, c="k")
    # stat, p_value = ttest_ind(age_lc_y[0], age_lc_y[2], equal_var=False)
    ax.text((xi[0] + xi[2]) * 0.5, yi_test + h + 0.05, convert_pvalue_to_asterisks(p_value[2]),
            ha='center', va='bottom', color="k", fontsize=ft18)
    ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)

    ax.set_xticks(xticks, xlabel)
    ax.set_yticks(np.arange(0, 1.01, 0.2), [f'{i}' for i in range(0, 101, 20)])
    ax.set_ylabel('Overlap (%)')
    ax.set_title('Microstates')
    ax.set_ylim(0.6, 1.1)
    ax.set_position([0.25, 0.15, 0.7, 0.75])
    plt.savefig(f'./paper_figure/Microstate_Accuracy.svg')
    plt.show()


def fig_s4_plot_raw_signal_psd():
    self = DatasetCreate(no=0, show=False)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 5))
    for i in range(5):
        sig = self.x_signal[np.where(self.y == i)[0][10]]
        ax.plot(sig[0: 1000] - 0.012 * i, c=color[i])
        # axs.set_ylim(-0.01, 0.01)
        # axs.axis('off')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_yticks([-i * 0.012 for i in range(5)], ['Wake', 'N1', 'N2', 'N3', 'REM'])
    ax.set_xticks([])
    ax.tick_params(bottom=False, top=False, left=False, right=False)  # 把刻度的小短横去掉
    ax.plot([1010, 1010], [-0.030, -0.042], c='black')
    ax.plot([900, 1000], [-0.055, -0.055], c='black')
    ax.text(942, -0.059, "1s", fontsize=ft18, c='black')
    ax.text(1015, -0.036, "100μV", fontsize=ft18, c='black', rotation=90, verticalalignment='center')
    ax.set_xlim(-10, 1011)
    plt.tight_layout()
    plt.savefig(f'./paper_figure/raw_signal.svg')
    plt.show()


# 绘制状态连接概率图（五角星）
def my_arrow_graph(transition_probabilities, title=None, save=False):
    # 计算五角星顶点的角度
    coords = np.array([(np.cos(i * 2 * np.pi / 5 + 0.5 * np.pi),
                        np.sin(i * 2 * np.pi / 5 + 0.5 * np.pi)) for i in range(5)])

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_xlim(-1, 1), ax.set_ylim(-1, 1), ax.set_xticks([]), ax.set_yticks([])
    # 隐藏边框
    ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False), ax.spines['left'].set_visible(False)
    # 设置坐标轴刻度线的位置
    ax.tick_params(axis='both', length=0)

    for i in range(5):
        ax.text(coords[i, 0], coords[i, 1], stage[i], color=color[i], size=ft18 * 1.5,
                horizontalalignment='center', verticalalignment='center', weight='bold')
    for i in range(5):
        for j in range(5):
            if i != j:
                np.fill_diagonal(transition_probabilities, 0)  # 对角线元素置零
                alpha = transition_probabilities[i, j] / transition_probabilities.max()  # 连接透明度
                if alpha > 0.5:
                    alpha = 1
                elif alpha > 0.1:
                    alpha = 0.5
                else:
                    alpha = 0.05
                dx, dy = coords[j][0] - coords[i][0], coords[j][1] - coords[i][1]
                angle, length = np.arctan2(dy, dx), np.hypot(dx, dy)
                # cos, sin = (coords[j] - coords[i]) / length
                x, y = coords[i, 0] + np.cos(angle) * length * 0.2, coords[i, 1] + np.sin(angle) * length * 0.2
                x, y = x + np.cos(angle + np.pi / 2) * length * 0.01, y + np.sin(angle + np.pi / 2) * length * 0.01
                ax.arrow(x, y, np.cos(angle) * length * 0.6, np.sin(angle) * length * 0.6,  # x,y起点，x,y长度
                         fc=color[i], ec=color[i],  # 箭头的填充颜色（face color）, 箭头的边框颜色（edge color）
                         alpha=alpha, width=0.04,  # 透明度，箭头宽度
                         head_width=0.1, head_length=0.1, shape='right',  # shape : {'full', 'left', 'right'}
                         length_includes_head=True)
                if alpha > 0.1:
                    rotation = np.degrees(np.arctan(dy / dx))
                    x, y = coords[i, 0] + np.cos(angle) * length * 0.5, coords[i, 1] + np.sin(angle) * length * 0.5
                    x, y = x + np.cos(angle + np.pi / 2) * length * 0.08, y + np.sin(angle + np.pi / 2) * length * 0.08
                    ax.text(x, y, '{:.1%}'.format(transition_probabilities[i, j]), size=ft18 - 4,
                            ha='center', va='center', color='black', rotation=rotation)
    ax.text(0, -1.1, title, size=ft18 + 2, ha='center', va='center', color='black')
    ax.set_position([0.08, 0.1, 0.8, 0.8])
    if save:
        plt.savefig(f'./paper_figure/{title}_stage_transition_probabilities.svg', bbox_inches='tight')
    plt.show()


# 状态转移
def state_transition():
    sub_data = sleep_stage_all()  # 所有数据进行睡眠分期
    transition_matrix_y, transition_matrix_label = np.zeros((5, 5)), np.zeros((5, 5))
    for self in sub_data:
        # 统计状态转移次数
        for i in range(len(self.y) - 1):
            # if self.y[i] != self.y[i + 1]:  # 发生状态转换时的概率统计
            transition_matrix_y[self.y[i], self.y[i + 1]] += 1
            # if self.label[i] != self.label[i+1]:   # 发生状态转换时的概率统计
            transition_matrix_label[self.label[i], self.label[i + 1]] += 1

    transition_probabilities_y = transition_matrix_y / np.sum(transition_matrix_y, axis=1, keepdims=True)
    transition_probabilities_laebl = transition_matrix_label / np.sum(transition_matrix_label, axis=1, keepdims=True)
    plot_confusion_matrix(c_m=transition_probabilities_y * 100, title='Transition probability\nhuman labeled',
                          name='human_labeled_tp', xlabel='To stage', ylabel='From stage', fmt=".1f")
    my_arrow_graph(transition_probabilities_y, title='Human labeled', save=True)
    plot_confusion_matrix(c_m=transition_probabilities_laebl * 100, title='Transition probability\nUK-sleep',
                          name='uk_sleep_tp', xlabel='To stage', ylabel='From stage', fmt=".1f")
    my_arrow_graph(transition_probabilities_laebl, title='UK-Sleep', save=True)


def fig_s2():
    import pandas as pd
    from collections import Counter
    df = pd.read_excel('E:\\data\\sleepedf\\SC-subjects.xls')
    df_np = np.array(df)
    sub_data, no_age = [], {}
    transition_matrix = np.zeros((3, 5, 5))  # 年轻人18-45，中年人46-69，老年人69以上
    state_count = np.zeros((3, 5))  # 三组被试的状态数统计
    count_all = {}  # 每个被试的各睡眠期的数量
    for no in range(166):
        if no in [27, 73, 78, 79, 104, 136, 137, 138, 139, 156, 157, 158, 159]:
            continue
        idx = np.where(df_np[:, 0] == no // 2)[0][0]
        no_age[no] = df_np[idx, 2]  # 年龄
        self = DatasetCreate(no=no, show=False)
        sub_data.append(self)
        if no_age[no] <= 45:
            idx = 0
        elif no_age[no] <= 69:
            idx = 1
        else:
            idx = 2
        # 统计状态转移次数
        for i in range(len(self.y) - 1):
            if self.y[i] != self.y[i + 1]:  # 发生状态转换时的概率统计
                transition_matrix[idx, self.y[i], self.y[i + 1]] += 1
        count = Counter(self.y)
        count_all[no] = [count[i] for i in range(5)]  # 每个被试的状态数量
        for s in range(5):
            state_count[idx, s] += count[s]

    # 计算状态转移概率矩阵
    # transition_probabilities = np.zeros_like(transition_matrix)
    # title = ['YA', 'MA', 'EA']
    # for i, t_m in enumerate(transition_matrix):
    #     transition_probabilities[i] = t_m / np.sum(t_m, axis=1, keepdims=True)
    #     plot_confusion_matrix(c_m=transition_probabilities[i]*100, title='Transition probabilities',
    #                           name='human_labeled_tp', xlabel='To stage', ylabel='From stage', fmt=".1f")
    #     my_arrow_graph(transition_probabilities[i], title=title[i], save=True)

    # for i in range(3):
    #     fig, ax = plt.subplots(figsize=(3.5, 3.5))
    #     ax.pie(state_count[i][1:5],
    #            labels=stage[1:5],
    #            colors=[color[j] for j in range(1, 5)],
    #            explode=(0.1, 0.1, 0.1, 0.1),
    #            startangle=180,
    #            shadow=False,
    #            autopct='%1.1f%%',
    #            pctdistance=0.6,
    #            textprops={'size': ft18}, wedgeprops={'alpha': 0.5})
    #     # ax.set_title(f'group {i}', fontsize=ft18)
    #     # 设置注释相对位置
    #     # plt.tight_layout()
    #     plt.show()

    # 直方图绘制
    y, m, o = [], [], []
    for no in range(166):
        if no in [27, 73, 78, 79, 104, 136, 137, 138, 139, 156, 157, 158, 159]:
            continue
        if no_age[no] <= 45:
            y.append(count_all[no])
        elif no_age[no] <= 69:
            m.append(count_all[no])
        else:
            o.append(count_all[no])
    y, m, o = np.array(y), np.array(m), np.array(o)
    y = y / y[:, 1:].sum(axis=1, keepdims=True)  # 统计时排除Wake期
    m = m / m[:, 1:].sum(axis=1, keepdims=True)  # 统计时排除Wake期
    o = o / o[:, 1:].sum(axis=1, keepdims=True)  # 统计时排除Wake期

    # # 创建大图对象
    fig = plt.figure(figsize=(12, 4))
    # ax = [plt.subplot2grid(shape=(10, 10), loc=(0, i)) for i in range(1, 5)]
    ax = plt.subplot2grid(shape=(10, 10), loc=(0, 0))
    xticks, xlabel = [], ['YA', 'MA', 'EA'] * 4
    for i in range(1, 5):
        p = np.zeros((5, 3))
        # 多组之间两两进行差异对比，先执行anova方差分析，确定有差异之后，再执行两两统计检验

        f_value, p_value = stats.f_oneway(y[:, i], m[:, i], o[:, i])
        print(f'ANOVA:{f_value, p_value}')
        stat, p_y_m = ttest_ind(y[:, i], m[:, i], equal_var=False)  # 年轻人与中年人
        stat, p_m_o = ttest_ind(m[:, i], o[:, i], equal_var=False)  # 中年人与老年人
        stat, p_y_o = ttest_ind(y[:, i], o[:, i], equal_var=False)  # 年轻人与老年人
        from statsmodels.stats.multitest import multipletests
        # 校正p值
        reject, p[i, :], _, alphacBonf = multipletests([p_y_m, p_m_o, p_y_o], method='bonferroni')
        print(p[i, :])

        xi, yi = np.arange(4 * (i - 1), 4 * i - 1), np.array([y[:, i].mean(), m[:, i].mean(), o[:, i].mean()])
        yerr = np.array([y[:, i].std(), m[:, i].std(), o[:, i].std()])
        yi_test, h = yi.max() + yerr.max() + 0.02, 0.05
        xticks += [j for j in xi]
        ax.bar(xi, yi, yerr=yerr, capsize=5, alpha=0.7, facecolor=color[i], label=stage[i])
        # 年轻人与中年人
        ax.plot([xi[0], xi[0], xi[1] - 0.1, xi[1] - 0.1],
                [yi_test, yi_test + h, yi_test + h, yi_test], lw=1, c="k")
        # stat, p_value = ttest_ind(y[:, i], m[:, i], equal_var=False)
        ax.text((xi[0] + xi[1]) * 0.5, yi_test + h, convert_pvalue_to_asterisks(p[i, 0]),
                ha='center', va='bottom', color="k", fontsize=ft18)
        # 中年人与老年人
        ax.plot([xi[1] + 0.1, xi[1] + 0.1, xi[2], xi[2]],
                [yi_test, yi_test + h, yi_test + h, yi_test], lw=1, c="k")
        # stat, p_value = ttest_ind(m[:, i], o[:, i], equal_var=False)
        ax.text((xi[1] + xi[2]) * 0.5, yi_test + h, convert_pvalue_to_asterisks(p[i, 1]),
                ha='center', va='bottom', color="k", fontsize=ft18)
        # 年轻人与老年人
        ax.plot([xi[0], xi[0], xi[2], xi[2]],
                [yi_test + 0.1, yi_test + h + 0.1, yi_test + h + 0.1, yi_test + 0.1], lw=1, c="k")
        # stat, p_value = ttest_ind(y[:, i], o[:, i], equal_var=False)
        ax.text((xi[0] + xi[2]) * 0.5, yi_test + h + 0.1, convert_pvalue_to_asterisks(p[i, 2]),
                ha='center', va='bottom', color="k", fontsize=ft18)
    ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
    ax.set_ylim(0, 1)
    ax.set_xticks(xticks, xlabel)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], [f'{i}%' for i in np.arange(0, 101, 20)])
    ax.set_ylabel('Percentage (%)')
    ax.legend(loc='center', handlelength=1, markerscale=1, ncol=4,
              bbox_to_anchor=(0.3, 0.95), framealpha=0, prop={'size': ft18})
    ax.text(10, 0.75, 'Young Adults(YA) \nMiddle-aged Adults(MA)\nElderly Adults(EA)',
            ha='left', va='bottom', color="k", fontsize=ft18)
    # 调整子图的大小和位置(x,y,width,heigh)
    # for i in range(4):
    #     ax[i].set_position([i / 4 + 0.04, 0.1, 1 / 4 * 0.8, 0.85])
    ax.set_position([0.1, 0.1, 0.85, 0.85])
    plt.savefig(f'./paper_figure/stage_percentage.svg', bbox_inches='tight')
    plt.show()
    #
    # transition_probabilities = transition_matrix / np.sum(transition_matrix, axis=2, keepdims=True)


# 统计检验，先判断是否符合正态分布，然后再做相应的统计检验
def shapiro_test_and_ttest_u_test(array1, array2, alpha=0.05):
    # 对每个数组进行正态性检验
    normal_test1 = stats.shapiro(array1)
    normal_test2 = stats.shapiro(array2)
    print('p1:', normal_test1.pvalue, ' - ', 'p2', normal_test2.pvalue)

    # 判断两个数组是否都服从正态分布
    both_normal = normal_test1.pvalue > alpha and normal_test2.pvalue > alpha

    if both_normal:
        print("服从正态分布，执行双样本t检验")
        t_statistic, p_value = stats.ttest_ind(array1, array2, equal_var=False)
        test_type = "t-test"
    else:
        print("不服从正态分布，执行Mann-Whitney U检验")
        u_statistic, p_value = stats.mannwhitneyu(array1, array2)
        test_type = "Mann-Whitney U test"
    print(test_type, 'p_value:', p_value)
    return test_type, p_value


def paired_samples_test(pair_data1, pair_data2, alpha=0.05):
    # 计算配对样本的差值
    differences = np.array(pair_data2) - np.array(pair_data1)

    # 进行Shapiro-Wilk正态性检验
    _, shapiro_pvalue = stats.shapiro(differences)
    print(stats.shapiro(differences))

    # 如果差值符合正态分布
    if shapiro_pvalue > alpha:
        # 执行配对样本t检验
        t_statistic, p_value = stats.ttest_rel(pair_data1, pair_data2)
        print(f"差值服从正态分布，进行配对样本t检验。\nT检验统计量为 {t_statistic}，p值为 {p_value:}")
    else:
        # 如果差值不符合正态分布，执行Wilcoxon符号秩检验
        wilcoxon_statistic, p_value = stats.wilcoxon(pair_data1, pair_data2, zero_method='pratt')
        print(f"差值不服从正态分布，进行Wilcoxon符号秩检验。\nWilcoxon统计量为 {wilcoxon_statistic}，p值为 {p_value}")
    return p_value


# def table_3():
#     sub_data = sleep_stage_all()
#     u_statistic, p_value = stats.mannwhitneyu(array1, array2)


# 该图弃用
def fig_7a_old():
    import seaborn as sns
    import matplotlib
    from statannotations.Annotator import Annotator
    import itertools
    import glob
    from tqdm import tqdm
    import sklearn.metrics as skmetrics

    # 加载UK-Sleep的准确率（153晚）
    if not os.path.exists('UK-Sleep.pkl'):
        sub_data = sleep_stage_all()
        y_p = [self.label for self in sub_data]
        y_t = [self.y for self in sub_data]
        # 保存数组到文件中
        with open('UK-Sleep.pkl', 'wb') as f:
            pickle.dump((y_p, y_t), f)

    # 从文件中加载数组
    with open('UK-Sleep.pkl', 'rb') as f:
        uk_sleep_yp, uk_sleep_yt = pickle.load(f)
    uk_sleep_acc = [skmetrics.accuracy_score(y_true=yt, y_pred=yp) for yt, yp in zip(uk_sleep_yt, uk_sleep_yp)]
    uk_sleep_mf1 = [skmetrics.f1_score(y_true=yt, y_pred=yp, average="macro") for yt, yp in
                    zip(uk_sleep_yt, uk_sleep_yp)]

    # 加载YASA的准确率（153晚）
    if not os.path.exists('YASA.pkl'):
        yp, yt, sub_data = [], [], sleep_stage_all()
        for self in sub_data:
            info = mne.create_info(ch_names=["EEG 1"], sfreq=100, ch_types=["eeg"])  # 创建info对象
            raw = mne.io.RawArray(self.x_signal.flatten()[np.newaxis, :], info)  # 利用mne.io.RawArray创建raw对象
            sls = yasa.SleepStaging(raw, eeg_name="EEG 1")
            yp_i = sls.predict()
            sleep_map = {'N1': 1, 'N2': 2, 'N3': 3, 'R': 4, 'W': 0}
            yp_i = [sleep_map[i] for i in yp_i]
            yp.append(np.array(yp_i))
            yt.append(self.y)

        # 保存数组到文件中
        with open('YASA.pkl', 'wb') as f:
            pickle.dump((y_p, y_t), f)

    # 从文件中加载数组
    with open('YASA.pkl', 'rb') as f:
        yasa_yp, yasa_yt = pickle.load(f)
    yasa_acc = [skmetrics.accuracy_score(y_true=yt, y_pred=yp) for yt, yp in zip(yasa_yt, yasa_yp)]
    yasa_mf1 = [skmetrics.f1_score(y_true=yt, y_pred=yp, average="macro") for yt, yp in zip(yasa_yt, yasa_yp)]

    # 加载tinysleepnet的准确率（153晚）
    # 获取目录下所有后缀为.npz的文件路径
    file_paths = glob.glob('.\\tinysleepnet_predict\\pred*.npz')
    yp_fold, yt_fold = [[] for _ in range(20)], [[] for _ in range(20)]
    for file in tqdm(file_paths):
        f = np.load(file)
        yp, yt, fold = f['y_pred'], f['y_true'], int(file.split('_')[-1].split('.')[0])
        yp_fold[fold].append(yp), yt_fold[fold].append(yt)
    # tinysleepnet在20上训练，在78上的平均准确率
    # np.mean([skmetrics.accuracy_score(np.concatenate(yp_fold[i][:]), np.concatenate(yt_fold[i][:])) for i in range(20)])

    tinysleep_acc = [0] * 153
    tinysleep_mf1 = [0] * 153
    for i in range(153):
        yt_sub = np.concatenate([yt_f[i] for yt_f in yt_fold])
        yp_sub = np.concatenate([yp_f[i] for yp_f in yp_fold])
        tinysleep_acc[i] = skmetrics.accuracy_score(y_true=yt_sub, y_pred=yp_sub)
        tinysleep_mf1[i] = skmetrics.f1_score(y_true=yt_sub, y_pred=yp_sub, average="macro")
    methods_20 = ['YASA', 'UK-Sleep']
    methods_78 = ['YASA', 'TinySleepNet ', ' UK-Sleep']
    methods = [methods_20, methods_20, methods_78, methods_78]
    pairs = [list(itertools.combinations(method, 2)) for method in methods]

    # 颜色设置
    color_20 = sns.color_palette([color[2], color[4]])
    color_78 = sns.color_palette([color[2], color[3], color[4]])
    colors = [color_20, color_20, color_78, color_78]

    df_acc_20 = pd.DataFrame(
        {'Method': ['YASA'] * 39 + ['UK-Sleep'] * 39, 'acc_sub': yasa_acc[0:39] + uk_sleep_acc[0:39], })
    df_mf1_20 = pd.DataFrame(
        {'Method': ['YASA'] * 39 + ['UK-Sleep'] * 39, 'acc_sub': yasa_mf1[0:39] + uk_sleep_mf1[0:39], })
    df_acc_78 = pd.DataFrame(
        {'Method': ['YASA'] * (153 - 39) + ['TinySleepNet '] * (153 - 39) + [' UK-Sleep'] * (153 - 39),
         'acc_sub': yasa_acc[39:] + tinysleep_acc[39:] + uk_sleep_acc[39:], })
    df_mf1_78 = pd.DataFrame(
        {'Method': ['YASA'] * (153 - 39) + ['TinySleepNet '] * (153 - 39) + [' UK-Sleep'] * (153 - 39),
         'acc_sub': yasa_mf1[39:] + tinysleep_mf1[39:] + uk_sleep_mf1[39:], })
    datas = [df_acc_20, df_mf1_20, df_acc_78, df_mf1_78]

    # 画图代码
    fig, ax = plt.subplots(ncols=4, figsize=(16, 5))

    for data, ax_i, color_pal in zip(datas, ax, colors):
        sns.swarmplot(x='Method', y='acc_sub', data=data, palette=color_pal, ax=ax_i, zorder=-1, size=4)
        # sns.stripplot(x='Method', y='acc_sub', data=data, palette=color_pal,ax=ax_i,zorder=-1, size=4)

    for ax_i in ax:
        for dots in ax_i.collections:  # 遍历所有点的颜色
            facecolors = dots.get_facecolors()
            dots.set_edgecolors(facecolors.copy())  # 点的边缘颜色
            dots.set_facecolors('none')  # 点的中心颜色
            dots.set_linewidth(1)  # 点的线宽

    for data, ax_i, w, color_pal in zip(datas, ax, [0.7, 0.7, 0.7, 0.7], colors):
        sns.violinplot(data=data, x='Method', y='acc_sub', palette=color_pal, cut=2, ax=ax_i, width=w, linewidth=3)

    # 清除violinplot的facecolor
    for ax_i in ax:
        color_list = []
        for collection in ax_i.collections:
            if isinstance(collection, matplotlib.collections.PolyCollection):
                color_list.append(collection.get_facecolor())  # 添加颜色
                collection.set_edgecolor(color_list[-1])
                collection.set_facecolor('none')

        # if len(ax[0].lines) == 2 * len(colors):  # suppose inner=='box'
        #     for lin1, lin2, c in zip(ax[0].lines[::2], ax[0].lines[1::2], colors):
        #         lin1.set_color(c)
        #         lin2.set_color(c)

    # ax.set_title(title,fontsize=20)
    for ax_i in ax:
        ax_i.spines['top'].set_visible(False), ax_i.spines['right'].set_visible(False)

    ax[0].set_yticks([0.40, 0.60, 0.80, 1], [40, 60, 80, 100])
    ax[1].set_yticks([0.40, 0.60, 0.80, 1], [0.40, 0.60, 0.80, 1])
    ax[2].set_yticks([0.40, 0.60, 0.80, 1], [40, 60, 80, 100])
    ax[3].set_yticks([0.20, 0.40, 0.60, 0.80, 1], [0.20, 0.40, 0.60, 0.80, 1])
    # ax[2].set_xlim(-0.5,2.6)
    # ax[3].set_xlim(-0.6, 2.5)
    for xlabel, ax_i in zip(['SleepEDF-20', 'SleepEDF-20', 'SleepEDF-78$^*$', 'SleepEDF-78$^*$'], ax):
        ax_i.set_xlabel(None, fontdict={'size': ft18})
    for ylabel, ax_i in zip(['Accuracy (%)', 'Macro-F1', 'Accuracy (%)', 'Macro-F1'], ax):
        ax_i.set_ylabel(ylabel, fontdict={'size': ft18})

    for data, pair, method, ax_i in zip(datas, pairs, methods, ax):
        annotator = Annotator(ax_i, data=data, x='Method', y='acc_sub', pairs=pair, order=method)
        # 检验方法：t-test_ind, t-test_paired, 'Mann-Whitney
        # # acc 两两检验
        annotator.configure(test='t-test_paired',  #
                            pvalue_format={'text_format': 'star'},  # 使用星星表示
                            line_height=0.02,
                            hide_non_significant=False, loc='outside')
        #  在 annotator的configure模块中搜索ymax_in_range_x1_x2 + offset，可以调整间距
        annotator.apply_and_annotate()

    # plt.savefig(f'{"fig7a"}.svg')
    ax[0].set_position([0.06, 0.15, 0.15, 0.6])
    ax[1].set_position([0.26, 0.15, 0.15, 0.6])
    ax[2].set_position([0.49, 0.15, 0.23, 0.6])
    ax[3].set_position([0.77, 0.15, 0.23, 0.6])
    plt.savefig(f'./paper_figure/fig7a.svg', bbox_inches='tight')
    plt.show()

    print(f'YASA acc:{np.mean(yasa_acc[39:]):.4f} mf1:{np.mean(yasa_mf1[39:]):.4f}')
    print(f'TinySleepNet acc:{np.mean(tinysleep_acc[39:]):.4f} mf1:{np.mean(tinysleep_mf1[39:]):.4f}')
    print(f'UK-Sleep acc:{np.mean(uk_sleep_acc[39:]):.4f} mf1:{np.mean(uk_sleep_mf1[39:]):.4f}')


if __name__ == "__main__":
    print('hello world!')

    # fig_4_so_boxplot()
    # sleep_stage_all()
    # 分类准确率与性别，年龄，稳定状态，非稳定状态
    # def gamma_and_kde():

    sub_data = sleep_stage_all()

    y_p, y_t = [], []
    for no, self in enumerate(sub_data):
        if self.no < 40:
            # y_p_i = distinguish_n1_rem_new(self, win=20)
            y_p_i = self.label
            y_p.append(y_p_i)
            y_t.append(self.y)
    y_p, y_t = np.concatenate(y_p), np.concatenate(y_t)
    acc_20 = accuracy_score(y_p, y_t)
    k_20 = cohen_kappa_score(y_p, y_t)
    mf1_20 = f1_score(y_p, y_t, average='macro')
    # plot_confusion_matrix(y_p, y_t, name='edf_20_uk_sleep', title=f'UK-Sleep\nSleepEDF-20')
    for i in range(5):
        precision = np.sum(np.logical_and(y_p == i, y_t == i)) / np.sum(y_t == i)
        recall = np.sum(np.logical_and(y_p == i, y_t == i)) / np.sum(y_p == i)
        print(i, precision, recall, 2*precision*recall/(precision+recall))

    y_p, y_t = [], []
    for no, self in enumerate(sub_data):
        # y_p_i = distinguish_n1_rem_new(self, win=20)
        y_p_i = self.label
        y_p.append(y_p_i)
        y_t.append(self.y)


    y_p, y_t = np.concatenate(y_p), np.concatenate(y_t)
    acc_78 = accuracy_score(y_p, y_t)
    k_78 = cohen_kappa_score(y_p, y_t)
    mf1_78 = f1_score(y_p, y_t, average='macro')
    # plot_confusion_matrix(y_p, y_t, name='edf_78_uk_sleep', title=f'UK-Sleep\nSleepEDF-78')
    for i in range(5):
        precision = np.sum(np.logical_and(y_p == i, y_t == i)) / np.sum(y_t == i)
        recall = np.sum(np.logical_and(y_p == i, y_t == i)) / np.sum(y_p == i)
        print(i, precision, recall, 2*precision*recall/(precision+recall))

    """
    """

    # N1与REM前30分钟的睡眠期占比
    # y = []
    # y_pre_60_all = []
    # for self in sub_data:
    #     print(self.no)
    #     y.append(self.y)
    #     y_pre_60 = []
    #     for i, y in enumerate(self.y):
    #         idx = i
    #         if self.y[idx] == 4 or self.y[idx] == 1:
    #             while self.y[idx] == 4 or self.y[idx] == 1:
    #                 idx = idx - 1  # 向前，找到不属于当前类的索引
    #                 print(idx)
    #             if idx<60:
    #                 y_pre_60.append(self.y[])
    #             y_pre_60.append(self.y[idx-60:idx])

    # 测试matplotlib字体大小与word字体大小
    # 有一种说法，在mpl中，默认单位是点（point）。
    # 1 点等于 1/72 英寸，而word中的字号单位是磅。1磅=4/3像素，所以size=12对应的是小五号字（9磅）。
    # import matplotlib.pyplot as plt
    # import matplotlib as mpl
    # mpl.rcParams['svg.fonttype'] = 'none'  # 生成的字体不再以svg的形式保存，这样可以在inkspace中开始编辑文字
    # mpl.rcParams['font.family'] = 'Arial'
    # fig,ax = plt.subplots(figsize=(5, 5))
    # plt.text(0.5,0.5,'Test',fontdict={'size':18})
    # plt.savefig('test.svg', dpi=600)
    # plt.show()
