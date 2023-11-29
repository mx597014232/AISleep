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
from scipy.stats import norm, ttest_ind
from collections import Counter
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from lspopt import spectrogram_lspopt
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter
from skimage.segmentation import watershed  # 分水岭分割算法
from skimage.feature import peak_local_max
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import pairwise_distances
from scipy.signal import medfilt, find_peaks, savgol_filter
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

ft18 = 18
import matplotlib as mpl

mpl.rcParams.update({'font.size': ft18})
mpl.rcParams['font.family'] = 'Arial'
# mpl.use('TkAgg')
inches2cm = 1 / 2.54  # centimeters in inches

# 第一步，先运行1_20提取原始信号.py 在目录下生成包含153次睡眠实验预处理后的信号的npz文件
npz_root = 'D:\\code\\实验室代码\\data\\eeg_fpz_cz\\'

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
        if p < u0:  # 低功率部分的均值
            proba[i] = 0
        elif p > u1:  # 高功率部分的均值
            proba[i] = 1
        elif p < threshold:  # 高功率与低功率部分的分割阈值
            proba[i] = 0.5 * np.sum(np.logical_and(u0 <= x, x <= p)) / num_u0
        else:
            proba[i] = 0.5 + 0.5 * np.sum(np.logical_and(threshold <= x, x <= p)) / num_u1
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
        self.so_percent = None
        self.fea = None
        self.irasa = {}
        self.osc_mask = None
        self.fast_sp, self.sp_11_16, self.sp_freq = None, None, None
        self.gamma_z = None
        self.wake_close = None
        self.osc_std = None

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
        freqs, psd_aperiodic, psd_osc = yasa.irasa(self.x_signal, sf=int(self.sample_rate),
                                                   hset=[1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4],
                                                   band=(1, 25), win_sec=4, return_fit=False)
        psd = psd_aperiodic + psd_osc  # 功率谱密度 = 周期性功率谱密度 + 非周期性功率谱密度
        log_psd_osc = np.log(psd) - np.log(psd_aperiodic)  # 对数值相减，得到非周期性功率谱密度的log数量级

        log_psd_osc_filter = gaussian_filter(log_psd_osc, sigma=2)  # 高斯滤波去除噪声
        self.irasa['freqs'] = freqs
        self.irasa['psd'], self.irasa['aperiodic'], self.irasa['osc'] = psd, psd_aperiodic, psd_osc
        self.irasa['log_osc'], self.irasa['log_osc_filter'] = log_psd_osc, log_psd_osc_filter

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

        # sg滤波器平滑睡眠曲线，区分N1REM期
        self.label = distinguish_n1_rem_new(self)

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

    def find_wake(self, save=False):
        psd, y = np.log(self.psd), self.y
        scaler = StandardScaler()  # 标准化处理
        psd = scaler.fit_transform(psd)
        mean = psd[:, np.logical_and(25 < self.freqs, self.freqs < 50)].mean(axis=1)  # 16至50Hz的归一化均值， 25-50的脑电波更好

        # 阈值分割
        label, proba = ostu(mean)

        wake_percent = (proba - 0.5) * 2  # 16-50占比，也可以用作样本权重
        wake_percent[wake_percent < 0] = 0  # 只考虑16-50部分的权重
        kde = KernelDensity(kernel="gaussian", bandwidth="scott").fit(self.psd_umap, sample_weight=wake_percent)
        psd_umap_p = np.exp(kde.score_samples(self.psd_umap))  # 概率密度
        levels = np.linspace(psd_umap_p.min(), psd_umap_p.max(), 10)  # 概率密度10等分
        self.wake = psd_umap_p > levels[1]
        self.gamma_z = mean

        if self.show:
            bins = np.linspace(mean.min(), mean.max(), 100)  # 直方图bins范围
            fig, (ax_line, ax_bins, ax_scatter) = plt.subplots(ncols=3, figsize=(15, 5))
            ax_line.plot(mean / mean.max())
            ax_line.plot(self.y / self.y.max() + 1)
            _ = ax_bins.hist(mean[label == 0], bins=bins, alpha=0.6, label='others')
            _ = ax_bins.hist(mean[label == 1], bins=bins, alpha=0.6, label='fast spindle')
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
        self.sp_freq = 14  # 默认峰值频率为14Hz
        if len(peak_idxs):
            sp_idx = np.argmin(np.abs(freqs[peak_idxs] - 14))  # 快纺锤波12-16Hz，寻找与中心频率14Hz最接近的纺锤波段
            if 11 <= freqs[peak_idxs[sp_idx]] <= 16:
                self.sp_freq = freqs[peak_idxs[sp_idx]]
                sp_l_idx, sp_r_idx = np.where(freqs >= self.sp_freq - 1)[0][0], \
                                     np.where(freqs <= self.sp_freq + 1)[0][-1]

        sp_mean = np.mean(log_psd_osc_filter[:, sp_l_idx:sp_r_idx], axis=1)
        self.fast_sp, self.sp_11_16 = sp_mean.copy(), sp_mean_11_16

        # 下一步的思路：
        # 理论上，0是区分NREM与REM的阈值
        # 实际上，该阈值应该随着分布稍作调整
        # 用高斯分布拟合 <0 和 >0 的两个分布
        # 然后用这两个分布的累积概率密度分配权重，然后KDE
        mean_0, mean_1 = np.mean(sp_mean[sp_mean <= 0]), np.mean(sp_mean[sp_mean >= 0])
        std_0, std_1 = np.std(sp_mean[sp_mean <= 0]), np.std(sp_mean[sp_mean >= 0])
        x = np.linspace(np.min(sp_mean), np.max(sp_mean), 100)
        pdf_0, pdf_1 = norm.pdf(x, loc=mean_0, scale=std_0), norm.pdf(x, loc=mean_1, scale=std_1)
        pdf_0[x < mean_0], pdf_1[x > mean_1] = pdf_0.max(), pdf_1.max()
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

        if self.show:
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

    def find_n3_yasa_check_mean_dis(self):
        # n3期判定：考虑点到3个类（w,nrem,unknow）的平均距离
        print(self.no, 'n3')
        sw = yasa.sw_detect(self.x_signal.flatten() * (10 ** 4), sf=100, freq_sw=(0.5, 2.0),
                            dur_neg=(0.1, 2.0), dur_pos=(0.1, 2.0), amp_neg=(10, 500),
                            amp_pos=(10, 500), amp_ptp=(75, 1000), coupling=False,
                            remove_outliers=False, verbose=False)
        sw_mask = sw.get_mask().reshape((-1, 3000))  # 慢波对应的区间,转换成睡眠帧相应的mask
        self.so_percent = sw_mask.sum(axis=1) / 3000  # 慢波时间占比
        self.so_percent = self.so_percent - 0.10  # 减去0.15，对应的正确率是79.83%，不减去0.15，正确率是：76.81%；-0.10，对应79.87%
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
        self.osc_std = osc_std

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


def distinguish_n1_rem_new(self, win=20):
    # 状态平滑过渡，由于把N1当做REM，N1处与平滑后的曲线差异较大。
    # new_label_sg = savgol_filter(self.label, 40, 2)  # 用S-G滤波器对状态序列平滑处理
    new_label_sg = np.convolve(self.label, np.ones(win) / win, mode='same')
    # 对于每个REM期，如果与1更近，则为N1期，如果与4更近，则为REM
    new_label = []
    for i, l in enumerate(self.label):
        if l == 4:
            if new_label_sg[i] < 2.5:
                new_label.append(1)
            else:
                new_label.append(4)
        else:
            new_label.append(l)
    new_label = np.array(new_label)
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


# ppt 补充画图


# 人工标注，功率谱密度，信号绘制
def fig_1_b_c_plot_y_psd_sig():
    self = DatasetCreate(no=0, show=False)
    import matplotlib.pyplot as plt

    cmap = ListedColormap([color[i] for i in range(5)])
    trimperc = 2.5  # 百分比范围2.5%——97.5%
    v_min, v_max = np.percentile(np.log(self.psd), [0 + trimperc, 100 - trimperc])
    norm_ = Normalize(vmin=v_min, vmax=v_max)
    time = np.linspace(0, self.psd.shape[0] / 120, self.psd.shape[0])

    # 创建大图对象
    fig = plt.figure(figsize=(16, 5))

    # 子图2：位于第三行，跨越3列
    ax2 = plt.subplot2grid(shape=(4, 4), loc=(2, 2))
    ax2.plot(self.x_signal.flatten() * 10000, c='black', alpha=0.9)
    ax2.set_xlim(0, time.shape[0] * 3000)
    ax2.set_xticks([]), ax2.set_yticks([-150, 0, 150], [-150, 0, 150])
    ax2.spines['right'].set_visible(False), ax2.spines['top'].set_visible(False), ax2.spines['bottom'].set_visible(
        False)
    ax2.set_title('EEG (Fpz-Cz)', fontdict={'size': ft18})
    ax2.set_ylabel('Amplitude (μV)', fontdict={'size': ft18})
    ax2.yaxis.set_label_coords(-0.09, 0.5)

    # 子图3：位于第一行第三列，跨越1行1列
    ax3 = plt.subplot2grid(shape=(4, 4), loc=(1, 1))
    ax3.pcolormesh(time, self.freqs, np.log(self.psd).T, norm=norm_, antialiased=True, shading="auto", cmap='RdBu_r')
    ax3.set_xticks([])
    ax3.set_ylabel('Frequency (Hz)', fontdict={'size': ft18})
    ax3.set_title('PSD', fontdict={'size': ft18})
    ax3.yaxis.set_label_coords(-0.09, 0.5)
    ax3.set_xlabel('Time (h)', fontdict={'size': ft18})
    ax3.set_xticks([i for i in range(8)], [i for i in range(8)])

    ax4 = plt.subplot2grid(shape=(4, 4), loc=(3, 3))
    ax4.scatter(self.psd_umap[:, 0], self.psd_umap[:, 1], c='gray', s=4, label='Sleep frame\n(30s)')
    ax4.spines['top'].set_visible(False), ax4.spines['right'].set_visible(False)
    ax4.set_xlabel('UMAP 1', fontdict={'size': ft18}), ax4.set_ylabel('UMAP 2', fontdict={'size': ft18})
    ax4.set_xticks([]), ax4.set_yticks([])
    ax4.tick_params(labelsize=ft18)  # 坐标轴字体大小
    legend = plt.legend(prop={'size': ft18}, handlelength=1, markerscale=4)
    legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0
    # plt.subplots_adjust(left=0.1, bottom=0.12, right=0.95, top=0.95)
    # plt.subplots_adjust(left=0.12, bottom=0.15, right=0.95, top=0.95)
    ax4.xaxis.set_label_coords(0.5, -0.08), ax4.yaxis.set_label_coords(-0.08, 0.5)
    ax4.set_title('UMAP with PSD (PSD$_{\mathrm{reduced}}$)')

    # 调整子图的大小和位置(x,y,width,heigh)
    ax2.set_position([0.09, 0.55, 0.55, 0.35])
    ax3.set_position([0.09, 0.15, 0.55, 0.35])
    ax4.set_position([0.71, 0.15, 0.25, 0.75])

    plt.savefig(f'./paper_figure/fig_1_b_c_plot_y_psd_sig.png', dpi=800)
    # 显示图形
    plt.show()


# gamma波的直方图分布
def fig_2_a_plot_gamma_hist(self):
    psd, y = np.log(self.psd), self.y
    scaler = StandardScaler()  # 标准化处理
    psd = scaler.fit_transform(psd)
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

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(mean[mean <= threshold], bins=bins[:th_idx], label='Low gamma', alpha=0.5, color='gray')
    ax.hist(mean[mean >= threshold], bins=bins[th_idx:], label='High gamma', alpha=0.5, color=color[0])
    ax.axvline(bins[th_idx], c='C3', label='Threshold', linestyle='dashed')
    ax.set_ylabel('Bin counts', fontdict={'size': ft18})
    ax.set_xlabel('Gamma (z-score)', fontdict={'size': ft18})
    ax.tick_params(labelsize=ft18)  # 坐标轴字体大小
    ax2 = ax.twinx()  # 双y轴显示
    ax2.plot(np.arange(start + interval, end - interval, interval), loss_list, label='Loss', c='black')
    ax2.set_ylabel('Otsu loss value', fontdict={'size': ft18})
    ax2.set_ylim(0, loss_list.max() * 2)
    ax2.scatter([bins[th_idx]], [loss_list.max()], marker='x', c='red', s=100, label='Maximum')
    legend = fig.legend(framealpha=0, bbox_to_anchor=(1, 1.05), loc=1, bbox_transform=ax.transAxes,
                        prop={'size': ft18})
    legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0
    # fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes,
    #            prop={'size': ft18})
    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.tick_params(labelsize=ft18)  # 坐标轴字体大小
    plt.subplots_adjust(left=0.12, bottom=0.15, right=0.88, top=1)
    plt.savefig('./paper_figure/Wake_Otsu.svg')
    plt.show()


# 绘制清醒期的umap分布
def fig_2_b_plot_wake(self):
    psd, y = np.log(self.psd), self.y
    scaler = StandardScaler()  # 标准化处理
    psd = scaler.fit_transform(psd)
    mean = psd[:, np.logical_and(25 < self.freqs, self.freqs < 50)].mean(axis=1)  # 16至50Hz的归一化均值， 25-50的脑电波更好

    # 阈值分割
    label, proba = ostu(mean)

    wake_percent = (proba - 0.5) * 2  # 16-50占比，也可以用作样本权重
    wake_percent[wake_percent < 0] = 0  # 只考虑16-50部分的权重
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
    cbar = plt.colorbar(cb)
    cbar.ax.set_yticklabels(['{:.2f}'.format(tick) for tick in np.arange(0, 0.3, 0.03)])
    cbar.ax.set_yticklabels(['0', '', '', '', '', '', '', '', '', '0.3'])
    # cbar.ax.set_title('Kernel Density Estimation', fontsize=12)
    cbar.ax.title.set_rotation(90)  # 将标题垂直显示
    cbar.set_label('Density', rotation=270, fontdict={'size': ft18}, labelpad=10)
    # 设置颜色条刻度标签的字体大小
    for label in cbar.ax.get_yticklabels():
        label.set_fontsize(ft18)
    legend = plt.legend(prop={'size': ft18}, handlelength=1, markerscale=4)
    legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0
    # plt.subplots_adjust(left=0.1, bottom=0.12, right=0.95, top=0.95)
    plt.subplots_adjust(left=0.12, bottom=0.15, right=0.95, top=0.95)
    ax.xaxis.set_label_coords(0.5, -0.08), ax.yaxis.set_label_coords(-0.08, 0.5)
    plt.savefig('./paper_figure/Wake_contourf.svg')
    plt.show()


# Fig.3irasa震荡功率分离
def fig_3_plot_irasa_new(self):
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

    # IRASA信号分解图
    # 设置全局字体大小
    import matplotlib
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mpatches
    matplotlib.rcParams.update({'font.size': ft18})
    fig = plt.figure(layout="constrained", figsize=(11.5, 14))
    gs = GridSpec(145, 120, figure=fig)
    # fig.supylabel('Frequency (Hz)', x=0.02, y=0.5, fontsize=ft18)
    # fig.suptitle('Power (dB)', x=0.97, y=0.5, fontsize=ft18, rotation=90, va='center')
    ax_sp2, ax_sp3, ax_sp4, ax_sp_mean = fig.add_subplot(gs[0:30, 0:120]), \
                                         fig.add_subplot(gs[35:65, 0:120]), \
                                         fig.add_subplot(gs[70:100, 0:120]), \
                                         fig.add_subplot(gs[105:145, 0:120])

    trimperc = 2.5  # 百分比范围2.5%——97.5%
    v_min, v_max = np.percentile(np.log(psd), [0 + trimperc, 100 - trimperc])
    norm_ = Normalize(vmin=v_min, vmax=v_max)

    ax_sp2.set_title('PSD$_{\mathrm{fra}}$', fontsize=ft18)
    ax_sp2.set_ylabel('Frequency (Hz)', fontdict={'size': ft18})
    ax_sp2.set_xlabel('Time (h)', fontdict={'size': ft18})
    im2 = ax_sp2.pcolormesh(time, freqs[0:psd.shape[1]], np.log(psd_aperiodic).T,
                            norm=norm_, antialiased=True, shading="auto", cmap='RdBu_r')

    fig.colorbar(im2, ax=ax_sp2, label='Power (dB)', fraction=0.05, aspect=30, pad=0.01)

    trimperc = 0.01  # 百分比范围2.5%——97.5%
    v_min, v_max = np.percentile(log_psd_osc, [0 + trimperc, 100 - trimperc])
    norm_ = Normalize(vmin=v_min, vmax=v_max)

    ax_sp3.set_title('PSD$_{\mathrm{osc}}$', fontsize=ft18)
    ax_sp3.set_ylabel('Frequency (Hz)', fontdict={'size': ft18})
    im3 = ax_sp3.pcolormesh(time, freqs[0:psd.shape[1]], log_psd_osc.T, norm=norm_,
                            antialiased=True, shading="auto", cmap='RdBu_r')
    fig.colorbar(im3, ax=ax_sp3, label='Power (dB)', fraction=0.05, aspect=30, pad=0.01)

    ax_sp4.set_title('PSD$_{\mathrm{osc}}$ (Gaussian filtered)', fontsize=ft18)
    ax_sp4.set_ylabel('Frequency (Hz)', fontdict={'size': ft18})
    ax_sp4.set_xlabel('Time (h)', fontdict={'size': ft18})
    im3 = ax_sp4.pcolormesh(time, freqs[0:psd.shape[1]], log_psd_osc_filter_.T, norm=norm_,
                            antialiased=True, shading="auto", cmap='RdBu_r')
    ax_sp4.plot([0, psd.shape[0] / 120], [20, 20], c='red', linestyle='--')
    ax_sp4.plot([0, psd.shape[0] / 120], [5, 5], c='red', linestyle='--')
    # fig.colorbar(im3, ax=ax_sp4)
    fig.colorbar(im3, ax=ax_sp4, label='Power (dB)', fraction=0.05, aspect=30, pad=0.01)

    # cmap = ListedColormap([color[i] for i in range(5)])
    # ax_y.imshow(self.y.reshape((-1, 1)).T, cmap=cmap, aspect='auto')  # 绘制颜色覆盖图
    # ax_y.set_xticks([]), ax_y.set_yticks([])  # 设置坐标轴刻度线的位置
    #
    # legends = [mpatches.Patch(color=color[i], label=stage[i]) for i in range(5)]
    # legend = ax_y.legend(handles=legends, loc='center', handlelength=1, markerscale=1,
    #                      ncol=5, bbox_to_anchor=(0.5, 1.5), framealpha=0,
    #                      bbox_transform=ax_y.transAxes, prop={'size': ft18})
    # legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0

    # ax_sp_mean.set_title('Fast spindle power', fontsize=ft18)
    ax_sp_mean.set_ylabel('Power (dB)', fontdict={'size': ft18})
    ax_sp_mean.set_xlabel('Time (h)', fontdict={'size': ft18})
    ax_sp_mean.plot(time, sp_mean, c='black', label='Fast spindle power')
    # ax_sp_mean.set_xlabel("Time (h)", fontdict={'size': ft18})
    ax_sp_mean.set_xlim(0, psd.shape[0] / 120), ax_sp_mean.set_ylim(sp_mean.min() - 0.01, sp_mean.max() + 0.3)
    ax_sp_mean.plot([0, psd.shape[0] / 120], [0, 0], c='red', linestyle='--')
    ax_sp_mean.fill_between(np.linspace(0, psd.shape[0] / 120, psd.shape[0]), sp_mean.min(), sp_mean.max(),
                            where=self.n2n3,
                            alpha=0.2, linewidth=0, color='green', label='UK-Sleep (N2N3)')
    ax_sp_mean.spines['top'].set_visible(False), ax_sp_mean.spines['right'].set_visible(False)
    legend = ax_sp_mean.legend(framealpha=0, bbox_to_anchor=(0.5, 0.95), loc='center', ncol=2,
                               bbox_transform=ax_sp_mean.transAxes, prop={'size': ft18}, handlelength=1, markerscale=4)
    legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0
    plt.savefig('./paper_figure/NREM_IRASA.png', dpi=800)
    plt.show()

    # 高斯滤波直方图
    # fig = plt.figure(layout="constrained", figsize=(5, 5.5))
    # gs = GridSpec(2, 5, figure=fig)
    # ax_hist = fig.add_subplot(gs[0:2, 0:5])

    fig, ax_hist = plt.subplots(figsize=(4, 3.5))
    ax_hist.hist(log_psd_osc.flatten(), bins=100, label='PSD$_{osc}$', alpha=0.5, density=True)
    ax_hist.hist(log_psd_osc_filter_.flatten(), bins=100, color='red',
                 label='PSD$_{osc}$ \n(Gaussian filtered)', alpha=0.5, density=True)
    legend = ax_hist.legend(framealpha=0, bbox_to_anchor=(0.5, 0.85), loc='center',
                            bbox_transform=ax_hist.transAxes, prop={'size': ft18}, handlelength=1)
    legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0
    ax_hist.set_ylim(0, 4)
    ax_hist.set_xlim(-2, 2)
    ax_hist.set_ylabel('Density', fontdict={'size': ft18})
    ax_hist.set_xlabel('Power (dB)', fontdict={'size': ft18})
    # ax_hist.set_title('Histogram', fontsize=ft18)
    ax_hist.spines['top'].set_visible(False)
    ax_hist.spines['right'].set_visible(False)
    plt.subplots_adjust(left=0.19, bottom=0.17, right=0.95, top=0.95)

    plt.savefig('./paper_figure/NREM_Gaussian filtered.svg', dpi=800)
    plt.show()

    import matplotlib
    matplotlib.rcParams.update({'font.size': 18})
    # fig = plt.figure(layout="constrained", figsize=(5, 6))
    # gs = GridSpec(1, 6, figure=fig)
    # ax_hist = fig.add_subplot(gs[0, 0:6])

    fig, ax_hist = plt.subplots(figsize=(4, 3.5))
    ax_hist.fill_between([sp_freq - 1, sp_freq + 1], 0, 0.65, alpha=0.2, linewidth=0, color=color[2],
                         label='Fast spindle range')
    ax_hist.hist(max_freq, bins=60, label='', alpha=0.8, density=True)
    ax_hist.plot(freqs, fast_sp_p0, label='KDE', color='black')
    ax_hist.scatter(sp_freq, fast_sp_p0[freqs == sp_freq], c='red', marker='x', s=100, label='Local maximum')
    ax_hist.plot([20, 20], [0, 0.65], c='red', linestyle='--')
    ax_hist.plot([5, 5], [0, 0.65], c='red', linestyle='--')
    legend = ax_hist.legend(framealpha=0, bbox_to_anchor=(0.5, 0.85), loc='center',
                            bbox_transform=ax_hist.transAxes, prop={'size': ft18}, handlelength=1)
    legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0
    # ax_hist.set_xlim(-2, 2)
    ax_hist.set_ylabel('Density', fontdict={'size': ft18})
    ax_hist.yaxis.set_label_coords(-0.1, 0.5)
    ax_hist.set_yticks([0, 1], [0, 1.0])
    ax_hist.set_xlabel('Frequency (Hz)', fontdict={'size': ft18})
    ax_hist.xaxis.set_label_coords(0.45, -0.11)
    # ax_hist.set_title(' ', fontsize=ft18)
    ax_hist.spines['top'].set_visible(False)
    ax_hist.spines['right'].set_visible(False)
    ax_hist.set_xlim(0, 25), ax_hist.set_ylim(0, 1)
    plt.subplots_adjust(left=0.19, bottom=0.17, right=0.95, top=0.95)
    plt.savefig('./paper_figure/NREM_max_power_frequency.svg', dpi=800, bbox_inches='tight')
    plt.show()

    # Histogram of fast spindle power 快纺锤波直方图
    # fig = plt.figure(layout="constrained", figsize=(4, 3.5))
    # gs = GridSpec(6, 2, figure=fig)
    fig, ax_hist = plt.subplots(figsize=(4, 3.5))
    bins = np.linspace(sp_mean.min() - 0.01, sp_mean.max() + 0.3, 100)
    ax_hist.hist(sp_mean[sp_mean > 0],
                 bins=bins, label='N2N3', alpha=0.4, density=True, color='green')
    ax_hist.hist(sp_mean[sp_mean <= 0],
                 bins=bins, label='Others', alpha=0.4, density=True, color='gray')
    ax_hist.plot([0, 0], [0, 4.5], c='red', linestyle='--')
    # ax_hist.set_xlim(0, 5)
    pdf0, pdf1 = norm.pdf(x_sp, loc=mean_0, scale=std_0), norm.pdf(x_sp, loc=mean_1, scale=std_1)
    ax_hist.plot(x_sp, pdf1, c='green', alpha=1), ax_hist.plot(x_sp, pdf0, c='gray', alpha=1)
    legend = ax_hist.legend(framealpha=1)
    legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0
    ax_hist.set_ylabel('Density', fontdict={'size': ft18})
    ax_hist.set_xlabel('Fast spindle power (dB)', fontdict={'size': ft18})
    # ax_hist.set_title('Histogram', fontsize=ft18)
    ax_hist.spines['top'].set_visible(False), ax_hist.spines['right'].set_visible(False)
    plt.subplots_adjust(left=0.19, bottom=0.17, right=0.95, top=0.95)
    plt.savefig('./paper_figure/NREM_Histogram_fast_spindle.svg', bbox_inches='tight')
    plt.show()

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
    legend = plt.legend(prop={'size': ft18}, handlelength=1, markerscale=4)
    legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0
    # plt.tight_layout()
    # plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.94)
    plt.subplots_adjust(left=0.12, bottom=0.15, right=0.95, top=0.95)
    ax.xaxis.set_label_coords(0.5, -0.08), ax.yaxis.set_label_coords(-0.08, 0.5)
    plt.savefig('./paper_figure/NREM_contourf.svg')
    plt.show()


# N3期相关绘图
def fig_4_plot_n3():
    # 数据读取
    no = 0
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
                        dur_neg=(0.1, 2.0), dur_pos=(0.1, 2.0), amp_neg=(10, 500),
                        amp_pos=(10, 500), amp_ptp=(75, 1000), coupling=False,
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

    fig, ax = plt.subplots(figsize=(4.8, 4))
    cb = ax.contourf(x, y, p, levels=levels, cmap=my_cmap)  # cmap='Blues'  # 将 cmap 参数设置为新的自定义线性颜色映射：
    ax.scatter(self.psd_umap[self.n3, 0], self.psd_umap[self.n3, 1], c='black', s=4, label='N3')
    ax.scatter(self.psd_umap[~self.n3, 0], self.psd_umap[~self.n3, 1], c='gray', s=4, label='Others')
    ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
    ax.set_xlabel('UMAP 1', fontdict={'size': ft18}), ax.set_ylabel('UMAP 2', fontdict={'size': ft18})
    ax.set_xticks([]), ax.set_yticks([])
    cb.set_clim(0, 0.4)
    cbar = plt.colorbar(cb)
    cbar.ax.set_yticklabels(['{:.2f}'.format(tick) for tick in np.arange(0, 0.4, 0.04)])
    cbar.ax.set_yticklabels(['0', '', '', '', '', '', '', '', '', '0.4'])
    # cbar.ax.set_title('Kernel Density Estimation', fontsize=12)
    cbar.ax.title.set_rotation(90)  # 将标题垂直显示
    cbar.set_label('Density', rotation=270, fontdict={'size': ft18}, labelpad=10)
    for label in cbar.ax.get_yticklabels():
        label.set_fontsize(ft18)
    legend = plt.legend(prop={'size': ft18}, handlelength=1, markerscale=4)
    legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0
    plt.subplots_adjust(left=0.12, bottom=0.15, right=0.95, top=0.95)
    ax.xaxis.set_label_coords(0.5, -0.08), ax.yaxis.set_label_coords(-0.08, 0.5)
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

    fig = plt.figure(layout="constrained", figsize=(11.5, 8))
    gs = GridSpec(80, 12, figure=fig)
    ax_n2 = fig.add_subplot(gs[0:20, 0:12])
    ax_n2.plot(self.x_signal[n2, :].flatten() * 10000, c='black')
    ax_n2.fill_between(np.linspace(0, 3000, 3000), - 100, + 100, where=sw_mask[n2],
                       alpha=0.2, linewidth=0, color='red', label='SO')
    ax_n2.spines['top'].set_visible(False), ax_n2.spines['right'].set_visible(False)
    ax_n2.spines['bottom'].set_visible(False)
    ax_n2.set_xlim(0, 3000)
    # ax_n2.set_ylabel('Ampllitude (μV)', fontdict={'size': ft18})
    fig.text(0.01, 0.80, 'Ampllitude (μV)', va='center', rotation='vertical', fontdict={'size': ft18})
    # ax_n2.set_xlabel('Time (s)', fontdict={'size': ft18})
    ax_n2.set_xticks([])
    ax_n2.set_yticks([-100, 0, 100], [-100, 0, 100], fontsize=ft18)
    ax_n2.set_title(f"Example 1 (SO percentage :{so_percent[n2] * 100}%)", fontdict={'size': ft18})
    legend = ax_n2.legend(framealpha=1, bbox_to_anchor=(0.9, 1.05), loc='center', ncol=3,
                          bbox_transform=ax_n2.transAxes,
                          prop={'size': ft18}, handlelength=1, markerscale=4)
    legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0

    ax_n3 = fig.add_subplot(gs[20:40, 0:12])
    ax_n3.plot(self.x_signal[n3, :].flatten() * 10000, c='black')
    ax_n3.fill_between(np.linspace(0, 3000, 3000), - 100, + 100, where=sw_mask[n3],
                       alpha=0.2, linewidth=0, color='red', label='SO')
    ax_n3.spines['top'].set_visible(False), ax_n3.spines['right'].set_visible(False)
    ax_n3.set_xlim(0, 3000)
    # ax_n3.set_ylabel('Ampllitude (μV)', fontdict={'size': ft18})
    ax_n3.set_xlabel('Time (s)', fontdict={'size': ft18})
    ax_n3.set_xticks([0, 1000, 2000, 3000], [0, 10, 20, 30], fontsize=ft18)
    ax_n3.set_yticks([-100, 0, 100], [-100, 0, 100], fontsize=ft18)
    ax_n3.set_title(f"Example 2 (SO percentage :{so_percent[n3] * 100}%)", fontdict={'size': ft18})

    # ax_y = fig.add_subplot(gs[30:33, 0:12])
    # cmap = ListedColormap([color[i] for i in range(5)])
    # ax_y.imshow(self.y.reshape((-1, 1)).T, cmap=cmap, aspect='auto')  # 绘制颜色覆盖图
    # ax_y.set_xticks([]), ax_y.set_yticks([])  # 设置坐标轴刻度线的位置
    #
    # legends = [mpatches.Patch(color=color[i], label=stage[i]) for i in range(5)]
    # legend = ax_y.legend(handles=legends, loc='center', handlelength=1, markerscale=1,
    #                      ncol=5, bbox_to_anchor=(0.5, 1.5), framealpha=0,
    #                      bbox_transform=ax_y.transAxes, prop={'size': ft18})
    # legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0

    ax_n2n3 = fig.add_subplot(gs[40:45, 0:12])
    cmap = ListedColormap(['gray', color[2]])
    ax_n2n3.imshow(nrem_mask.reshape((-1, 1)).T, cmap=cmap, aspect='auto')  # 绘制颜色覆盖图
    ax_n2n3.set_xticks([]), ax_n2n3.set_yticks([])  # 设置坐标轴刻度线的位置

    legends = [mpatches.Patch(color='gray', label='Others'), mpatches.Patch(color=color[2], label='N2N3')]
    legend = ax_n2n3.legend(handles=legends, loc='center', handlelength=1, markerscale=1,
                            ncol=5, bbox_to_anchor=(0.5, 1.5), framealpha=0,
                            bbox_transform=ax_n2n3.transAxes, prop={'size': ft18})
    legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0

    ax_so = fig.add_subplot(gs[45:, 0:12])
    ax_so.plot(so_percent, c='black')
    ax_so.fill_between(np.linspace(0, len(self.y), len(self.y)), 0, 1, where=self.n3, interpolate=True,
                       alpha=0.4, linewidth=0, color=color[3], label='N3 (UK-Sleep)')
    ax_so.fill_between(np.linspace(0, len(self.y), len(self.y)), 0, 1,
                       where=nrem_mask.astype('int') - self.n3.astype('int'),
                       interpolate=True, alpha=0.2, linewidth=0, color=color[2], label='N2 (UK-Sleep)')
    ax_so.plot([0, len(self.y)], [0.1, 0.1], c='red', linestyle='--', label='Cutoff (10%)')
    ax_so.spines['top'].set_visible(False), ax_so.spines['right'].set_visible(False)
    ax_so.set_xlim(0, len(self.y)), ax_so.set_ylim(0, 0.8)
    # ax_so.set_ylabel('Percentage', fontdict={'size': ft18})
    ax_so.set_xticks([0, 120, 240, 360, 480, 600, 720], [0, 1, 2, 3, 4, 5, 6], fontsize=ft18)
    ax_so.set_yticks([0, 0.2, 0.4, 0.6, 0.8], ['0%', '20%', '40%', '60%', '80%'], fontsize=ft18)
    ax_so.set_xlabel('Time (h)', fontdict={'size': ft18})
    ax_so.set_ylabel(f"SO percentage", fontdict={'size': ft18})
    legend = ax_so.legend(framealpha=0, loc='center', bbox_to_anchor=(0.5, 1.1), bbox_transform=ax_so.transAxes,
                          prop={'size': ft18}, columnspacing=0.5, ncol=3)
    legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0
    plt.savefig('./paper_figure/so_percentage_n2_n3_sleep_stage.svg', bbox_inches='tight')
    plt.show()


def fig_4_so_boxplot():
    so = []
    for no in range(166):
        if no in [27, 73, 78, 79, 104, 136, 137, 138, 139, 156, 157, 158, 159]:
            continue
        print(self.no)
        self = DatasetCreate(no=no, show=False)
        self.clc_irasa()  # 1/f分形信号分解
        self.find_wake()  # W期判定
        self.find_nrem_irasa_max_freq()  # nrem期判定
        self.find_n3_yasa_check_mean_dis()  # n3期判定
        so.append([self.so_percent[self.n3].mean(), self.so_percent[~self.n2n3].mean()])
    so = np.array(so)
    so = np.nan_to_num(so)
    fig, ax = plt.subplots(figsize=(4, 4))
    xticks, xlabel = [1, 2], ['N3', 'Others']
    xi, yi = [1, 2], np.array([so[:, i].mean() for i in range(2)])
    yi_test, h = so.max() + 0.05, 0.02
    bx = ax.boxplot([so[:, 0], so[:, 1]], notch=True, widths=0.5,
                    boxprops={'linewidth': 2, 'facecolor': color[3]}, patch_artist=True)
    # N3期与N3期以外的睡眠期
    ax.plot([xi[0], xi[0], xi[1], xi[1]],
            [yi_test, yi_test + h, yi_test + h, yi_test], lw=1, c="k")
    stat, p_value = ttest_ind(so[:, 0], so[:, 1], equal_var=False)
    ax.text((xi[0] + xi[1]) * 0.5, yi_test + h, convert_pvalue_to_asterisks(p_value),
            ha='center', va='bottom', color="k", fontsize=ft18)
    ax.spines['right'].set_visible(False), ax.spines['top'].set_visible(False)
    ax.set_xticks(xticks, xlabel)
    # ax.set_yticks(np.arange(0, 1.01, 0.2), [f'{i}%' for i in range(0, 101, 20)])
    ax.set_ylabel('SO percentage (%)')
    ax.set_ylim(-0.05, 0.7)
    plt.subplots_adjust(left=0.22, bottom=0.15, right=0.95, top=0.95)
    plt.savefig(f'./paper_figure/N3_so_others_power.svg')
    plt.show()


# 绘制α波较强的区域
def fig_5_plot_alpha():
    # 数据读取
    no = 0
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
    fig, ax = plt.subplots(figsize=(4.8, 4))
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
    cbar = plt.colorbar(cb)
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

    legend = plt.legend(handler_map={PathCollection: HandlerPathCollection(update_func=update_scatter)}, loc='center',
                        framealpha=0,
                        bbox_to_anchor=(0.25, 0.8), prop={'size': ft18}, handlelength=1, markerscale=4,
                        bbox_transform=ax.transAxes)
    legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0
    plt.subplots_adjust(left=0.12, bottom=0.15, right=0.95, top=0.95)
    ax.xaxis.set_label_coords(0.5, -0.08), ax.yaxis.set_label_coords(-0.08, 0.5)
    plt.savefig('./paper_figure/alpha_ax.contourf.svg', bbox_inches='tight')
    plt.show()

    # 方差
    time = np.linspace(0, psd.shape[0] / 120, psd.shape[0])

    fig = plt.figure(layout="constrained", figsize=(15, 4))
    gs = GridSpec(40, 15, figure=fig)

    ax_osc = fig.add_subplot(gs[0:3, 0:15])
    cmap = ListedColormap(['gray', color[5]])
    ax_osc.imshow(self.osc_mask.reshape((-1, 1)).T, cmap=cmap, aspect='auto')  # 绘制颜色覆盖图
    ax_osc.set_xticks([]), ax_osc.set_yticks([])  # 设置坐标轴刻度线的位置
    ax_osc.plot([0, 0], [0, 0], linestyle='--', label='Mean std')

    legends = [mpatches.Patch(color=color[5], label='Osc'),
               mpatches.Patch(color=color[2], label='N2N3'),
               mpatches.Patch(color=color[0], label='Wake$_{\mathrm{close}}$', alpha=0.4),
               mlines.Line2D([], [], color='red', linestyle='dashed', label='Mean std')]
    legend = ax_osc.legend(handles=legends, loc='center', handlelength=1, markerscale=1,
                           ncol=5, bbox_to_anchor=(0.5, 1.6), framealpha=0,
                           bbox_transform=ax_osc.transAxes, prop={'size': ft18})
    legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0

    ax_n2n3 = fig.add_subplot(gs[3:6, 0:15])
    cmap = ListedColormap(['gray', color[2]])
    ax_n2n3.imshow(self.n2n3.reshape((-1, 1)).T, cmap=cmap, aspect='auto')  # 绘制颜色覆盖图
    ax_n2n3.set_xticks([]), ax_n2n3.set_yticks([])  # 设置坐标轴刻度线的位置
    # legends = [mpatches.Patch(color='gray', label='Others'), mpatches.Patch(color=color[2], label=stage[2])]
    # legend = ax_n2n3.legend(handles=legends, loc='center', handlelength=1, markerscale=1,
    #                      ncol=5, bbox_to_anchor=(0.5, 1.5), framealpha=0,
    #                      bbox_transform=ax_n2n3.transAxes, prop={'size': ft18})
    # legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0

    ax_std = fig.add_subplot(gs[6:40, 0:15])
    ax_std.set_title(''), ax_std.set_ylabel('SD of PSD$_{\mathrm{osc}}$ (dB)', fontdict={'size': ft18})
    ax_std.plot(time, osc_std, c='black')
    ax_std.plot([0, time[-1]], [th, th], c='red', linestyle='--', label='Mean std')
    # ax_std.fill_between(np.linspace(0, psd.shape[0] / 120, psd.shape[0]), th, osc_std.max(),
    #                     where=self.osc_mask, alpha=0.4, linewidth=0, color=color[5], label='Osc')
    ax_std.fill_between(np.linspace(0, len(self.y) / 120, len(self.y)), osc_std.min(), osc_std.max(),
                        where=np.logical_and(self.osc_mask, ~self.n2n3), interpolate=True,
                        alpha=0.4, linewidth=0, color=color[0], label='Wake$_{\mathrm{close}}$')
    # ax_std.fill_between(np.linspace(0, len(self.y) / 120, len(self.y)), osc_std.min(), th,
    #                     where=np.logical_and(self.osc_mask, self.n2n3),
    #                     alpha=0.2, linewidth=0, color=color[2], label='Osc ∩ N2&N3')
    ax_std.set_xlabel("Time (h)")
    ax_std.set_xlim(0, psd.shape[0] / 120), ax_std.set_ylim(osc_std.min() - 0.01, osc_std.max() + 0.02)
    ax_std.plot([0, psd.shape[0] / 120], [0, 0], c='red', linestyle='--')
    ax_std.spines['top'].set_visible(False), ax_std.spines['right'].set_visible(False)
    # legend = ax_std.legend(framealpha=0, bbox_to_anchor=(0.5, 0.9), ncol=4, bbox_transform=ax_std.transAxes,
    #                        loc='center')
    # legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0
    plt.savefig('./paper_figure/Oscillatory_std.svg', bbox_inches='tight')
    plt.show()

    # fig = plt.figure(layout="constrained", figsize=(10, 5))
    # gs = GridSpec(30, 1, figure=fig)
    # ax_y = fig.add_subplot(gs[0:30, 0])
    # ax_y.fill_between(np.linspace(0, len(self.y), len(self.y)), 0, 4, where=np.logical_and(self.osc_mask, ~self.n2n3),
    #                   alpha=0.4, linewidth=0, color=color[0], label='Osc ∩ ~NREM')
    # ax_y.fill_between(np.linspace(0, len(self.y), len(self.y)), 0, 4, where=np.logical_and(self.osc_mask, self.n2n3),
    #                   alpha=0.2, linewidth=0, color=color[5], label='Osc ∩ NREM')
    # ax_y.plot(self.y, c='black')
    # ax_y.spines['top'].set_visible(False), ax_y.spines['right'].set_visible(False)
    # ax_y.set_xlim(0, len(self.y)), ax_y.set_ylim(-0.1, 4.7)
    # ax_y.set_xticks([])
    # ax_y.set_yticks([0, 1, 2, 3, 4], ['Wake', 'N1', 'N2', 'N3', 'REM'], fontsize=ft18)
    # ax_y.set_title(f"Sleep stage (human annotated)", fontdict={'size': ft18})
    # ax_y.set_xticks([0, 120, 240, 360, 480, 600, 720], [0, 1, 2, 3, 4, 5, 6], fontsize=ft18)
    # ax_y.set_xlabel('Time (h)', fontdict={'size': ft18})
    # legend = ax_y.legend(framealpha=0, loc=1, bbox_to_anchor=(0.8, 1), bbox_transform=ax_y.transAxes, ncol=2,
    #                      prop={'size': ft18})
    # legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0
    # plt.savefig('./paper_figure/Osc_human annotated.svg', bbox_inches='tight')
    # plt.show()

    fig = plt.figure(layout="constrained", figsize=(6, 4))
    gs = GridSpec(1, 1, figure=fig)
    ax_plot = fig.add_subplot(gs[0, 0])
    log_psd_osc_filter = self.irasa['log_osc_filter'].copy()
    ax_plot.plot(0, 0, c=color[0], label="SD > mean std")
    ax_plot.plot(0, 0, c='gray', label="SD < mean std")
    # ax_plot.plot(freqs, log_psd_osc_filter[np.logical_and(self.y == 0, osc_std > th)].T, alpha=0.1, c=color[0])
    # ax_plot.plot(freqs, log_psd_osc_filter[np.logical_and(self.y == 0, osc_std < th)].T, alpha=0.1, c='gray')

    mean = log_psd_osc_filter[~np.logical_or(wake_close, self.n2n3)].mean(axis=0)
    std = log_psd_osc_filter[~np.logical_or(wake_close, self.n2n3)].std(axis=0)
    ax_plot.plot(freqs, mean, alpha=1, c='gray')
    ax_plot.fill_between(freqs, mean - std, mean + std, alpha=0.5, color='gray')

    mean = log_psd_osc_filter[self.n2n3].mean(axis=0)
    std = log_psd_osc_filter[self.n2n3].std(axis=0)
    ax_plot.plot(freqs, mean, alpha=1, c=color[2])
    ax_plot.fill_between(freqs, mean - std, mean + std, alpha=0.5, color=color[2])

    mean = log_psd_osc_filter[wake_close].mean(axis=0)
    std = log_psd_osc_filter[wake_close].std(axis=0)
    ax_plot.plot(freqs, mean, alpha=1, c=color[0])
    ax_plot.fill_between(freqs, mean - std, mean + std, alpha=0.5, color=color[0])

    # handles, labels = plt.gca().get_legend_handles_labels()
    # from collections import OrderedDict
    # by_label = OrderedDict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys())
    # legend = ax_plot.legend(framealpha=0, bbox_to_anchor=(0.7, 0.8), ncol=1, bbox_transform=ax_plot.transAxes,
    #                        loc='center')
    # legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0

    legends = [mpatches.Patch(color=color[2], label='N2N3'),
               mpatches.Patch(color=color[0], label='Wake$_{\mathrm{close}}$'),
               mpatches.Patch(color='gray', label='Others')]
    legend = ax_plot.legend(handles=legends, framealpha=0, bbox_to_anchor=(0.8, 0.8), loc='center', ncol=1,
                            bbox_transform=ax_plot.transAxes,
                            prop={'size': ft18}, handlelength=1, markerscale=4)
    legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0

    # legend = ax_plot.legend(by_label.values(), by_label.keys(), framealpha=1, bbox_transform=ax_plot.transAxes)
    # legend.get_frame().set_linewidth(0)  # 设置边框宽度为
    ax_plot.set_xlabel('Frequency (Hz)', fontdict={'size': ft18})
    ax_plot.set_ylabel('Power (dB)', fontdict={'size': ft18})
    ax_plot.set_title('PSD$_{\mathrm{osc}}$')
    ax_plot.set_xlim(0, 25)
    ax_plot.spines['top'].set_visible(False), ax_plot.spines['right'].set_visible(False)
    plt.savefig('./paper_figure/Oscillatory_psd.svg', bbox_inches='tight')
    plt.show()


# 绘制sg滤波器平滑过后的曲线
def fig_6_a_plot_sg_n1_rem():
    sub_data = []
    for no in range(1):  # 166
        print(no)  # 104, 105一致
        if no in [27, 73, 78, 79, 104, 136, 137, 138, 139, 156, 157, 158, 159]:
            continue
        self = DatasetCreate(no=no, show=False)
        sub_data.append(self)

    # 执行睡眠分期
    for self in sub_data:
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

    # 对于每个REM期，如果与1更近，则为N1期，如果与4更近，则为REM
    new_label = self.label.copy()
    # 状态平滑过渡，由于把N1当做REM，N1处与平滑后的曲线差异较大。
    new_label_sg = np.convolve(self.label, np.ones(20) / 20, mode='same')
    # 对于每个REM期，如果与1更近，则为N1期，如果与4更近，则为REM
    for i, l in enumerate(self.label):
        if l == 4:
            if new_label_sg[i] < 2.5:
                new_label[i] = 1
                new_label_sg = np.convolve(new_label, np.ones(20) / 20, mode='same')
            else:
                new_label[i] = 4
    time = np.linspace(0, self.label.shape[0] / 120, self.label.shape[0])

    fig = plt.figure(figsize=(16, 9))
    # 子图1：位于第一行第一列，跨越2行2列
    ax1 = plt.subplot2grid(shape=(4, 4), loc=(0, 0))
    cmap = ListedColormap([color[i] for i in range(5)])
    ax1.imshow(self.y.reshape((-1, 1)).T, cmap=cmap, aspect='auto')  # 绘制颜色覆盖图
    ax1.set_xticks([]), ax1.set_yticks([])  # 设置坐标轴刻度线的位置
    # 创建图例对象
    legends = [mpatches.Patch(color=color[i], label=stage[i]) for i in range(5)]
    legend = ax1.legend(handles=legends, loc='center', handlelength=1, markerscale=1,
                        ncol=5, columnspacing=1.5, bbox_to_anchor=(0.5, 1.5), framealpha=0,
                        bbox_transform=ax1.transAxes, prop={'size': ft18})
    legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0

    # fig, ax = plt.subplots(figsize=(12, 4))
    ax2 = plt.subplot2grid(shape=(4, 4), loc=(1, 1))
    # cmap='Blues'  # 将 cmap 参数设置为新的自定义线性颜色映射：
    ax2.plot(time, self.label, c='black', linestyle='dotted', label='Hypothesis stage')
    ax2.plot(time, new_label_sg, c='red', linestyle='--', label='Smooth curve')
    ax2.plot(time, new_label, c='black', label='Result')

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    # ax2.set_ylabel('Sleep Stage', fontdict={'size': ft18})
    ax2.set_xlabel('Time (h)', fontdict={'size': ft18})
    ax2.set_yticks([0, 1, 2, 3, 4], ['Wake', 'N1', 'N2', 'N3', 'REM'])
    ax2.set_ylim(-0.5, 4.5)
    ax2.set_xlim(0, time[-1])
    for text in ax2.get_yticklabels():
        # text.set_fontfamily('Times New Roman')
        text.set_fontsize(ft18)
        text.set_horizontalalignment('center')  # 垂直中心对齐
        text.set_x(-0.03)
    for text in ax2.get_xticklabels():
        # text.set_fontfamily('Times New Roman')
        text.set_fontsize(ft18)
    legend = ax2.legend(framealpha=1, bbox_to_anchor=(0.5, 1), loc='center', ncol=3, bbox_transform=ax2.transAxes,
                        prop={'size': ft18}, handlelength=1, markerscale=4)
    legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0
    legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0

    ax3 = plt.subplot2grid(shape=(4, 4), loc=(2, 2))
    # 创建图例对象
    legends = [mpatches.Patch(color=color[i], label=stage[i]) for i in range(5)]

    for i in range(5):
        psd = np.log(self.psd.T)[:, self.y == i]
        ax3.plot(self.freqs, psd.mean(axis=1), c=color[i], alpha=1)
        ax3.fill_between(self.freqs,
                         psd.mean(axis=1) - psd.std(axis=1),
                         psd.mean(axis=1) + psd.std(axis=1), alpha=0.5, color=color[i])

    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    # ax.set_yticks([]), ax.set_xticks([])
    ax3.set_ylabel('PSD (dB)'), ax3.set_xlabel('Frequency (Hz)')
    # ax3.xaxis.set_label_coords(0.5, -0.05)
    legend = ax3.legend(handles=legends, framealpha=0, bbox_to_anchor=(1, 0.8), loc='right', ncol=2,
                        bbox_transform=ax3.transAxes,
                        prop={'size': ft18}, handlelength=1, markerscale=4)
    legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0
    ax3.set_ylim(-23, -11), ax3.set_xlim(0, 50)
    ax3.set_yticks([-12, -16, -20], [-12, -16, -20])
    # ax3.set_xticks([0, 10, 20, 30, 40, 50], [0, '', '', '', '', 50])

    ax4 = plt.subplot2grid(shape=(4, 4), loc=(2, 3))
    for i in range(5):
        ax4.scatter(self.psd_umap[new_label == i, 0], self.psd_umap[new_label == i, 1], c=color[i], s=4, label=stage[i])
    ax4.spines['top'].set_visible(False), ax4.spines['right'].set_visible(False)
    # ax4.spines['bottom'].set_visible(False), ax4.spines['left'].set_visible(False)
    ax4.set_xlabel('UMAP 1', fontdict={'size': ft18}), ax4.set_ylabel('UMAP 2', fontdict={'size': ft18})
    ax4.set_xticks([]), ax4.set_yticks([])
    # ax.tick_params(labelsize=ft18)  # 坐标轴字体大小
    legend = ax4.legend(prop={'size': ft18}, framealpha=0, handlelength=1, markerscale=4, ncol=2, columnspacing=0.5)
    legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0

    ax5 = plt.subplot2grid(shape=(4, 4), loc=(3, 3), projection='3d')
    t = np.linspace(0, self.psd_umap.shape[0] * 30 / 3600, self.psd_umap.shape[0])
    ax5.scatter(t, self.psd_umap[:, 0], self.psd_umap[:, 1], c=[color[i] for i in self.y], alpha=0.3)
    ax5.scatter(self.psd_umap[:, 0], self.psd_umap[:, 1], zs=-1.2, zdir='x', c=[color[i] for i in self.y])
    window = np.ones(19) / 19
    x, y = np.pad(self.psd_umap[:, 0], 9, 'edge'), np.pad(self.psd_umap[:, 1], 9, 'edge')
    ax5.plot(t, np.convolve(x, window, mode='valied'), np.convolve(y, window, mode='valied'), c='black')
    ax5.set_box_aspect([4, 1, 1])  # 设置 X 轴长度为 10，Y 轴和 Z 轴长度均为 1
    ax5.set_ylim(self.psd_umap[:, 0].min(), self.psd_umap[:, 0].max())
    ax5.set_zlim(self.psd_umap[:, 1].min(), self.psd_umap[:, 1].max())
    ax5.set_xlim(-1.2, t.max())
    ax5.set_yticks(np.linspace(self.psd_umap[:, 0].min() + 1, self.psd_umap[:, 0].max() - 1, 4))
    ax5.set_zticks(np.linspace(self.psd_umap[:, 1].min() + 1, self.psd_umap[:, 1].max() - 1, 4))
    ax5.set_yticks([])
    ax5.set_zticks([])
    # ax.plot(self.psd_umap[:, 0], self.psd_umap[:, 1], t, label='parametric curve')
    # 隐藏z轴与x轴交面的网格
    # 设置网格线条颜色
    ax5.grid(color=(0.8, 0.8, 0.8, 1))
    ax5.w_xaxis.gridlines.set_visible(False)
    ax5.w_zaxis.gridlines.set_visible(False)
    ax5.w_yaxis.gridlines.set_visible(False)
    # 设置展示角度
    ax5.view_init(elev=20, azim=-65)  # 设置仰角为 30 度，方位角为 45 度
    ax5.w_xaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
    ax5.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax5.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax5.set_xlabel('\n\nTime (h)')
    ax5.xaxis.set_label_position('bottom')
    ax5.set_ylabel('UMAP1')
    ax5.set_zlabel('UMAP2')
    ax5.set_zorder(-1)  # 最底层

    # 调整子图的大小和位置(x,y,width,heigh)
    ax1.set_position([0.05, 0.9, 0.55, 0.03])
    ax2.set_position([0.05, 0.55, 0.55, 0.3])
    ax3.set_position([0.67, 0.55, 0.30, 0.35])
    ax4.set_position([0.05, 0.05, 0.25, 0.4])
    ax5.set_position([-0.12, -0.25, 1.5, 1])
    plt.savefig('./paper_figure/fig_6.svg', bbox_inches='tight')
    plt.show()


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
                    [82.0, 76.0, 0, 0],  # 'UK-Sleep'
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


# 分水岭区域分割
def fig_s1_a_b_local_cluster(self):
    # 数据读取
    sub_data = []
    for no in range(1):  # 166
        print(no)  # 104, 105一致
        if no in [27, 73, 78, 79, 104, 136, 137, 138, 139, 156, 157, 158, 159]:
            continue
        self = DatasetCreate(no=no, show=False)
        sub_data.append(self)
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

    # 图1
    from matplotlib.cm import get_cmap
    from matplotlib.colors import LinearSegmentedColormap
    cmap1 = get_cmap('Purples')
    colors1 = cmap1(np.linspace(0, 1, 10))
    cmap2 = LinearSegmentedColormap.from_list('my_cmap', ['white', 'white'], N=10)
    colors2 = cmap2(np.linspace(0, 1, 2))
    # 使用 concatenate() 函数将两个颜色列表连接一起
    colors = np.concatenate([colors2, colors1[1:]])
    # colors = np.concatenate([colors1[1:]])
    # 使用 LinearSegmentedColormap 函数创建新的自定义线性颜色映射
    my_cmap = LinearSegmentedColormap.from_list('my_cmap', colors)
    levels = np.linspace(z.min(), z.max(), 10)  # 概率密度10等分
    fig, ax = plt.subplots(figsize=(6, 5))
    cb = ax.contourf(x, y, z, levels=levels, cmap=my_cmap, norm=plt.Normalize(vmin=0, vmax=0.07))
    cb.set_clim(0, 0.09)
    # cmap='Blues'  # 将 cmap 参数设置为新的自定义线性颜色映射：
    mask = np.logical_and(self.osc_mask, ~np.logical_or(self.y == 2, self.y == 3))
    ax.scatter(self.psd_umap[:, 0], self.psd_umap[:, 1], c='black', s=1, label='Sleep frame\n(30s)')
    ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
    ax.set_xlabel('UMAP 1', fontdict={'size': ft18})
    ax.set_ylabel('UMAP 2', fontdict={'size': ft18})
    ax.set_xticks([]), ax.set_yticks([])
    cbar = plt.colorbar(cb)
    cbar.ax.set_yticklabels(['0', '', '', '', '', '', '', '', '', '0.07'])
    # cbar.ax.set_title('Kernel Density Estimation', fontsize=12)
    cbar.ax.title.set_rotation(90)  # 将标题垂直显示
    cbar.set_label('Density', rotation=270, fontdict={'size': ft18}, labelpad=10)
    for label in cbar.ax.get_yticklabels():
        label.set_fontsize(ft18)
    from matplotlib.collections import PathCollection
    from matplotlib.legend_handler import HandlerPathCollection
    def update_scatter(handle, orig):
        handle.update_from(orig)
        handle.set_sizes([64])

    legend = plt.legend(handler_map={PathCollection: HandlerPathCollection(update_func=update_scatter)},
                        prop={'size': ft18}, handlelength=1, markerscale=4)
    legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0
    plt.subplots_adjust(left=0.12, bottom=0.15, right=0.95, top=0.95)
    ax.xaxis.set_label_coords(0.5, -0.08), ax.yaxis.set_label_coords(-0.08, 0.5)
    plt.savefig('./paper_figure/Sleep_contourf.svg')
    plt.show()

    # 图2
    fig, ax = plt.subplots(figsize=(5, 5))
    # ax.scatter(self.psd_umap[:, 0], self.psd_umap[:, 1], c='black', s=4, label='Sleep frame\n(30s)')
    ax.pcolormesh(x, y, labels, alpha=image_mask, cmap=cm.tab20)
    ax.scatter(x_umap[:, 0], x_umap[:, 1], c='black', s=1)  # 分布散点图
    ax.scatter(x[tuple(local_max.T)], y[tuple(local_max.T)], c='red', s=16, label='Local maximum')  # 局部最大值
    ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
    ax.set_xlabel('UMAP 1', fontdict={'size': ft18})
    ax.set_ylabel('UMAP 2', fontdict={'size': ft18})
    ax.set_xticks([]), ax.set_yticks([])
    from matplotlib.collections import PathCollection
    from matplotlib.legend_handler import HandlerPathCollection

    def update_scatter(handle, orig):
        handle.update_from(orig)
        handle.set_sizes([64])

    legend = plt.legend(handler_map={PathCollection: HandlerPathCollection(update_func=update_scatter)},
                        prop={'size': 18}, handlelength=1, markerscale=4, bbox_to_anchor=(0.3, 0.95), framealpha=0,
                        loc='center')
    legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0
    plt.subplots_adjust(left=0.12, bottom=0.15, right=0.95, top=0.95)
    ax.xaxis.set_label_coords(0.5, -0.08), ax.yaxis.set_label_coords(-0.08, 0.5)
    plt.tight_layout()
    plt.savefig('./paper_figure/watershed_contourf.svg')
    plt.show()


# PSD曲线，5个睡眠期的原始信号
def fig_s1_plot_raw_signal_psd():
    self = sub_data[0]

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
    ax.tick_params(bottom=False, top=False, left=True, right=False)
    ax.plot([1010, 1010], [0.005, -0.005], c='black')
    ax.plot([900, 1000], [0.005, 0.005], c='black')
    ax.text(942, 0.006, "1s", fontsize=ft18, c='black')
    ax.text(1015, 0, "100μV", fontsize=ft18, c='black', rotation=90, verticalalignment='center')
    ax.set_xlim(-10, 1011)
    plt.tight_layout()
    plt.savefig(f'./paper_figure/raw_signal.svg')
    plt.show()


# 探究年龄与irasa平均差值的关系
def plot_age_irasa():
    import pandas as pd
    df = pd.read_excel('E:\\data\\sleepedf\\SC-subjects.xls')
    df_np = np.array(df)
    sub_data = []
    no_age, no_sex = {}, {}
    for no in range(166):
        if no in [27, 73, 78, 79, 104, 136, 137, 138, 139, 156, 157, 158, 159]:
            continue
        idx = np.where(df_np[:, 0] == no // 2)[0][0]
        no_age[no], no_sex[no] = df_np[idx, 2], df_np[idx, 3]
        self = DatasetCreate(no=no, show=False)
        self.clc_irasa()  # 计算分形功率与震荡功率
        self.find_nrem_irasa_max_freq()  # 根据快纺锤波判断NREM
        sub_data.append(self)
    print(no_age)

    age_sp_dis, sex_sp_dis, age_sp_11_16_dis, age_fast_sp_freq = [], [], [], []
    for i, self in enumerate(sub_data):
        sp = self.fast_sp[np.logical_or(self.y == 2, self.y == 3)].mean()  # 快纺锤波功率
        sp_11_16 = self.sp_11_16[np.logical_or(self.y == 2, self.y == 3)].mean()  # 纺锤波功率
        non_sp = self.fast_sp[~np.logical_or(self.y == 2, self.y == 3)].mean()  # NREM期以外的功率

        sex, age, sp_dis, sp_11_16_dis = no_sex[self.no], no_age[self.no], sp - non_sp, sp_11_16 - non_sp
        age_sp_dis.append((age, sp_dis))  # 年龄与快纺锤波功率差值
        sex_sp_dis.append((sex, sp_dis))  # 性别与快纺锤波功率差值
        age_sp_11_16_dis.append((age, sp_11_16_dis))
        age_fast_sp_freq.append((age, self.sp_freq))  # 快纺锤波的频率

    age_sp_dis = np.array(age_sp_dis)
    sex_sp_dis = np.array(sex_sp_dis)
    age_sp_11_16_dis = np.array(age_sp_11_16_dis)
    age_fast_sp_freq = np.array(age_fast_sp_freq)

    import statsmodels.api as sm
    # 定义年龄和功率
    x, y_sp, y_sp_11_16 = age_sp_dis[:, 0], age_sp_dis[:, 1], age_sp_11_16_dis[:, 1]
    # 建立线性回归模型
    X = sm.add_constant(x)
    model = sm.OLS(y_sp, X).fit()
    print(model.summary(), np.corrcoef([x, y_sp])[0, 1])
    X = sm.add_constant(x)
    model = sm.OLS(y_sp_11_16, X).fit()
    print(model.summary(), np.corrcoef([x, y_sp_11_16])[0, 1])
    import seaborn
    fig, ax = plt.subplots(figsize=(9, 5))
    df = pd.DataFrame(age_sp_dis, columns=['Age', 'Fast spindle power'])
    seaborn.regplot(x='Age', y='Fast spindle power', data=df, ax=ax, label='Fast spindle', color='red')  # 绘制散点图
    df = pd.DataFrame(age_sp_11_16_dis, columns=['Age', 'Power difference (db)'])
    seaborn.regplot(x='Age', y='Power difference (db)', data=df, ax=ax, label='Spindle', color='black')  # 绘制散点图
    ax.set_ylabel('PSD$_{NREM}$ - PSD$_{Others}$')
    ax.plot([30], [0.5], label='y = -0.0078x + 0.8797', c='red')
    ax.plot([30], [0.5], label='y = -0.0052x + 0.5769', c='black')
    # ax.set_title('Regression with Confidence Intervals', fontsize=ft18)
    # ax.text(0.25, 0.85, "R-squared: 0.541, k: -0.0097\n"  # R^2=1 拟合程度好
    #                     "P(F-statistic) < 0.001",  # P=2.76*10-27<0.001 统计上有意义
    #         transform=plt.gca().transAxes, fontsize=ft18, c='C0')
    # ax.text(0.25, 0.7, "R-squared: 0.519, k: -0.0056\n"  # R^2=1 拟合程度好
    #                    "P(F-statistic) < 0.001",  # P=8.31e-26<0.001 统计上有意义
    #         transform=plt.gca().transAxes, fontsize=ft18, c='C1')
    ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
    legend = plt.legend(markerscale=2)
    legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0
    plt.tight_layout()
    plt.savefig('./paper_figure/spindle_age.svg', bbox_inches='tight')
    plt.show()

    # 不同性别之间的对比，没有显著差异
    # plt.scatter(age_sp_dis[sex_sp_dis[:, 0] == 1, 0], age_sp_dis[sex_sp_dis[:, 0] == 1, 1])
    # plt.scatter(age_sp_dis[sex_sp_dis[:, 0] == 2, 0], age_sp_dis[sex_sp_dis[:, 0] == 2, 1])
    # plt.show()

    plt.hist(age_fast_sp_freq[:, 1], bins=15)
    plt.xlabel('Freq')
    plt.ylabel('count')
    plt.title('max freq')
    plt.show()
    plt.scatter(age_fast_sp_freq[:, 0], age_fast_sp_freq[:, 1])
    plt.show()


# 降维特征与时间
def plot_umap_t(self):
    # self = DatasetCreate(no=0, show=False)

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(projection='3d')
    t = np.linspace(0, self.psd_umap.shape[0] * 30 / 3600, self.psd_umap.shape[0])
    ax.scatter(t, self.psd_umap[:, 0], self.psd_umap[:, 1], c=[color[i] for i in self.y], alpha=0.3)
    ax.scatter(self.psd_umap[:, 0], self.psd_umap[:, 1], zs=-1.2, zdir='x', c=[color[i] for i in self.y])
    window = np.ones(19) / 19
    x, y = np.pad(self.psd_umap[:, 0], 9, 'edge'), np.pad(self.psd_umap[:, 1], 9, 'edge')
    ax.plot(t, np.convolve(x, window, mode='valied'), np.convolve(y, window, mode='valied'), c='black')
    ax.set_box_aspect([3, 1, 1])  # 设置 X 轴长度为 10，Y 轴和 Z 轴长度均为 1
    ax.set_ylim(self.psd_umap[:, 0].min(), self.psd_umap[:, 0].max())
    ax.set_zlim(self.psd_umap[:, 1].min(), self.psd_umap[:, 1].max())
    ax.set_xlim(-1.2, t.max())
    ax.set_yticks(np.linspace(self.psd_umap[:, 0].min() + 1, self.psd_umap[:, 0].max() - 1, 4))
    ax.set_zticks(np.linspace(self.psd_umap[:, 1].min() + 1, self.psd_umap[:, 1].max() - 1, 4))
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    # ax.plot(self.psd_umap[:, 0], self.psd_umap[:, 1], t, label='parametric curve')
    # 隐藏z轴与x轴交面的网格
    # 设置网格线条颜色
    ax.grid(color=(0.8, 0.8, 0.8, 1))
    # ax.w_xaxis.gridlines.set_visible(False)
    ax.w_zaxis.gridlines.set_visible(False)
    ax.w_yaxis.gridlines.set_visible(False)
    # 设置展示角度
    ax.view_init(elev=20, azim=-60)  # 设置仰角为 30 度，方位角为 45 度
    ax.w_xaxis.set_pane_color((0.8, 0.8, 0.8, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.set_xlabel('\n\nTime (h)')
    ax.xaxis.set_label_position('bottom')
    ax.set_ylabel('Umap1')
    ax.set_zlabel('Umap2')
    fig.subplots_adjust(top=1.3, bottom=-0.2, left=-0.23, right=1.07)  # plot outside the normal area
    plt.savefig(f'./paper_figure/{self.no}_time_umap_3d.svg', bbox_inches='tight')
    # plt.close()
    plt.show()


# UK-Sleep预测的睡眠曲线
def plot_pred():
    sub_data = []
    for no in range(1):  # 166
        print(no)  # 104, 105一致
        if no in [27, 73, 78, 79, 104, 136, 137, 138, 139, 156, 157, 158, 159]:
            continue
        self = DatasetCreate(no=no, show=False)
        sub_data.append(self)
    # 执行睡眠分期
    for self in sub_data:
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
    # 状态平滑过渡，由于把N1当做REM，N1处与平滑后的曲线差异较大。
    # new_label_sg = savgol_filter(self.label, 40, 2)  # 用S-G滤波器对状态序列平滑处理
    new_label_sg = np.convolve(self.label, np.ones(20) / 20, mode='same')
    # 对于每个REM期，如果与1更近，则为N1期，如果与4更近，则为REM
    new_label = []
    for i, l in enumerate(self.label):
        if l == 4:
            if new_label_sg[i] < 2.5:
                new_label.append(1)
            else:
                new_label.append(4)
        else:
            new_label.append(l)
    new_label = np.array(new_label)
    time = np.linspace(0, self.label.shape[0] / 120, self.label.shape[0])
    fig, ax = plt.subplots(figsize=(12, 4))
    # cmap='Blues'  # 将 cmap 参数设置为新的自定义线性颜色映射：
    ax.plot(time, self.label, c='black', linestyle='dotted', label='Hypothesis')
    ax.plot(time, new_label_sg, c='red', linestyle='--', label='Smooth hypothesis')
    ax.plot(time, new_label, c='black', label='Result')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.set_ylabel('Sleep Stage', fontdict={'size': ft18})
    ax.set_xlabel('Time (h)', fontdict={'size': ft18})
    ax.set_yticks([0, 1, 2, 3, 4], ['Wake', 'N1', 'N2', 'N3', 'REM'])
    ax.set_ylim(-0.5, 4.5)
    ax.set_xlim(0, time[-1])
    for text in ax.get_yticklabels():
        # text.set_fontfamily('Times New Roman')
        text.set_fontsize(ft18)
        text.set_horizontalalignment('center')  # 垂直中心对齐
        text.set_x(-0.06)
    for text in ax.get_xticklabels():
        # text.set_fontfamily('Times New Roman')
        text.set_fontsize(ft18)
    legend = ax.legend(framealpha=1, bbox_to_anchor=(0.5, 1), loc='center', ncol=3, bbox_transform=ax.transAxes,
                       prop={'size': ft18}, handlelength=1, markerscale=4)
    legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0
    plt.tight_layout()
    plt.savefig('./paper_figure/pred.svg', bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(figsize=(4, 4))
    for i in range(5):
        ax.scatter(self.psd_umap[new_label == i, 0], self.psd_umap[new_label == i, 1], c=color[i], s=4, label=stage[i])
    ax.spines['top'].set_visible(False), ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False), ax.spines['left'].set_visible(False)
    # ax.set_xlabel('UMAP 1', fontdict={'size': ft18}), ax.set_ylabel('UMAP 2', fontdict={'size': ft18})
    ax.set_xticks([]), ax.set_yticks([])
    # ax.tick_params(labelsize=ft18)  # 坐标轴字体大小
    legend = plt.legend(prop={'size': ft18}, framealpha=0, handlelength=1, markerscale=4, ncol=2, columnspacing=0.5)
    legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0
    plt.tight_layout()
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
    plt.savefig('./paper_figure/scatter.svg')
    plt.show()


# YASA的混淆矩阵
def yasa_acc():
    # 数据读取
    sub_data = []
    for no in range(166):  # 166
        print(no)  # 104, 105一致
        if no in [27, 73, 78, 79, 104, 136, 137, 138, 139, 156, 157, 158, 159]:
            continue
        self = DatasetCreate(no=no, show=False)
        sub_data.append(self)
    # yasa 分期结果 153分类acc0.71% 39分类：0.77%
    from sklearn.metrics import classification_report
    y_pred_all, y_real_all = [], []
    for self in sub_data:
        if self.no >= 40:
            break
        info = mne.create_info(ch_names=["EEG 1"], sfreq=100, ch_types=["eeg"])  # 创建info对象
        raw = mne.io.RawArray(self.x_signal.flatten()[np.newaxis, :], info)  # 利用mne.io.RawArray创建raw对象
        sls = yasa.SleepStaging(raw, eeg_name="EEG 1")
        y_pred = sls.predict()
        sleep_map = {'N1': 1, 'N2': 2, 'N3': 3, 'R': 4, 'W': 0}
        y_pred = [sleep_map[i] for i in y_pred]
        print(self.no, np.sum(y_pred == self.y) / self.y.shape[0])
        y_pred_all.append(np.array(y_pred))
        y_real_all.append(self.y)
    y_pred, y_real = np.concatenate(y_pred_all), np.concatenate(y_real_all)
    print(np.sum(y_pred == y_real) / y_real.shape[0])
    print(classification_report(y_pred, y_real))

    from sklearn.metrics import classification_report
    y_pred_all, y_real_all = [], []
    for self in sub_data:
        info = mne.create_info(ch_names=["EEG 1"], sfreq=100, ch_types=["eeg"])  # 创建info对象
        raw = mne.io.RawArray(self.x_signal.flatten()[np.newaxis, :], info)  # 利用mne.io.RawArray创建raw对象
        sls = yasa.SleepStaging(raw, eeg_name="EEG 1")
        y_pred = sls.predict()
        sleep_map = {'N1': 1, 'N2': 2, 'N3': 3, 'R': 4, 'W': 0}
        y_pred = [sleep_map[i] for i in y_pred]
        print(self.no, np.sum(y_pred == self.y) / self.y.shape[0])
        y_pred_all.append(np.array(y_pred))
        y_real_all.append(self.y)
    y_pred, y_real = np.concatenate(y_pred_all), np.concatenate(y_real_all)
    print(np.sum(y_pred == y_real) / y_real.shape[0])
    print(classification_report(y_pred, y_real))


def plot_confusion_matrix():
    # 数据读取
    sub_data = []
    for no in range(166):  # 166
        print(no)  # 104, 105一致
        if no in [27, 73, 78, 79, 104, 136, 137, 138, 139, 156, 157, 158, 159]:
            continue
        self = DatasetCreate(no=no, show=False)
        sub_data.append(self)

    # 执行睡眠分期
    for self in sub_data:
        self.sleep_stage()

    for no, self in enumerate(sub_data):
        self.label = distinguish_n1_rem_new(self)

    y_real = np.concatenate([self.y for self in sub_data])
    y_pred = np.concatenate([self.label for self in sub_data])
    # 输出混淆矩阵
    import sklearn
    c_m = sklearn.metrics.confusion_matrix(y_real, y_pred)
    print(c_m)
    from sklearn.metrics import classification_report
    print(classification_report(y_pred, y_real))
    # 混淆矩阵
    classes = ['Wake', 'N1 ', 'N2 ', 'N3 ', 'REM']
    fontsize = 20
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.grid(False)
    c_m = c_m.astype("float") / c_m.sum(axis=1)[:, np.newaxis]
    im = ax.imshow(c_m, interpolation="nearest", cmap=plt.cm.Blues)
    # We want to show all ticks...
    plt.title(f'UK-Sleep\n'
              f'SleepEDF-20', fontdict={'fontsize': fontsize})
    plt.xlabel('Predicted label', fontdict={'fontsize': fontsize})
    plt.ylabel('True label', fontdict={'fontsize': fontsize})
    plt.xticks([0, 1, 2, 3, 4], classes, fontsize=fontsize)
    plt.yticks([0, 1, 2, 3, 4], classes, fontsize=fontsize)

    # Loop over data dimensions and create text annotations.
    fmt = ".2f"
    thresh = c_m.max() / 2.0
    for i in range(c_m.shape[0]):
        for j in range(c_m.shape[1]):
            ax.text(j, i, format(c_m[i, j], fmt), ha="center", va="center",
                    color="white" if c_m[i, j] > thresh else "black", size=20)
    plt.tight_layout()
    plt.savefig('./paper_figure/cm_uk_edf_20.svg', dpi=600)
    plt.show()

    y_real = np.concatenate([self.y for i, self in enumerate(sub_data) if i < 39])
    y_pred = np.concatenate([self.label for i, self in enumerate(sub_data) if i < 39])
    # 输出混淆矩阵
    import sklearn
    c_m = sklearn.metrics.confusion_matrix(y_real, y_pred)
    print(c_m)
    from sklearn.metrics import classification_report
    print(classification_report(y_pred, y_real))
    # 混淆矩阵
    classes = ['Wake', 'N1 ', 'N2 ', 'N3 ', 'REM']
    fontsize = 20
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.grid(False)
    c_m = c_m.astype("float") / c_m.sum(axis=1)[:, np.newaxis]
    im = ax.imshow(c_m, interpolation="nearest", cmap=plt.cm.Blues)
    # We want to show all ticks...
    plt.title(f'UK-Sleep\n'
              f'SleepEDF-78', fontdict={'fontsize': fontsize})
    plt.xlabel('Predicted label', fontdict={'fontsize': fontsize})
    plt.ylabel('True label', fontdict={'fontsize': fontsize})
    plt.xticks([0, 1, 2, 3, 4], classes, fontsize=fontsize)
    plt.yticks([0, 1, 2, 3, 4], classes, fontsize=fontsize)

    # Loop over data dimensions and create text annotations.
    fmt = ".2f"
    thresh = c_m.max() / 2.0
    for i in range(c_m.shape[0]):
        for j in range(c_m.shape[1]):
            ax.text(j, i, format(c_m[i, j], fmt), ha="center", va="center",
                    color="white" if c_m[i, j] > thresh else "black", size=20)
    plt.tight_layout()
    plt.savefig('./paper_figure/cm_uk_edf_78.svg', dpi=600)
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
        ax.text(coords[i, 0], coords[i, 1], stage[i], color=color[i], size=ft18 * 2,
                horizontalalignment='center', verticalalignment='center', weight='bold')
    for i in range(5):
        for j in range(5):
            if i != j:
                if transition_probabilities[i, j] > 0.05:
                    alpha = 1
                else:
                    alpha = transition_probabilities[i, j] / 0.05
                alpha = alpha = transition_probabilities[i, j]
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
    transition_probabilities = np.zeros_like(transition_matrix)
    title = ['YA', 'MA', 'EA']
    for i, t_m in enumerate(transition_matrix):
        transition_probabilities[i] = t_m / np.sum(t_m, axis=1, keepdims=True)
        my_arrow_graph(transition_probabilities[i], title=title[i], save=True)

    for i in range(3):
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        ax.pie(state_count[i][1:5],
               labels=stage[1:5],
               colors=[color[j] for j in range(1, 5)],
               explode=(0.1, 0.1, 0.1, 0.1),
               startangle=180,
               shadow=False,
               autopct='%1.1f%%',
               pctdistance=0.6,
               textprops={'size': ft18}, wedgeprops={'alpha': 0.5})
        # ax.set_title(f'group {i}', fontsize=ft18)
        # 设置注释相对位置
        # plt.tight_layout()
        plt.show()

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
        xi, yi = np.arange(4 * (i - 1), 4 * i - 1), np.array([y[:, i].mean(), m[:, i].mean(), o[:, i].mean()])
        yerr = np.array([y[:, i].std(), m[:, i].std(), o[:, i].std()])
        yi_test, h = yi.max() + yerr.max() + 0.02, 0.05
        xticks += [j for j in xi]
        ax.bar(xi, yi, yerr=yerr, capsize=5, alpha=0.7, facecolor=color[i], label=stage[i])
        # 年轻人与中年人
        ax.plot([xi[0], xi[0], xi[1] - 0.1, xi[1] - 0.1],
                [yi_test, yi_test + h, yi_test + h, yi_test], lw=1, c="k")
        stat, p_value = ttest_ind(y[:, i], m[:, i], equal_var=False)
        ax.text((xi[0] + xi[1]) * 0.5, yi_test + h, convert_pvalue_to_asterisks(p_value),
                ha='center', va='bottom', color="k", fontsize=ft18)
        # 中年人与老年人
        ax.plot([xi[1] + 0.1, xi[1] + 0.1, xi[2], xi[2]],
                [yi_test, yi_test + h, yi_test + h, yi_test], lw=1, c="k")
        stat, p_value = ttest_ind(m[:, i], o[:, i], equal_var=False)
        ax.text((xi[1] + xi[2]) * 0.5, yi_test + h, convert_pvalue_to_asterisks(p_value),
                ha='center', va='bottom', color="k", fontsize=ft18)
        # 年轻人与老年人
        ax.plot([xi[0], xi[0], xi[2], xi[2]],
                [yi_test + 0.1, yi_test + h + 0.1, yi_test + h + 0.1, yi_test + 0.1], lw=1, c="k")
        stat, p_value = ttest_ind(y[:, i], o[:, i], equal_var=False)
        ax.text((xi[0] + xi[2]) * 0.5, yi_test + h + 0.1, convert_pvalue_to_asterisks(p_value),
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

    transition_probabilities = transition_matrix / np.sum(transition_matrix, axis=2, keepdims=True)


# 局部类连接关系图
def plot_local_class(self):
    # 状态数量
    class_num = len(np.unique(self.local_class))
    # 局部类中人工标注最多的状态 , 使用Counter计算数字的出现频率
    class_y = np.array([Counter(self.y[self.local_class == c]).most_common()[0] for c in range(1, class_num + 1)])

    # 根据人工标注和局部类修正后的状态变化序列
    # y_fix = [class_y[i-1][0] for i in self.local_class]
    # 状态转移矩阵
    transition_matrix = np.zeros((class_num, class_num))
    # 统计状态转移次数
    for i in range(len(self.local_class) - 1):
        if self.local_class[i] != self.local_class[i + 1]:  # 排除不发生状态转移的点
            transition_matrix[self.local_class[i] - 1, self.local_class[i + 1] - 1] += 1
    # plt.imshow(transition_matrix)
    # plt.show()
    # 状态转移概率
    transition_probabilities = transition_matrix / np.sum(transition_matrix, axis=1, keepdims=True)
    # plt.imshow(transition_probabilities)
    # plt.show()

    from matplotlib.patches import Ellipse
    # 类中心
    fig = plt.figure(figsize=(5, 5))
    # 子图1：局部类区域
    ax1 = plt.subplot2grid(shape=(10, 10), loc=(0, 0))
    xy = np.array([self.psd_umap[self.local_class == i + 1].mean(axis=0) for i in range(class_num)])
    x_min, x_max = self.psd_umap[:, 0].min(), self.psd_umap[:, 0].max()
    y_min, y_max = self.psd_umap[:, 1].min(), self.psd_umap[:, 1].max()
    ax1.set_xlim(x_min - 0.5, x_max + 0.5)
    ax1.set_ylim(y_min - 0.5, y_max + 0.5)
    # 绘制连接
    for i in range(class_num):
        for j in range(class_num):
            if i != j:
                # 确定连接的起始和结束位置
                (x_start, y_start), (x_end, y_end) = xy[i], xy[j]
                # 计算连接的透明度和宽度
                alpha = transition_probabilities[i, j]
                if alpha > 0.01:
                    linewidth = alpha * 10
                    # 绘制平滑曲线连接
                    ax1.plot([x_start, x_end], [y_start, y_end], color=f'C{class_y[i, 0]}', alpha=0,
                             linewidth=linewidth)
                    ax1.annotate("", xy=xy[i], xytext=xy[j], size=4, va="center", ha="center", zorder=1,
                                 arrowprops=dict(facecolor=f'C{class_y[i, 0]}',
                                                 edgecolor=f'C{class_y[i, 0]}',
                                                 alpha=min(alpha * 2, 1),
                                                 arrowstyle="simple",
                                                 connectionstyle="arc3,rad=0.4",
                                                 linewidth=linewidth))
    # 绘制节点
    for i in range(class_num):
        # 确定节点中心位置
        (x, y), k = xy[i], np.sqrt(np.sum(self.local_class == i + 1) / len(self.y) * class_num)
        # 绘制节点圆圈
        circle = Ellipse((x, y), k, (y_max - y_min) / (x_max - x_min) * k, color=f'C{class_y[i, 0]}', ec='black', lw=1)
        ax1.add_artist(circle)
        # 绘制节点名称
        ax1.text(x, y, f'{i + 1}', ha='center', va='center', fontsize=ft18 * k / 1.5)
    # 调整子图的大小和位置(x,y,width,heigh)
    ax1.set_position([0.05, 0.05, 0.9, 0.9])
    # 隐藏坐标轴刻度
    ax1.set_xticks([])
    ax1.set_yticks([])

    plt.savefig(f'./local_class/{self.no}.png', dpi=600)
    plt.close()
    # plt.show()


def loacl_cluster_plot():
    # 数据读取
    sub_data = []
    for no in range(166):  # 166
        print(no)  # 104, 105一致
        if no in [27, 73, 78, 79, 104, 136, 137, 138, 139, 156, 157, 158, 159]:
            continue
        self = DatasetCreate(no=no, show=False)
        self.local_cluster()
        sub_data.append(self)

    for self in sub_data:
        plot_local_class(self)

    # 统计微状态的数量
    t_num = np.array([(len(self.y), len(np.unique(self.local_class))) for self in sub_data])
    plt.scatter(t_num[:, 0], t_num[:, 1])
    plt.show()

    # 状态数量
    class_num = len(np.unique(self.local_class))
    # 局部类中人工标注最多的状态 , 使用Counter计算数字的出现频率
    class_y = np.array([Counter(self.y[self.local_class == c]).most_common()[0] for c in range(1, class_num + 1)])

    # 似乎N1期偏离unknow更多一些？
    window = np.ones(19) / 19
    x, y = np.pad(self.psd_umap[:, 0], 9, 'edge'), np.pad(self.psd_umap[:, 1], 9, 'edge')
    x, y = np.convolve(x, window, mode='valied'), np.convolve(y, window, mode='valied')
    # 平滑前后位置会发生漂移
    shift = np.array([x - self.psd_umap[:, 0], y - self.psd_umap[:, 1]]).T
    shift_dis = np.hypot(shift[:, 0], shift[:, 1])
    plt.plot(shift_dis)
    plt.plot(self.y)
    plt.show()

    plt.scatter(shift[self.y == 4, 0], shift[self.y == 4, 1])
    plt.scatter(shift[self.y == 1, 0], shift[self.y == 1, 1])
    plt.show()
    for i in range(5):
        plt.scatter(x[self.y == i], y[self.y == i])
    plt.show()

    # 绘制REM期和N1期前20分钟的运动曲线
    for self in sub_data:
        window = np.ones(19) / 19
        x, y = np.pad(self.psd_umap[:, 0], 9, 'edge'), np.pad(self.psd_umap[:, 1], 9, 'edge')
        x, y = np.convolve(x, window, mode='valied'), np.convolve(y, window, mode='valied')

        # 绘制REM期前20分钟的运动曲线
        mask = np.zeros_like(self.y)
        for idx in np.where(self.y == 4)[0]:
            mask[idx - 20:idx] = 1
        mask[self.y == 4] = 0  # rem期去除

        start = np.where(np.diff(mask) == 1)[0] + 1
        end = np.where(np.diff(mask) == -1)[0]
        for i, j in zip(start, end):
            plt.plot(x[i:j], y[i:j], color=color[4])
        # plt.show()

        # 绘制N1期前20分钟的运动曲线
        mask = np.zeros_like(self.y)
        for idx in np.where(self.y == 1)[0]:
            mask[idx - 20:idx] = 1
        mask[self.y == 1] = 0  # N1期去除

        start = np.where(np.diff(mask) == 1)[0] + 1
        end = np.where(np.diff(mask) == -1)[0]
        for i, j in zip(start, end):
            plt.plot(x[i:j], y[i:j], color=color[1])
        plt.show()

    # # 创建大图对象
    fig = plt.figure(figsize=(15, 5))
    # 子图1：局部类区域
    ax1 = plt.subplot2grid(shape=(10, 10), loc=(0, 0))
    ax1.scatter(self.psd_umap[:, 0], self.psd_umap[:, 1], c=[f'C{i}' for i in self.local_class])
    # 子图2：局部类区域与主状态
    ax2 = plt.subplot2grid(shape=(10, 10), loc=(0, 1))
    ax2.scatter(x, y, c=[f'C{i}' for i in self.y])
    ax2.scatter(self.psd_umap[:, 0], self.psd_umap[:, 1], alpha=0)
    # 子图2：人工标注
    ax3 = plt.subplot2grid(shape=(10, 10), loc=(0, 2))
    ax3.scatter(self.psd_umap[:, 0], self.psd_umap[:, 1], c=[f'C{i}' for i in self.y])

    # 调整子图的大小和位置(x,y,width,heigh)
    ax1.set_position([0.05, 0.1, 0.25, 0.85])
    ax2.set_position([0.35, 0.1, 0.25, 0.85])
    ax3.set_position([0.65, 0.1, 0.25, 0.85])
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
    return "ns"


if __name__ == "__main__":
    fig_1_b_c_plot_y_psd_sig()
    """
    

    # 数据读取
    sub_data = []
    for no in range(1):  # 166
        print(no)  # 104, 105一致
        if no in [27, 73, 78, 79, 104, 136, 137, 138, 139, 156, 157, 158, 159]:
            continue
        self = DatasetCreate(no=no, show=False)
        sub_data.append(self)

    # 执行睡眠分期
    for self in sub_data:
        self.sleep_stage()

    for no, self in enumerate(sub_data):
        self.label = distinguish_n1_rem_new(self)

    # 输出正确率
    acc78 = clc_acc(sub_data=sub_data, max_no=166)
    acc20 = clc_acc(sub_data=sub_data, max_no=39)
    # 画出睡眠状态曲线
    # for self in sub_data:
    #     self.plot_y_pred_y_real()

    fig, ax = plt.subplots(figsize=(6, 4))
    # legends = [mpatches.Patch(color=color[i], label=stage[i]) for i in range(5)]

    for i in range(5):
        psd = self.irasa['log_osc'].T[:, self.y == i]
        # psd = np.log(self.psd.T)[:, self.y == i]
        ax.plot(np.arange(0, len(psd)), psd.mean(axis=1), c=color[i], alpha=1)
        ax.fill_between(np.arange(0, len(psd)),
                        psd.mean(axis=1) - psd.std(axis=1),
                        psd.mean(axis=1) + psd.std(axis=1), alpha=0.5, color=color[i])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.set_yticks([]), ax.set_xticks([])
    ax.set_ylabel('PSD$_{osc}$ (dB)'), ax.set_xlabel('Frequency (Hz)')
    # ax3.xaxis.set_label_coords(0.5, -0.05)
    # legend = ax.legend(handles=legends, framealpha=0, bbox_to_anchor=(1, 0.8), loc='right', ncol=2,
    #                     bbox_transform=ax.transAxes,
    #                     prop={'size': ft18}, handlelength=1, markerscale=4)
    # legend.get_frame().set_linewidth(0)  # 设置边框宽度为 0
    # ax.set_xlim(0, 50)
    # ax.set_yticks([-12, -16, -20], [-12, -16, -20])
    plt.tight_layout()
    plt.show()

    # 同一个被试，不同的两个晚上的的功率谱密度存在显著差异
    sub_data = []
    for no in range(2):  # 166
        print(no)  # 104, 105一致
        if no in [27, 73, 78, 79, 104, 136, 137, 138, 139, 156, 157, 158, 159]:
            continue
        self = DatasetCreate(no=no, show=False)
        sub_data.append(self)
    log_psd = np.log(np.concatenate([sub_data[0].psd, sub_data[1].psd]))
    psd_umap = get_umap(log_psd)

    fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(6, 8))
    ax0.scatter(psd_umap[:sub_data[0].psd.shape[0], 0], psd_umap[:sub_data[0].psd.shape[0], 1],
                c='C0', alpha=0.05, marker='o')
    ax0.scatter(psd_umap[sub_data[0].psd.shape[0]:, 0], psd_umap[sub_data[0].psd.shape[0]:, 1],
                c='C1', alpha=0.05, marker='o')
    ax0.spines['top'].set_visible(False), ax0.spines['right'].set_visible(False)
    ax0.spines['bottom'].set_visible(False), ax0.spines['left'].set_visible(False)
    ax0.set_xticks([]), ax0.set_yticks([])
    ax1.scatter(psd_umap[:sub_data[0].psd.shape[0], 0], psd_umap[:sub_data[0].psd.shape[0], 1],
               c=[color[i] for i in sub_data[0].y], alpha=0.5, marker='o')
    ax1.scatter(psd_umap[sub_data[0].psd.shape[0]:, 0], psd_umap[sub_data[0].psd.shape[0]:, 1],
               c=[color[i] for i in sub_data[1].y], alpha=1, marker='o')
    ax1.spines['top'].set_visible(False), ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False), ax1.spines['left'].set_visible(False)
    ax1.set_xticks([]), ax1.set_yticks([])
    plt.tight_layout()
    plt.show()
    """
