"""
数据处理主模块 - 算法部分
"""

import os
import mne
import umap
import yasa
import pickle
import numpy as np
from scipy import signal
import scipy.stats as stats
from scipy.stats import norm, ttest_ind, ttest_rel
from scipy import ndimage as ndi
from lspopt import spectrogram_lspopt
from scipy.ndimage import gaussian_filter
from skimage.segmentation import watershed
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
from aisleep_code import root_path

# 输出正确率
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score

# 第一步，先运行1_20提取原始信号.py 在目录下生成包含153次睡眠实验预处理后的信号的npz文件
npz_root = 'D:\\code\\实验室代码\\data\\eeg_fpz_cz\\'

# 颜色定义
stage = ['Wake', 'N1', 'N2', 'N3', 'REM']


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
    threshold, loss, interval = start, 0, (end - start) / 100
    
    for th in np.arange(start + interval, end - interval, interval):
        c0, c1 = x[x <= th], x[x > th]
        w0, w1 = c0.shape[0] / num_all, c1.shape[0] / num_all
        u0, u1 = c0.mean(), c1.mean()
        temp_loss = w0 * w1 * (u0 - u1) * (u0 - u1)
        if loss < temp_loss:
            threshold, loss = th, temp_loss
    
    label = np.zeros_like(x) + (x > threshold)
    
    if return_threshold:
        return label, threshold
    
    proba = np.copy(x)
    u0, u1 = x[label == 0].mean(), x[label == 1].mean()
    num_u0 = np.sum(np.logical_and(u0 < x, x < threshold))
    num_u1 = np.sum(np.logical_and(threshold < x, x < u1))
    
    for i, p in enumerate(proba):
        if p < u0:
            proba[i] = 0
        elif p > u1:
            proba[i] = 1
        elif p < threshold:
            proba[i] = 0.5 * np.sum(np.logical_and(u0 <= x, x <= p)) / num_u0
        else:
            proba[i] = 0.5 + 0.5 * np.sum(np.logical_and(threshold <= x, x <= p)) / num_u1
    
    proba = (proba - 0.5) * 2
    proba[proba < 0] = 0
    return label, proba


# 非参数核密度估计
def kernel_density_estimation(data, weight):
    kde = KernelDensity(kernel="gaussian", bandwidth="scott").fit(data, sample_weight=weight)
    x, y = data[:, 0], data[:, 1]
    nx, ny, dx, dy = 100, 100, (x.max() - x.min()) / 98, (y.max() - y.min()) / 98
    x, y = np.linspace(x.min() - dx, x.max() + dx, nx), np.linspace(y.min() - dy, y.max() + dx, ny)
    xv, yv = np.meshgrid(x, y)
    xy = np.vstack([xv.ravel(), yv.ravel()]).T
    p = np.exp(kde.score_samples(xy).reshape(100, 100))
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

        if os.path.exists(f'{root_path}/data/class_data_sub_{self.no}.pkl'):
            f = open(f'{root_path}/data/class_data_sub_{self.no}.pkl', 'rb')
            self.__dict__.update(pickle.load(f))
            print("Successfully read!")
        else:
            self.read_all_data_per_person()  # 读取单次实验的信号
            self.compute_sxx()  # 计算信号特征
            self.compute_psd()  # 计算功率谱密度特征
            self.save()

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
        f = open(f'{root_path}/data/class_data_sub_{self.no}.pkl', 'wb')
        pickle.dump(self.__dict__, f)
        print('Successfully save!')


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



        # ax[2].pcolormesh(x, y, mask_so)
        x, y = x_umap[:, 0], x_umap[:, 1]  # 按照位置映射关系给每个局部簇打上标签
        nx, ny, dx, dy = 100, 100, (x.max() - x.min()) / 98, (y.max() - y.min()) / 98  # 100*100像素
        x, y = np.linspace(x.min() - dx, x.max() + dx, nx), np.linspace(y.min() - dy, y.max() + dx, ny)
        px, py = np.copy(x_umap[:, 0]), np.copy(x_umap[:, 1])
        px, py = (px - x.min()) / (x.max() - x.min()), (py - y.min()) / (y.max() - y.min())
        px, py = np.around(px * 100), np.around(py * 100)
        self.local_class = labels[py.astype('int'), px.astype('int')]  # 每个点所属的局部类别


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


if __name__ == "__main__":
    # 使用示例
    sub_data = sleep_stage_all()

    y_p, y_t = [], []
    for no, self in enumerate(sub_data):
        if self.no < 40:
            y_p_i = self.label
            y_p.append(y_p_i)
            y_t.append(self.y)
    y_p, y_t = np.concatenate(y_p), np.concatenate(y_t)
    acc_20 = accuracy_score(y_p, y_t)
    k_20 = cohen_kappa_score(y_p, y_t)
    mf1_20 = f1_score(y_p, y_t, average='macro')
    print(classification_report(y_p, y_t))

    y_p, y_t = [], []
    for no, self in enumerate(sub_data):
        y_p_i = self.label
        y_p.append(y_p_i)
        y_t.append(self.y)

    y_p, y_t = np.concatenate(y_p), np.concatenate(y_t)
    acc_78 = accuracy_score(y_p, y_t)
    k_78 = cohen_kappa_score(y_p, y_t)
    mf1_78 = f1_score(y_p, y_t, average='macro')
    print(classification_report(y_p, y_t))

    