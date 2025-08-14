# -*- coding: utf-8 -*-
"""
# @Time    : 2025/8/14 16:12
# @Author  : maixun
# @File    : stats_tools.py.py
"""
import numpy as np
import scipy.stats as stats


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