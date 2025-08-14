#!/usr/bin/env python3
"""
UK-Sleep 重构模块使用示例

这个文件展示了如何使用重构后的模块：
- data_processor.py: 算法和数据处理
- plotting_functions.py: 画图和可视化

使用方法:
1. 先运行 data_processor.py 进行数据处理
2. 再运行 plotting_functions.py 进行可视化
3. 本文件提供完整的流程示例
"""
from aisleep_code.plotting_functions import (fig_2a_plot_gamma_hist, fig_2_b_plot_wake, fig_2_c_boxplot,
                                             fig_3_plot_irasa_new, fig_3_e_spindle_boxplot, fig_4_plot_n3,
                                             fig_4_so_boxplot, fig_5d_osc_boxplot, fig_5_plot_alpha)

def main():
    # fig_2a_plot_gamma_hist()
    # fig_2_b_plot_wake()
    # fig_2_c_boxplot()
    # fig_3_plot_irasa_new()
    # fig_4_plot_n3()
    # fig_4_so_boxplot()
    fig_5_plot_alpha()


if __name__ == "__main__":
    main()