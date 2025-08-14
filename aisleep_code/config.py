"""
UK-Sleep 项目配置文件
将所有配置参数集中管理
"""

import os
from typing import Dict, List, Tuple

# ===================
# 路径配置
# ===================
class PathConfig:
    """路径配置类"""
    # 数据文件根目录
    NPZ_ROOT = 'D:\\code\\实验室代码\\data\\eeg_fpz_cz\\'
    
    # 输出目录
    FIGURE_DIR = './figure/'
    DATA_DIR = './data/'
    PAPER_FIGURE_DIR = './paper_figure/'
    
    # 确保目录存在
    @classmethod
    def ensure_directories(cls):
        """确保所有必要的目录都存在"""
        directories = [cls.FIGURE_DIR, cls.DATA_DIR, cls.PAPER_FIGURE_DIR]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

# ===================
# 可视化配置
# ===================
class VisualizationConfig:
    """可视化配置类"""
    # 字体设置
    FONT_SIZE = 18
    FONT_FAMILY = 'Arial'
    SVG_FONTTYPE = 'none'  # 生成的字体不再以svg的形式保存
    
    # 颜色配置
    STAGE_COLORS = {
        0: '#317EC2',  # Wake 蓝色
        1: '#F2B342',  # N1 粉色
        2: '#5AAA46',  # N2 绿色
        3: '#C03830',  # N3 红色
        4: '#825CA6',  # REM 紫色
        5: '#C43D96',  # Wake(Open eye) 黄橙色
        6: '#8D7136'   # NREM
    }
    
    # 睡眠阶段名称
    STAGE_NAMES = ['Wake', 'N1', 'N2', 'N3', 'REM']
    
    # 图像设置
    FIGURE_DPI = 300
    DEFAULT_FIGSIZE = (12, 8)
    
    @classmethod
    def setup_matplotlib(cls):
        """设置matplotlib全局配置"""
        import matplotlib as mpl
        mpl.rcParams.update({'font.size': cls.FONT_SIZE})
        mpl.rcParams['font.family'] = cls.FONT_FAMILY
        mpl.rcParams['svg.fonttype'] = cls.SVG_FONTTYPE



# ===================
# 全局配置类
# ===================
class Config:
    """全局配置类"""
    
    # 导入所有配置类
    PATH = PathConfig
    VIS = VisualizationConfig

    
    @classmethod
    def setup_all(cls):
        """设置所有配置"""
        cls.PATH.ensure_directories()
        cls.VIS.setup_matplotlib()



# 初始化配置
Config.setup_all()