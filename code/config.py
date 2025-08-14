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
# 数据处理配置
# ===================
class ProcessingConfig:
    """数据处理配置类"""
    # 采样率
    SAMPLE_RATE = 100
    
    # 频谱分析参数
    NPERSEG = int(30 * 100)  # 30秒窗口
    NOVERLAP = 0
    
    # 频率范围
    FREQ_MIN = 0.2
    FREQ_MAX = 30.0
    
    # Gamma频率范围
    GAMMA_FREQ_MIN = 25
    GAMMA_FREQ_MAX = 50
    
    # UMAP降维参数
    UMAP_N_COMPONENTS = 2
    UMAP_RANDOM_STATE = 42
    
    # 中值滤波参数
    MEDFILT_KERNEL_SIZE = 5

# ===================
# 算法参数配置
# ===================
class AlgorithmConfig:
    """算法参数配置类"""
    
    # IRASA算法参数
    IRASA_HSET = [1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4]
    IRASA_BAND = (1, 25)
    IRASA_WIN_SEC = 4
    
    # 阈值分割参数
    OTSU_INTERVALS = 100  # 阈值搜索间隔数
    
    # 核密度估计参数
    KDE_BANDWIDTH = "scott"
    KDE_KERNEL = "gaussian"
    
    # 纺锤波检测参数
    SPINDLE_FREQ_MIN = 11
    SPINDLE_FREQ_MAX = 16
    SPINDLE_CENTER_FREQ = 14
    
    # 慢波检测参数
    SLOW_WAVE_THRESHOLD = 0.2

# ===================
# 文件命名配置
# ===================
class NamingConfig:
    """文件命名配置类"""
    
    # 数据文件命名模板
    PICKLE_FILE_TEMPLATE = './data/class_data_sub_{}.pkl'
    
    # 图像文件命名模板
    FIGURE_FILE_TEMPLATE = './figure/no_{}_{}.png'
    PAPER_FIGURE_FILE_TEMPLATE = './paper_figure/{}.svg'
    
    # NPZ文件命名规则
    NPZ_FILE_PATTERNS = [
        'SC4{}{}E0.npz',
        'SC4{}{}F0.npz', 
        'SC4{}{}G0.npz'
    ]

# ===================
# 全局配置类
# ===================
class Config:
    """全局配置类"""
    
    # 导入所有配置类
    PATH = PathConfig
    VIS = VisualizationConfig
    PROC = ProcessingConfig
    ALGO = AlgorithmConfig
    NAME = NamingConfig
    
    @classmethod
    def setup_all(cls):
        """设置所有配置"""
        cls.PATH.ensure_directories()
        cls.VIS.setup_matplotlib()

# ===================
# 便捷访问函数
# ===================
def get_stage_color(stage_id: int) -> str:
    """获取睡眠阶段对应的颜色"""
    return Config.VIS.STAGE_COLORS.get(stage_id, '#000000')

def get_stage_name(stage_id: int) -> str:
    """获取睡眠阶段名称"""
    if 0 <= stage_id < len(Config.VIS.STAGE_NAMES):
        return Config.VIS.STAGE_NAMES[stage_id]
    return f"Stage_{stage_id}"

def get_npz_filename(sub: int, day: str) -> str:
    """获取NPZ文件名"""
    sub_str = f"0{sub}" if sub < 10 else str(sub)
    for pattern in Config.NAME.NPZ_FILE_PATTERNS:
        filename = pattern.format(sub_str, day)
        full_path = os.path.join(Config.PATH.NPZ_ROOT, filename)
        if os.path.exists(full_path):
            return full_path
    return None

# 初始化配置
Config.setup_all()