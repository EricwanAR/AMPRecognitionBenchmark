import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def get_metrics(y_true, y_scores):
    # 计算 ROC 曲线
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    # 计算 AUC 值
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc

# 读取单个CSV文件
try:
    data = pd.read_csv('preds.csv')
except FileNotFoundError:
    print("未找到 preds.csv 文件。请确保文件存在于当前目录。")
    exit()

# 获取真实值
y_true = data['gt'].values

# 定义要处理的列名和对应的显示名称
columns_to_process = {
    'Concatenation': 'Concatenation',
    'Weighted': 'Weighted',
    'Attention': 'Attention'
}

# 定义学术风格的配色方案
colors = plt.cm.tab10.colors  # 使用 Matplotlib 的 tab10 调色板
color_cycle = iter(colors)    # 颜色循环器

plt.style.use('seaborn-v0_8-white')
# 绘制 ROC 曲线
plt.figure(figsize=(6, 6))

# 为每个方法绘制 ROC 曲线
for col, display_name in columns_to_process.items():
    try:
        y_scores = data[col].values
        fpr, tpr, roc_auc = get_metrics(y_true, y_scores)
        color = next(color_cycle)  # 从配色方案中获取下一个颜色
        plt.plot(fpr, tpr, lw=2, color=color, label=f'{display_name} (AUC = {roc_auc:.3f})')
    except KeyError:
        print(f"警告：在CSV文件中未找到列 '{col}'")

# 添加随机猜测的基线
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, label='Random guess')

# 设置图表样式
plt.xlabel('False Positive Rate', fontsize=14)
plt.ylabel('True Positive Rate', fontsize=14)
plt.title('ROC Curve for Classification Methods', fontsize=16, weight='bold')
plt.legend(loc='lower right', fontsize=12, frameon=False, fancybox=False)

# 去除上框线和右框线
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 调整图表边框、刻度等样式以适合学术风格
plt.tight_layout()
plt.savefig('auroc.svg')
plt.show()