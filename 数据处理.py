import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

# ====================== 强制加载中文字体 ======================
# 定义跨系统兼容的中文字体候选列表（Windows/Mac/Linux 全覆盖）
chinese_fonts = [
    'SimHei',          # Windows 黑体
    'Microsoft YaHei',  # Windows 微软雅黑
    'PingFang SC',     # Mac 苹方
    'WenQuanYi Zen Hei',# Linux 文泉驿正黑
    'Arial Unicode MS' # 通用 fallback
]

# 遍历候选字体，找到第一个可用的并设置为全局默认
selected_font = None
for font in chinese_fonts:
    try:
        font_prop = FontProperties(family=[font])
        if font_prop.get_name():  # 验证字体是否存在
            selected_font = font
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            print(f"✅ 成功加载中文字体：{font}")
            break
    except Exception as e:
        continue

if not selected_font:
    print("⚠️ 未找到可用中文字体，请手动安装 SimHei/Microsoft YaHei 后重试")

# 统一图表风格
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300  # 高清输出

# ====================== 读取数据 ======================
df = pd.read_csv('diabetes_cleaned.csv')

# ====================== 图1：特征与标签相关性排序 ======================
corr_with_target = df.corr()['Outcome'].drop('Outcome').sort_values(ascending=False)
plt.figure(figsize=(10, 6))
corr_with_target.plot(kind='bar', color='#3498db', alpha=0.85)
plt.title('各特征与糖尿病风险的相关性排序', fontproperties=font_prop, fontsize=15, pad=20)
plt.xlabel('特征名称', fontproperties=font_prop, fontsize=12)
plt.ylabel('相关系数', fontproperties=font_prop, fontsize=12)
plt.xticks(rotation=30, ha='right', fontproperties=font_prop)
plt.yticks(fontproperties=font_prop)
plt.tight_layout()
plt.savefig('特征与标签相关性排序.png', dpi=300, bbox_inches='tight')
plt.close()

# ====================== 图2：特征间相关性热力图 ======================
corr_matrix = df.corr()
plt.figure(figsize=(11, 9))
sns.heatmap(corr_matrix, 
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            linewidths=0.6,
            vmin=-1, vmax=1,
            annot_kws={'size':10, 'fontproperties':font_prop})
plt.title('特征间相关性热力图', fontproperties=font_prop, fontsize=15, pad=20)
plt.xticks(fontproperties=font_prop)
plt.yticks(fontproperties=font_prop)
plt.tight_layout()
plt.savefig('特征间相关性热力图.png', dpi=300, bbox_inches='tight')
plt.close()

# ====================== 图3：正负样本特征箱线图 ======================
features = df.drop('Outcome', axis=1).columns
df_melt = pd.melt(df, id_vars='Outcome', value_vars=features, var_name='特征', value_name='标准化数值')

plt.figure(figsize=(15, 8))
sns.boxplot(x='特征', y='标准化数值', hue='Outcome', 
            data=df_melt, 
            palette=['#2ecc71', '#e74c3c'],
            width=0.7)
plt.title('糖尿病患者与健康人群特征分布对比箱线图', fontproperties=font_prop, fontsize=15, pad=20)
plt.xlabel('特征', fontproperties=font_prop, fontsize=12)
plt.ylabel('标准化数值', fontproperties=font_prop, fontsize=12)
plt.xticks(rotation=30, ha='right', fontproperties=font_prop)
plt.yticks(fontproperties=font_prop)
plt.legend(title='糖尿病状态', labels=['健康(0)', '患病(1)'], loc='upper right', prop=font_prop)
plt.tight_layout()
plt.savefig('正负样本特征分布箱线图.png', dpi=300, bbox_inches='tight')
plt.close()

# ====================== 图4：Top4关键特征直方图 ======================
top4_features = ['Glucose', 'BMI', 'Age', 'Insulin']
diabetic = df[df['Outcome'] == 1]
healthy = df[df['Outcome'] == 0]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, col in enumerate(top4_features):
    axes[i].hist(healthy[col], bins=22, alpha=0.7, label='健康(0)', color='#2ecc71', edgecolor='white')
    axes[i].hist(diabetic[col], bins=22, alpha=0.7, label='患病(1)', color='#e74c3c', edgecolor='white')
    axes[i].set_title(f'{col} 特征分布对比', fontproperties=font_prop, fontsize=13)
    axes[i].set_xlabel('标准化数值', fontproperties=font_prop)
    axes[i].set_ylabel('样本数量', fontproperties=font_prop)
    axes[i].legend(loc='upper right', prop=font_prop)
    axes[i].grid(alpha=0.3)
    axes[i].tick_params(axis='both', labelsize=10)
    axes[i].set_xticklabels(axes[i].get_xticks(), fontproperties=font_prop)
    axes[i].set_yticklabels(axes[i].get_yticks(), fontproperties=font_prop)

plt.suptitle('Top4关键特征分布对比直方图', fontproperties=font_prop, fontsize=16, y=0.98)
plt.tight_layout()
plt.savefig('关键特征分布对比直方图.png', dpi=300, bbox_inches='tight')
plt.close()

