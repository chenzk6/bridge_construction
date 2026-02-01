import pandas as pd
import matplotlib.pyplot as plt
import os

# 读取CSV文件
csv_path = '/home/xilifeng/bridge_construction/logs/lh_finetune/progress.csv'
df = pd.read_csv(csv_path)

# 选择关键列进行可视化（可以根据需要调整）
columns_to_plot = [
    'success_rate',
    'ep_reward_mean',
    'policy_loss',
    'value_loss',
    'entropy',
    'kl_loss'
]

# 设置x轴为n_updates
x = df['n_updates']

# 创建子图
fig, axes = plt.subplots(len(columns_to_plot), 1, figsize=(10, 15))
fig.suptitle('Training Progress Visualization', fontsize=16)

for i, col in enumerate(columns_to_plot):
    axes[i].plot(x, df[col], label=col)
    axes[i].set_xlabel('n_updates')
    axes[i].set_ylabel(col)
    axes[i].set_title(f'{col} over Updates')
    axes[i].legend()
    axes[i].grid(True)

plt.tight_layout()

# 保存图表为图片
output_dir = '/home/xilifeng/bridge_construction/logs/lh_finetune/visualizations'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'training_progress.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"图表已保存到: {output_path}")

# 显示图表（可选）
plt.show()