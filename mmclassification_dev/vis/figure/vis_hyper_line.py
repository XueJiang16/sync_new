import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置Seaborn风格
sns.set_style("whitegrid")
sns.set_palette("deep")
# 横坐标和纵坐标数据
x_values = [0.01, 0.012, 0.014, 0.016, 0.018, 0.02, 0.022, 0.024, 0.026, 0.028, 0.03, 0.008, 0.006]
y_values = [94.91, 95.01, 95.06, 95.08, 95.07, 95.05, 95.01, 94.96, 94.91, 94.84, 94.78, 94.76, 94.52]

x_values, y_values = zip(*sorted(zip(x_values, y_values)))

# 创建折线图
plt.figure(figsize=(8, 6))  # 设置图形大小
plt.plot(x_values, y_values, marker='o', linestyle='-', markersize=6)

# 添加标题和标签
plt.title("FS+TA+Energy", fontsize=16)
plt.xlabel("λ", fontsize=14)
plt.ylabel("AUROC (%)", fontsize=14)

# 设置横坐标刻度
plt.xticks(x_values, fontsize=12, rotation=45)

# 设置纵坐标刻度
plt.yticks(fontsize=12)
plt.ylim(92, 98)


# 添加网格线
plt.grid(True, linestyle='--', alpha=0.6)

# 显示图例

# 保存图像为文件（可选）
plt.savefig("hyper1.pdf")

# # 显示图像
# plt.show()
