import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置Seaborn风格
sns.set_style("whitegrid")
sns.set_palette("deep")
# 横坐标和纵坐标数据
x_values = [500, 600, 700, 800, 900, 1000, 1100, 1200, 1300]
y_values = [93.21, 93.26, 93.30, 93.35, 93.39, 93.44, 93.48, 93.52, 93.57]

x_values, y_values = zip(*sorted(zip(x_values, y_values)))

# 创建折线图
plt.figure(figsize=(8, 6))  # 设置图形大小
plt.plot(x_values, y_values, marker='o', linestyle='-', markersize=6)

# 添加标题和标签
plt.title("FS+TA+ODIN", fontsize=16)
plt.xlabel("λ", fontsize=14)
plt.ylabel("AUROC (%)", fontsize=14)

# 设置横坐标刻度
plt.xticks(x_values, fontsize=12, rotation=45)

# 设置纵坐标刻度
plt.yticks(fontsize=12)
plt.ylim(90, 96)


# 添加网格线
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
# 显示图例

# 保存图像为文件（可选）
plt.savefig("hyper2.pdf")

# # 显示图像
# plt.show()
