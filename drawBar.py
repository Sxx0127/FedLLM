import matplotlib.pyplot as plt
import numpy as np

# 示例数据
max_len = 10
categories = list(range(1, 21))
part1_group1 = []
with open('topk_AB_time.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        part1_group1.append(float(line[line.find('in') + 9:line.find(', Comp') - 1]))
max_len = min(max_len, len(part1_group1))
part2_group1 = []
with open('topk_AB_time.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        part2_group1.append(float(line[line.find('on') + 9:line.find(', Aggre') - 1]))
max_len = min(max_len, len(part2_group1))
part3_group1 = []
with open('topk_AB_time.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        part3_group1.append(float(line[line.find('gre') + 10:]))
max_len = min(max_len, len(part3_group1))

part1_group2 = []
with open('new_time.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        part1_group2.append(float(line[line.find('in') + 9:line.find(', Comp') - 1]))
max_len = min(max_len, len(part1_group2))
part2_group2 = []
with open('new_time.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        part2_group2.append(float(line[line.find('on') + 9:line.find(', Aggre') - 1]))
max_len = min(max_len, len(part2_group2))
part3_group2 = []
with open('new_time.txt', encoding='utf-8') as f:
    lines = f.readlines()
    for idx, line in enumerate(lines):
        part3_group2.append(float(line[line.find('gre') + 10:]))
max_len = min(max_len, len(part3_group2))

categories = categories[:max_len]
part1_group1 = part1_group1[:max_len]
part2_group1 = part2_group1[:max_len]
part3_group1 = part3_group1[:max_len]
part1_group2 = part1_group2[:max_len]
part2_group2 = part2_group2[:max_len]
part3_group2 = part3_group2[:max_len]
# 设置图形大小
plt.figure(figsize=(10, 6))

# 柱子的宽度
width = 0.35

# 计算柱子的位置
x = np.arange(len(categories))

# 绘制第一组的堆叠柱状图
bottom_group1 = np.zeros(len(categories))
plt.bar(x - width/2, part1_group1, width, label='TopK - Train', color='blue')
bottom_group1 += part1_group1
plt.bar(x - width/2, part2_group1, width, bottom=bottom_group1, label='TopK - Compression', color='green')
bottom_group1 += part2_group1
plt.bar(x - width/2, part3_group1, width, bottom=bottom_group1, label='TopK - Aggre', color='orange')

# 绘制第二组的堆叠柱状图
bottom_group2 = np.zeros(len(categories))
plt.bar(x + width/2, part1_group2, width, label='New - Train', color='red')
bottom_group2 += part1_group2
plt.bar(x + width/2, part2_group2, width, bottom=bottom_group2, label='New - Compression', color='purple')
bottom_group2 += part2_group2
plt.bar(x + width/2, part3_group2, width, bottom=bottom_group2, label='New - Aggre', color='brown')

# 设置 x 轴标签
plt.xticks(x, categories)

# 添加标题和标签
# plt.title('distilbert-base-multilingual-cased (0.1B)')
# plt.title('roberta-large (0.3B)')
plt.title('llama-2-7B (7B)')
plt.xlabel('Global Itaration')
plt.ylabel('Time')

# 添加图例
plt.legend()

# 显示图形
plt.show()
    