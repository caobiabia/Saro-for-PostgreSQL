import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 生成示例数据
data = np.random.rand(10000, 50)  # 100个样本，50个特征

# 执行t-SNE
tsne = TSNE(n_components=2, random_state=0)
reduced_data = tsne.fit_transform(data)

# 可视化结果
plt.figure(figsize=(8, 6))
plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
plt.title('t-SNE Visualization')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()