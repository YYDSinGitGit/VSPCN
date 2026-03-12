


import matplotlib.pyplot as plt
import numpy as np

# ???????
pooling_methods = ['Max Pooling', 'Average Pooling', 'Stochastic Pooling']
categories = ['CUB', 'AWA2', 'SUN']

# ???????????
data = [
    [57.0, 88.2, 58.2],  # Max Pooling ?? CUB, AWA2, SUN ???
    [58.2, 90.7, 61.4],  # Average Pooling ?? CUB, AWA2, SUN ???
    [55.9, 85.9, 58.9]   # Stochastic Pooling ?? CUB, AWA2, SUN ???
]

# ??????????????
x = np.arange(len(categories))
width = 0.2  # ?????

fig, ax = plt.subplots()
for i, method in enumerate(pooling_methods):
    ax.bar(x + i * width, data[i], width, label=method)

# ????????
ax.set_ylabel('value')
#ax.set_title('Scores by pooling method and category')
ax.set_xticks(x + width)
ax.set_xticklabels(categories)
ax.legend()

# ??????????
plt.xticks(x + width / 2, categories)

plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# ??????????
plt.savefig('pooling_method.pdf', dpi=300)

# ????
#plt.show()

