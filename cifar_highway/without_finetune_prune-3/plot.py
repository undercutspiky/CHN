import numpy as np
from matplotlib import pyplot as plt

accuracies = [88.5, 88.39, 87.8, 87.61, 87.64, 87.17, 86.91, 86.35, 85.7]
sizes = [32, 12, 9.2, 8.3, 7.8, 7.3, 7.0, 6.7, 6.4]

fig, ax1 = plt.subplots()
ax1.plot([89.3]*len(accuracies), '--', c='b', alpha=0.5)
ax1.plot(accuracies, 'bo', ls='--')
ax1.set_xlabel('Threshold')
ax1.set_ylabel('Accuracy in %')

ax2 = ax1.twinx()
ax2.plot([34]*len(sizes), '--', c='r', alpha=0.5)
ax2.plot(sizes, 'ro', ls='--')
ax2.set_ylabel('Size in Mb')

plt.xticks(range(9), ['%.3f' % i for i in np.arange(0.005, 0.046, 0.005)])
plt.tight_layout(pad=0)
fig.savefig('results.pdf')
plt.show()
