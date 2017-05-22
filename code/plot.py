import numpy as np
import matplotlib
matplotlib.use('pdf')
from matplotlib import pyplot as plt

data = np.loadtxt('outputData/ip_raw_chain.m')

N = 0
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

ax1.plot(data[N:,0])
ax2.plot(data[N:,1])
fig.savefig('samples.pdf')
