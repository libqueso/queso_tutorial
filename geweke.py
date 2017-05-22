import numpy as np
import matplotlib
matplotlib.use('pdf')
from matplotlib import pyplot as plt

def geweke(chain, num_samples):
    chain1 = np.mean(chain[:num_samples])  # The first num_samples samples
    chain2 = np.mean(chain[-num_samples:])  # The last num_samples samples
    return chain1 - chain2

N = 5000
data = np.loadtxt('outputData/ip_raw_chain.m')
D = data[:,0]
D = D[N:]
white_noise = np.random.normal(0, 1, 100000)

fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

gewekes = []
gewekes_white_noise = []
for i in range(1, 50000):  # This loop is (N^2) but it's only for the workshop
    g = geweke(D, i)
    gewekes.append(geweke(D, i))
    gewekes_white_noise.append(geweke(white_noise, i))

ax1.plot(gewekes)
ax1.set_title('D')
ax2.plot(gewekes_white_noise)
ax2.set_title('White noise')
fig.savefig('geweke.pdf')
