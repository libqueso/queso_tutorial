import numpy as np
import matplotlib
matplotlib.use('pdf')
from matplotlib import pyplot as plt

# Shamelessly stolen from statsmodels but I changed fft to True and unbiased
# to True
def acovf(x, unbiased=True, demean=True, fft=True):
    '''
    Autocovariance for 1D

    Parameters
    ----------
    x : array
    Time series data. Must be 1d.
    unbiased : bool
    If True, then denominators is n-k, otherwise n
    demean : bool
    If True, then subtract the mean x from each element of x
    fft : bool
    If True, use FFT convolution.  This method should be preferred
    for long time series.

    Returns
    -------
    acovf : array
    autocovariance function
    '''
    x = np.squeeze(np.asarray(x))
    if x.ndim > 1:
        raise ValueError("x must be 1d. Got %d dims." % x.ndim)
    n = len(x)

    if demean:
        xo = x - x.mean()
    else:
        xo = x
    if unbiased:
        xi = np.arange(1, n + 1)
        d = np.hstack((xi, xi[:-1][::-1]))
    else:
        d = n * np.ones(2 * n - 1)
    if fft:
        nobs = len(xo)
        Frf = np.fft.fft(xo, n=nobs * 2)
        acov = np.fft.ifft(Frf * np.conjugate(Frf))[:nobs] / d[n - 1:]
        return acov.real
    else:
        return (np.correlate(xo, xo, 'full') / d)[n - 1:]

N = 5000
data = np.loadtxt('outputData/ip_raw_chain.m')
D = data[:,0]
beta = data[:,1]
D = D[N:]
beta = beta[N:]

fig = plt.figure()
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)

acov_D = acovf(D)
acov_D /= acov_D[0]

acov_beta = acovf(beta)
acov_beta /= acov_beta[0]

M = 3000
ax1.plot(D)
ax1.set_title('D')
ax2.plot(beta)
ax2.set_title(r'$\beta$')
ax3.plot(acov_D[:M])
ax4.plot(acov_beta[:M])

sum = 0
for i in range(len(acov_D)):
    if acov_D[i] < 0.01:
        break
    sum += acov_D[i]

ESS = N / (1.0 + 2.0 * sum)
print ESS

fig.savefig('autocorrs.pdf')

white_noise = np.random.normal(0, 1, 1000)

fig2 = plt.figure()
ax1 = fig2.add_subplot(2, 1, 1)
ax2 = fig2.add_subplot(2, 1, 2)
ax1.plot(white_noise)
ax2.plot(acovf(white_noise)[:50])

fig2.savefig('autocorrs_white_noise.pdf')
