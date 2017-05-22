import numpy as np

def generate_data(T, D, beta):
    return D * (T ** beta)

T = np.array([313.7, 314.9, 375.2, 474.7, 481.0, 573.5, 671.1])
D = 0.197
beta = 1.75
mean = 0.0
std = 10.0
noise = np.random.normal(mean, std, len(T))

data = generate_data(T, D, beta) + noise

np.savetxt('data.txt', data, fmt='%5.2f')
