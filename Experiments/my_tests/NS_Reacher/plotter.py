import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from os import path

sns.set()
N = 100
x = np.arange(N)

load_path = path.join(path.abspath(path.join(path.dirname(__file__), '..')), 'NS_Reacher')

prowls_mean = np.load(load_path+'/ProWLS/30_Trials_0_Speed/rewards_mean.npy')
prowls_sem = np.load(load_path+'/ProWLS/30_Trials_0_Speed/rewards_sem.npy')

onpg_mean = np.load(load_path+'/ONPG/30_Trials_0_Speed/rewards_mean.npy')
onpg_sem = np.load(load_path+'/ONPG/30_Trials_0_Speed/rewards_mean.npy')

prowls_mean_subsampled = prowls_mean[::10]
prowls_sem_subsampled = prowls_sem[::10]

onpg_mean_subsampled = onpg_mean[::10]
onpg_sem_subsampled = onpg_sem[::10]

plt.plot(x, prowls_mean_subsampled, 'b-', label='ProWLS')
plt.fill_between(x, prowls_mean_subsampled - prowls_sem_subsampled, prowls_mean_subsampled + prowls_sem_subsampled, color='b', alpha=0.2)
plt.plot(x, onpg_mean_subsampled, 'r-', label='ONPG')
plt.fill_between(x, onpg_mean_subsampled - onpg_sem_subsampled, onpg_mean_subsampled + onpg_sem_subsampled, color='r', alpha=0.2)

'''
plt.plot(x, prowls_mean, 'b-', label='ProWLS')
plt.fill_between(x, prowls_mean - prowls_sem, prowls_mean + prowls_sem, color='b', alpha=0.2)
plt.plot(x, onpg_mean, 'r-', label='ONPG')
plt.fill_between(x, onpg_mean - onpg_sem, onpg_mean + onpg_sem, color='r', alpha=0.2)
'''
plt.legend(title='NS_Reacher')
plt.show()
