import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt


# load data
n_episodes = 80000
v_1 = np.empty((0,n_episodes))
v_2 = np.empty((0,n_episodes))
for i in range(30):
    vec = np.load("disc_return_history_alg_ONPG_speed_1_seed_{}.npy".format(i))
    #vec = np.load("disc_return_history_alg_ProWLS_speed_4_seed_{}.npy".format(i))
    #v_prowls = np.vstack((v_prowls, vec))
    v_1 = np.vstack((v_1, vec))

    vec = np.load("episode_length_alg_ONPG_speed_1_seed_{}.npy".format(i))
    #vec = np.load("disc_return_history_alg_ONPG_speed_0_seed_{}.npy".format(i))
    #v_baseline = np.vstack((v_baseline, vec))
    v_2 = np.vstack((v_2, vec))

bin_size = 100
rep_size = int(n_episodes/bin_size)

v_1_mean = np.zeros(rep_size)
v_2_mean = np.zeros(rep_size)

v_1_sem = np.zeros(rep_size)
v_2_sem = np.zeros(rep_size)

for i in range(rep_size):
    v_1_mean[i] = np.mean(v_1[:,bin_size*i:bin_size*(i+1)])
    v_1_sem[i] = stats.sem(v_1[:,bin_size*i:bin_size*(i+1)], axis=None)

    v_2_mean[i] = np.mean(v_2[:, bin_size * i:bin_size * (i + 1)])
    v_2_sem[i] = stats.sem(v_2[:, bin_size * i:bin_size * (i + 1)], axis=None)

np.save("Mean_ONPG_returns.npy", v_1_mean)
np.save("Sem_ONPG_returns.npy", v_1_sem)

np.save("Mean_ONPG_length.npy", v_2_mean)
np.save("Sem_ONPG_length.npy", v_2_sem)

