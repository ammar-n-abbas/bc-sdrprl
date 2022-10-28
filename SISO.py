############################################################################################################
#                                           IMPORTING LIBRARIES
# ##########################################################################################################

import warnings
import time

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import gym
from sklearn.preprocessing import StandardScaler, MinMaxScaler

'''import random
import sklearn.preprocessing
import sklearn.pipeline
import torch
import random
import copy
import seaborn as sns
import tensorflow.keras
import math
from IPython.display import clear_output
from gym import spaces
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import LnMlpPolicy, LnCnnPolicy
from stable_baselines import DQN
from stable_baselines.common.env_checker import check_env
from IPython.display import display
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn.decomposition import PCA
from collections import deque
import itertools'''

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.simplefilter("ignore", category=RuntimeWarning)

standard = StandardScaler()
minmax = MinMaxScaler()
dir_path = os.getcwd()

############################################################################################################
# **********************************************************************************************************
#                                              CASE STUDIES
# **********************************************************************************************************
# ##########################################################################################################

############################################################################################################
#                                              NASA-CMAPSS
# ##########################################################################################################

'''dataset = 'train_FD002.txt'
df_full = pd.read_csv(dir_path + r'/CMAPSSData/' + dataset, sep=" ", header=None, skipinitialspace=True).dropna(axis=1)
df_full = df_full.rename(columns={0: 'unit', 1: 'cycle', 2: 'W1', 3: 'W2', 4: 'W3'})
# df = df_full

# mapping
train_set = [*range(1, 23), *range(24, 32), *range(33, 39), 40, 44, 45, 46, 49, 51, 53,
             *range(55, 61), 62, 63, 64, 66, 67, 69, 70, 71, 74, 78, 81, 88, 94, 97, 102, 103, 105,
             106, 107, 108, 118, 120, 128, 133, 136, 137, 141, 165, 173, 176, 178]
test_set = [185, 188, 192, 194, 197, 208, 212, 214, 217, 219, 225, 231, 234, 238, 244, 252, 253, 256, 258, 260]
combined_set = train_set + test_set

df = pd.DataFrame()
for engines in combined_set:
    df = pd.concat([df, df_full[df_full['unit'] == engines]], ignore_index=True)

zip_iterator = zip(combined_set, list(range(1, 101)))
mapping_dict = dict(zip_iterator)
df["unit"] = df["unit"].map(mapping_dict)

df_A = df[df.columns[[0, 1]]]
df_W = df[df.columns[[2, 3, 4]]]
df_S = df[df.columns[list(range(5, 26))]]
df_X = pd.concat([df_W, df_S], axis=1)

# RUL as sensor reading
# df_A['RUL'] = 0
# for i in range(1, 101):
# df_A['RUL'].loc[df_A['unit'] == i] = df_A[df_A['unit'] == i].cycle.max() - df_A[df_A['unit'] == i].cycle

# Standardization
df_X = standard.fit_transform(df_X)

# train_test split
engine_unit = 1'''

'''##
# %% ENGINE UNIT SPECIFIC DATA
engine_unit = 1
engine_df_A = df_A[df_A['unit'] == engine_unit]
engine_df_X = df_X.iloc[engine_df_A.index[0]:engine_df_A.index[-1] + 1]
engine_df_W = df_W.iloc[engine_df_A.index[0]:engine_df_A.index[-1] + 1]

##
# %% NORMALIZE DATA
X = scaler.fit_transform(engine_df_X)
# X = (((engine_df_X - engine_df_X.mean()) / engine_df_X.std()).fillna(0))
# X = ((engine_df_X - engine_df_X.min()) / (engine_df_X.max() - engine_df_X.min())).fillna(0)).values'''

'''
##
# %% READ RUL & APPEND

# df_RUL = pd.read_csv(dir_path + '/CMAPSSData/RUL_FD001.txt', sep=" ", header=None, skipinitialspace=True).dropna(axis=1)
# df_RUL.columns = ['RUL']
# df_z_scaled_RUL = df_z_scaled.join(df_RUL, 1)

##
# %% REGRESSION TO GET "RUL distribution"

# x = df_z_scaled_RUL.iloc[:,list(range(5, 26))]
# y = df_RUL

##
# %% DIMENSIONALITY REDUCTION TO GET "HEALTH INDICATOR"

sensor_data = df_z_scaled.iloc[:, list(range(5, 26))].dropna(axis=1)
pca = PCA(n_components=1)
principalComponents = (1 - pca.fit_transform(sensor_data))

pdf = pd.DataFrame(data=principalComponents, columns=['health indicator'])
pdf_normalized = (pdf - pdf.min()) / (pdf.max() - pdf.min()) * 100

df_scaled_principal = df_z_scaled.join(pdf_normalized, 1)
df_scaled_principal = df_scaled_principal.rename(columns={0: 'engine unit', 1: 'cycle'})


##
# %% VISUALIZATION
engine_unit = 76
engine_df = df_scaled_principal[df_scaled_principal['engine unit'] == engine_unit]
# engine_df.plot.line('cycle', 'health indicator')
# plt.show()

HI = np.array(engine_df['health indicator'])[0:191].astype(np.float32)
# plt.plot(HI)
# plt.show()
'''

############################################################################################################
#                                            HYDRAULIC SYSTEM
# ##########################################################################################################
''''''
# PS1 = np.loadtxt(r'C:\Users\abbas\Desktop\case_studies\Datasets\HydraulicSystems\PS1.txt').mean(axis=1)
# PS2 = np.loadtxt(r'C:\Users\abbas\Desktop\case_studies\Datasets\HydraulicSystems\PS2.txt').mean(axis=1)
# PS3 = np.loadtxt(r'C:\Users\abbas\Desktop\case_studies\Datasets\HydraulicSystems\PS3.txt').mean(axis=1)
# PS4 = np.loadtxt(r'C:\Users\abbas\Desktop\case_studies\Datasets\HydraulicSystems\PS4.txt').mean(axis=1)
# PS5 = np.loadtxt(r'C:\Users\abbas\Desktop\case_studies\Datasets\HydraulicSystems\PS5.txt').mean(axis=1)
# PS6 = np.loadtxt(r'C:\Users\abbas\Desktop\case_studies\Datasets\HydraulicSystems\PS6.txt').mean(axis=1)
# EPS1 = np.loadtxt(r'C:\Users\abbas\Desktop\case_studies\Datasets\HydraulicSystems\EPS1.txt').mean(axis=1)
# FS1 = np.loadtxt(r'C:\Users\abbas\Desktop\case_studies\Datasets\HydraulicSystems\FS1.txt').mean(axis=1)
# FS2 = np.loadtxt(r'C:\Users\abbas\Desktop\case_studies\Datasets\HydraulicSystems\FS2.txt').mean(axis=1)
# TS1 = np.loadtxt(r'C:\Users\abbas\Desktop\case_studies\Datasets\HydraulicSystems\TS1.txt').mean(axis=1)
# TS2 = np.loadtxt(r'C:\Users\abbas\Desktop\case_studies\Datasets\HydraulicSystems\TS2.txt').mean(axis=1)
# TS3 = np.loadtxt(r'C:\Users\abbas\Desktop\case_studies\Datasets\HydraulicSystems\TS3.txt').mean(axis=1)
# TS4 = np.loadtxt(r'C:\Users\abbas\Desktop\case_studies\Datasets\HydraulicSystems\TS4.txt').mean(axis=1)
# VS1 = np.loadtxt(r'C:\Users\abbas\Desktop\case_studies\Datasets\HydraulicSystems\VS1.txt').mean(axis=1)
# CE = np.loadtxt(r'C:\Users\abbas\Desktop\case_studies\Datasets\HydraulicSystems\CE.txt').mean(axis=1)
# CP = np.loadtxt(r'C:\Users\abbas\Desktop\case_studies\Datasets\HydraulicSystems\CP.txt').mean(axis=1)
# SE = np.loadtxt(r'C:\Users\abbas\Desktop\case_studies\Datasets\HydraulicSystems\SE.txt').mean(axis=1)
#
# X = np.array([PS1, PS2, PS3, PS4, PS5, PS6, EPS1, FS1, FS2, TS1, TS2, TS3, TS4, VS1, CE, CP, SE]).T
# np.savetxt(r"C:\Users\abbas\Desktop\case_studies\Datasets\HydraulicSystems\sensors.csv", X, delimiter=" ")

# df_X = pd.read_csv(r"C:\Users\abbas\Desktop\case_studies\Datasets\HydraulicSystems\sensors.csv", sep=" ", header=None)
# df_X.columns = ['PS1', 'PS2', 'PS3', 'PS4', 'PS5', 'PS6', 'EPS1', 'FS1', 'FS2', 'TS1', 'TS2', 'TS3', 'TS4', 'VS1',
#                 'CE', 'CP', 'SE']
#
# df_Y = pd.read_csv(r"C:\Users\abbas\Desktop\case_studies\Datasets\HydraulicSystems\profile.txt", sep="\t", header=None)
# df_Y.columns = ['Cooler_condition', 'Valve_condition', 'Internal_pump_leakage', 'Hydraulic_accumulator', 'stable_flag']

'''fig, ax = plt.subplots()

ax.plot(df_Y['Valve condition'], color='red', label='Valve condition')
ax.tick_params(axis='y', labelcolor='red')
ax.legend(loc="upper right")

ax2 = ax.twinx()
ax2.plot(df_Y['Cooler condition'], color='green', label='Cooler condition')
ax2.tick_params(axis='y', labelcolor='green')
ax2.legend(loc="upper left")

ax3 = ax.twinx()
ax3.plot(df_Y['Internal pump leakage'], color='green', label='Internal pump leakage')
ax3.tick_params(axis='y', labelcolor='green')
ax3.legend(loc="lower right")

ax4 = ax.twinx()
ax4.plot(df_Y['Hydraulic accumulator'], color='blue', label='Hydraulic accumulator')
ax4.tick_params(axis='y', labelcolor='blue')
ax4.legend(loc="lower left")

ax5 = ax.twinx()
ax5.plot(df_Y['stable flag'], color='orange', label='stable flag')
ax5.tick_params(axis='y', labelcolor='orange')
ax5.legend(loc="upper center")

plt.show()'''

############################################################################################################
#                                               YOKOGAWA
# ##########################################################################################################

'''import datetime

df_yoko = pd.read_csv(r'C:/Users/abbas/Desktop/case_studies/Datasets/Yokogawa/CYG-2019-OCT-NOV edited.csv', sep=";")
df_yoko[['Date', 'Time']] = df_yoko['TimeStamp'].str.split(' ', 1, expand=True)
df_yoko['Date'] = pd.to_datetime(df_yoko['Date'], format='%d.%m.%Y')
df_yoko['Time_'] = pd.to_datetime(df_yoko['TimeStamp'])
df_yoko["Month"] = df_yoko["Date"].dt.month
df_yoko["Day"] = df_yoko["Date"].dt.day
df_yoko["Time"] = df_yoko["Time_"].dt.time
df_yoko["Hour"] = df_yoko["Time_"].dt.hour
df_yoko["Minute"] = df_yoko["Time_"].dt.minute

df_yoko["SeverityLevel"] = df_yoko["Severity"] * df_yoko["AlarmLevel"]
df_yoko['Cons'] = df_yoko['TagName'].shift(periods=1)
df_yoko['Cons2'] = df_yoko['TagName'].shift(periods=2)
df_yoko['Cons3'] = df_yoko['TagName'].shift(periods=3)

df_yoko = df_yoko.drop(columns=['SubConditionName', 'TimeStampNS', 'EngUnit', 'StationTimeMilli', 'ItemName', 'Source',
                                'ModeStatName'])
df_yoko_sub = df_yoko[df_yoko['TagName'] != '025TIC010']

df_yoko_sub = df_yoko_sub[['TimeStamp', 'TagName', 'StationName', 'ConditionName', 'Severity', 'AlarmLevel',
                           'SeverityLevel', 'AlarmOff', 'AlarmBlink', 'AckRequired', "Month", "Day", "Time",
                           "Hour", "Minute", 'Cons', 'Cons2', 'Cons3']]

count = df_yoko_sub['Severity'].value_counts()
sort = df_yoko_sub.groupby(["TagName", "Severity"]).size().reset_index(name="Frequency").sort_values(by='Frequency',
                                                                                                     ascending=[False])
df_yoko_sub.groupby(["Month", "Day", "Hour"])['TagName'].nunique().reset_index(name="Frequency").sort_values(by='Month')
event = df_yoko_sub.loc[
    (df_yoko_sub["Month"] == 11) & (df_yoko_sub["Day"] == 5) & (df_yoko_sub["Time"] == datetime.time(15, 16, 3))]

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(df_yoko_sub.corr(), dtype=bool))
sns.heatmap(df_yoko_sub.corr(), mask=mask)
'''

############################################################################################################
#                                                 TEP
# ##########################################################################################################

''''''
# test_set = 62
# T = pd.read_excel(dir_path + '/Datasets/TEP/' + str(test_set) + '/3_Deadband/TOUT.xlsx', header=None)
# Y = pd.read_excel(dir_path + '/Datasets/TEP/' + str(test_set) + '/3_Deadband/SIMOUT.xlsx', header=None).drop([18],
#                                                                                                              axis=1).to_numpy()
# U = pd.read_excel(dir_path + '/Datasets/TEP/' + str(test_set) + '/3_Deadband/XMV.xlsx', header=None).drop([4, 8, 11],
#                                                                                                           axis=1).to_numpy()
# Alarms = pd.read_excel(dir_path + '/Datasets/TEP/' + str(test_set) + '/3_Deadband/ALARMS.xlsx', header=None)
# AlarmCounts = np.count_nonzero(Alarms, axis=1)
#
# Alarms['sequence'] = Alarms.values.tolist()
# Alarms['seq_id'] = pd.factorize(Alarms['sequence'].apply(str))[0]
# Alarms['alarm_counts'] = AlarmCounts
#
# ############################################### Suppression ###############################################
#
# Y_Thresh = pd.read_csv(r"C:\Users\abbas\Desktop\case_studies\Datasets\TEP\PV_thresh.csv", header=0, delimiter=';')
# U_Thresh = pd.read_csv(r"C:\Users\abbas\Desktop\case_studies\Datasets\TEP\CV_thresh.csv", header=0, delimiter=',')
#
# ##****************************************** Alarm Sequencing **********************************************
#
# '''freq_alarm_seq = Alarms.pivot_table(index=list(Alarms.columns), aggfunc='size')
# alarm_seq = np.array(freq_alarm_seq.index.tolist())
# alarm_count_per_seq = np.count_nonzero(alarm_seq, axis=1)
# Sequence_alarms = pd.DataFrame(freq_alarm_seq)
# Sequence_alarms = Sequence_alarms.rename({0: 'freq_alarm_seq'}, axis=1)
# Sequence_alarms = pd.DataFrame(Alarms['seq_id'].value_counts(), 'freq_alarm_seq')
# Sequence_alarms['num_alarms'] = alarm_count_per_seq'''
#
# freq, count = 0.25, 5
#
# sequence = Alarms.groupby(["seq_id", "alarm_counts"]).size().reset_index(name="freq").sort_values(by='seq_id')
# thresh_freq = int(Alarms.shape[0] * freq / 100)
# thresh_count = int(Alarms.shape[1] * count / 100)
# seq_thresh = sequence.loc[((sequence['freq'] >= thresh_freq) & (sequence['alarm_counts'] >= thresh_count)), :]
#
# # TODO: add thresholding to alarm sequence frequency that doesn't occur consecutively
#
# Alarms['alarm_counts_supp_seq'] = np.where(Alarms['seq_id'].isin(seq_thresh['seq_id']), 1, Alarms['alarm_counts'])
#
# '''alarm_mean = [np.mean(Alarms['alarm_counts'])] * len(Alarms['alarm_counts'])
# supp_alarm_mean_seq = [np.mean(Alarms['alarm_counts_supp_seq'])] * len(Alarms['alarm_counts_supp_seq'])
# plt.plot(T, Alarms['alarm_counts'], label='Alarm Counts')
# plt.plot(T, alarm_mean, label='Mean Alarm', linestyle='--')
# plt.plot(T, supp_alarm_mean_seq, label='Mean Alarm Suppressed', linestyle='--')
# plt.plot(T, Alarms['alarm_counts_supp_seq'], label='Alarm Counts Suppressed Sequence')
# plt.legend()
# plt.show()'''
#
# ##***************************************** Alarm Correlation **********************************************
#
# import itertools
#
# alarms_corr = abs(Alarms[list(range(0, 81))].corr()).fillna(0)
# correlated_alarms = []
# for i in (np.array(alarms_corr.index)):
#     alarms_sort = alarms_corr[i].sort_values(ascending=False, axis=0)
#     sort_list = alarms_sort.index[(alarms_sort > 0.8) & (alarms_sort != 1)].tolist()
#     correlated_alarms.append(sort_list)
# correlated_alarms = np.array(correlated_alarms)
#
# alarm_correlation = pd.DataFrame([[i] for i in range(0, 81)], columns=['alarm'])
# alarm_correlation['corr'] = correlated_alarms
# alarm_correlation["alarm_corr"] = pd.Series([[i] for i in alarm_correlation['alarm'].values]) + alarm_correlation[
#     'corr']
#
# alarm_correlation = alarm_correlation[(alarm_correlation['alarm_corr'].str.len() != 1)]
# alarm_correlation['alarm_corr'] = alarm_correlation['alarm_corr'].apply(set)
# alarm_correlation['corr_id'] = pd.factorize(alarm_correlation['alarm_corr'].apply(str))[0]
#
# alarm_correlation = alarm_correlation.drop_duplicates(subset=['corr_id'])
#
# alarm_ids_lists = [np.nonzero(np.array(alarms_list)) for alarms_list in Alarms['sequence'].values]
# alarm_ids_sets = pd.DataFrame(np.array([set(np.array(alarm_ids_list).flatten()) for alarm_ids_list in alarm_ids_lists]))
#
# alarm_set_comb = set(np.concatenate((np.array([list(map(frozenset, itertools.combinations(sets_comb, 2))) for
#                                                sets_comb in alarm_correlation['alarm_corr'].values]))))
#
# supp_alarm_corr = []
# for A in alarm_ids_sets.values:
#     D = set()
#     for B in alarm_set_comb:
#         C = A[0] - B
#         D = D.union(C)
#     supp_alarm_corr.append(D)
# supp_alarm_corr = [list(alarm_set_seq) for alarm_set_seq in supp_alarm_corr]
#
# Alarms['alarm_counts_supp_corr'] = [np.count_nonzero(np.array(supp_corr)) for supp_corr in supp_alarm_corr]
# Alarms['alarm_counts_supp'] = Alarms[['alarm_counts_supp_seq', 'alarm_counts_supp_corr']].min(axis=1)
#
# '''supp_alarm_mean = [np.mean(Alarms['alarm_counts_supp'])] * len(Alarms['alarm_counts_supp'])
# plt.plot(T, Alarms['alarm_counts'], label='Alarm Counts')
# plt.plot(T, alarm_mean, label='Mean Alarm', linestyle='--')
# plt.plot(T, Alarms['alarm_counts_supp_seq'], label='Alarm Counts Suppressed Sequence')
# plt.plot(T, supp_alarm_mean_seq, label='Mean Alarm Suppressed Sequence', linestyle='--')
# plt.plot(T, Alarms['alarm_counts_supp'], label='Alarm Counts Suppressed (Sequence + Correlation)')
# plt.plot(T, supp_alarm_mean, label='Mean Alarm Suppressed (Sequence + Correlation)', linestyle='--')
# plt.legend()
# plt.show()'''
#
#
# ##*********************************** State Values (Reward Function) ***************************************
#
# def compute_R(Y, Y_Thresh):
#     R_values = []
#     for y in Y:
#         if np.all(y == Y_Thresh['Normal'].to_numpy()):
#             R = 0
#         elif (pd.Series(y).between(Y_Thresh['LO-Alarm'].tolist(), Y_Thresh['HI-Alarm'].tolist())).all:
#             R = -np.sum(abs(y - Y_Thresh['Normal'].to_numpy()))
#         else:
#             R = -2 * np.sum(abs(y - Y_Thresh['Normal'].to_numpy()))
#         R_values.append(R)
#     return R_values
#
#
# Alarms['Q_values'] = abs(np.array(compute_R(Y, Y_Thresh)))
# thresh_q = np.mean(Alarms['Q_values']) * 0.5
# Alarms['alarm_counts_supp_q'] = Alarms['alarm_counts_supp']
# Alarms.loc[(Alarms['Q_values'] <= thresh_q), 'alarm_counts_supp_q'] = 0
#
# '''supp_alarm_mean_q = [np.mean(Alarms['alarm_counts_supp_q'])] * len(Alarms['alarm_counts_supp_q'])
# plt.plot(T, Alarms['alarm_counts'], label='Alarm Counts')
# plt.plot(T, alarm_mean, label='Mean Alarm', linestyle='--')
# plt.plot(T, Alarms['alarm_counts_supp_seq'], label='Alarm Counts Suppressed Sequence')
# plt.plot(T, supp_alarm_mean_seq, label='Mean Alarm Suppressed Sequence', linestyle='--')
# plt.plot(T, Alarms['alarm_counts_supp'], label='Alarm Counts Suppressed (Sequence + Correlation)')
# plt.plot(T, supp_alarm_mean, label='Mean Alarm Suppressed (Sequence + Correlation)', linestyle='--')
# plt.plot(T, Alarms['alarm_counts_supp_q'], label='Alarm Counts Suppressed (Sequence + Correlation + Q)')
# plt.plot(T, supp_alarm_mean_q, label='Mean Alarm Suppressed (Sequence + Correlation + Q)', linestyle='--')
# plt.legend()
# plt.show()
# '''
#
# ############################################# Prioritization ###############################################
#
# prioritization = Alarms.loc[(Alarms['alarm_counts_supp_q'] > 10)]
# values_prior = pd.DataFrame(Y).loc[prioritization.index, :]
#
# prior = []
# for process_values in values_prior.to_numpy():
#     Q_values = []
#     for i in range(len(process_values)):
#         if process_values[i] == Y_Thresh['Normal'].to_numpy()[i]:
#             q_value = 0
#         elif Y_Thresh['LO-Alarm'].tolist()[i] <= process_values[i] <= Y_Thresh['HI-Alarm'].tolist()[i]:
#             q_value = -abs(process_values[i] - Y_Thresh['Normal'].to_numpy()[i])
#         else:
#             q_value = -2 * abs(process_values[i] - Y_Thresh['Normal'].to_numpy()[i])
#         Q_values.append(q_value)
#     prior.append(Q_values)
# prior = np.array(prior)
#
# alarms_prior = np.array([a for a in prioritization['sequence'].to_numpy()])[:, :72]
# Prior = pd.DataFrame(abs(prior * alarms_prior))
#
# priority = Prior.iloc[0]
# priority_sorted = pd.DataFrame(priority).sort_values(by=0, ascending=False)
''''''
############################################################################################################
# **********************************************************************************************************
#                                       HIDDEN MARKOV MODEL (LIBRARY)
# **********************************************************************************************************
# ##########################################################################################################

'''from hmmlearn import hmm
from random import randint
import pickle

# df_S = df[df.columns[[6, 8, 11, 12, 15, 16, 19]]]
# df_hmm = pd.concat([df_A['cycle'], df_S], axis=1)

df_hmm = minmax.fit_transform(df_S)

df_hmm = pd.DataFrame(df_hmm)
cols_to_drop = df_hmm.nunique()[df_hmm.nunique() == 1].index
df_hmm = df_hmm.drop(cols_to_drop, axis=1)
cols_to_drop = df_hmm.nunique()[df_hmm.nunique() == 2].index
df_hmm = df_hmm.drop(cols_to_drop, axis=1).to_numpy()

lengths = [df[df['unit'] == i].cycle.max() for i in range(1, df_A['unit'].max() + 1)]
# o = df_X[df_A[df_A['unit'] == 1].index[0]:df_A[df_A['unit'] == 1].index[-1] + 1]

num_states = 15
remodel = hmm.GaussianHMM(n_components=num_states,
                          n_iter=500,
                          verbose=True, )
                          # init_params="cm", params="cmt")


transmat = np.zeros((num_states, num_states))
# Left-to-right: each state is connected to itself and its
# direct successor.
for i in range(num_states):
    if i == num_states - 1:
        transmat[i, i] = 1.0
    else:
        transmat[i, i] = transmat[i, i + 1] = 0.5

# Always start in first state
# startprob = np.zeros(num_states)
# startprob[0] = 1.0

# remodel.startprob_ = startprob
# remodel.transmat_ = transmat


remodel.fit(df_hmm, lengths)


# with open("HMM_model.pkl", "wb") as file: pickle.dump(remodel, file)
# with open("filename.pkl", "rb") as file: pickle.load(file)

state_seq = remodel.predict(df_hmm, lengths)
pred = [state_seq[df[df['unit'] == i].index[0]:df[df['unit'] == i].index[-1] + 1] for i in
        range(1, df_A['unit'].max() + 1)]

prob = remodel.predict_proba(df_hmm, lengths)
prob_next_step = remodel.transmat_[state_seq, :]

HMM_out = [prob[df[df['unit'] == i].index[0]:df[df['unit'] == i].index[-1] + 1]
           for i in range(1, df_A['unit'].max() + 1)]
failure_states = [pred[i][-1] for i in range(df_A['unit'].max())]


# RUL Prediction - Monte Carlo Simulation
from sklearn.utils import check_random_state

transmat_cdf = np.cumsum(remodel.transmat_, axis=1)
random_state = check_random_state(remodel.random_state)


predRUL = []
for i in range(df_A[df_A['unit'] == 1]['cycle'].max()):
    RUL = []
    for j in range(100):
        cycle = 0
        pred_obs_seq = [df_hmm[i]]
        pred_state_seq = remodel.predict(pred_obs_seq)
        while pred_state_seq[-1] not in set(failure_states):
            cycle += 1
            prob_next_state = (transmat_cdf[pred_state_seq[-1]] > random_state.rand()).argmax()
            prob_next_obs = remodel._generate_sample_from_state(prob_next_state, random_state)
            pred_obs_seq = np.append(pred_obs_seq, [prob_next_obs], 0)
            pred_state_seq = remodel.predict(pred_obs_seq)
        RUL.append(cycle)
    # noinspection PyTypeChecker
    predRUL.append(round(np.mean(RUL)))

plt.plot(predRUL)
plt.plot(df_A[df_A['unit'] == 1].RUL)
plt.show()


plt.figure(0)
plt.plot(pred[0])
plt.plot(pred[1])
plt.plot(pred[2])
plt.xlabel('# Flights')
plt.ylabel('HMM states')
plt.show()

plt.figure(1)
E = [randint(1, df_A['unit'].max()) for p in range(0, 10)]
for e in E:
    plt.plot(pred[e - 1])
plt.xlabel('# Flights')
plt.ylabel('HMM states')
plt.legend(E, title='engine unit')

plt.figure(2)
plt.scatter(list(range(1, len(failure_states) + 1)), failure_states)
plt.xlabel('Engine units')
plt.ylabel('HMM states')
plt.legend(title='failure states')

plt.figure(3)
pca = PCA(n_components=2).fit_transform(df_hmm)
for class_value in range(num_states):
    # get row indexes for samples with this class
    row_ix = np.where(state_seq == class_value)
    plt.scatter(pca[row_ix, 0], pca[row_ix, 1])
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.legend(list(range(0, num_states)), title='HMM states')

plt.show()

import seaborn as sns
sns.heatmap(df_hmm.corr())'''

'''# Generate samples
X, Y = remodel._generate_sample_from_state(np.array([df_X[0]]))
# Plot the sampled data
plt.plot(X[:, 0], X[:, 1], ".-", label="observations", ms=6,
         mfc="orange", alpha=0.7)
plt.show()'''

'''
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture

df_X = PCA(n_components=2).fit_transform(df_X)

gmm = GaussianMixture(n_components=2, n_init=10)
gmm.fit(df_X)

print("using sklearn")
print("best pi : ", gmm.weights_)
print("best mu :", gmm.means_)


# def plot_densities(data, mu, sigma, alpha = 0.5, colors='grey'):
# grid_x, grid_y = np.mgrid[X[:,0].min():X[:,0].max():200j,
#  X[:,1].min():X[:,1].max():200j]
# grid = np.stack([grid_x, grid_y], axis=-1)
# for mu_c, sigma_c in zip(mu, sigma):
# plt.contour(grid_x, grid_y, multivariate_normal(mu_c, sigma_c).pdf(grid), colors=colors, alpha=alpha)


def plot_contours(data, means, covs):
    """visualize the gaussian components over the data"""
    plt.figure()
    plt.plot(data[:, 0], data[:, 1], 'ko')

    delta = 0.025
    k = means.shape[0]
    x = np.arange(-2.5, 10.0, delta)
    y = np.arange(-2.5, 10.0, delta)
    x_grid, y_grid = np.meshgrid(x, y)
    coordinates = np.array([x_grid.ravel(), y_grid.ravel()]).T

    col = ['green', 'red', 'indigo', 'yellow', 'blue']
    for i in range(k):
        mean = means[i]
        cov = covs[i]
        z_grid = multivariate_normal(mean, cov).pdf(coordinates).reshape(x_grid.shape)
        plt.contour(x_grid, y_grid, z_grid, colors=col[i])
    plt.tight_layout()


print("check whether the best one converged: ", gmm.converged_)
print("how many steps to convergence: ", gmm.n_iter_)

plot_contours(df_X, gmm.means_, gmm.covariances_)
plt.xlabel("data[:, 0]", fontsize=12)
plt.ylabel("data[:, 1]", fontsize=12)
plt.show()


def E_step(data, pi, mu, sigma):
    N = data.shape[0]  # number of data-points
    K = pi.shape[0]  # number of clusters, following notation used before in description
    d = mu.shape[1]  # dimension of each data point, think of these as attributes
    zeta = np.zeros((N, K))  # this is basically responsibility which should be equal to posterior.

    for nk in range(K):
        zeta[:, nk] = pi[nk] * multivariate_normal.pdf(data, mean=mu[nk], cov=sigma[nk])
        # calculate responsibility for each cluster
    zeta = zeta / np.sum(zeta, axis=1, keepdims=True)
    # use the sum over all the clusters, thus axis=1. Denominator term.
    # print ("zeta shape: ", zeta.shape)
    return zeta


def M_step(data, zeta):
    N, D = data.shape
    K = zeta.shape[1]  # use the posterior shape calculated in E-step to determine the no. of clusters
    pi = np.zeros(K)
    mu = np.zeros((K, D))
    sigma = np.zeros((K, D, D))

    for ik in range(K):
        n_k = zeta[:, ik].sum()  # we use the definition of N_k
        pi[ik] = n_k / N  # definition of the weights
        elements = np.reshape(zeta[:, ik], (zeta.shape[0], 1))
        # get each columns and reshape it (K, 1) form so that later broadcasting is possible.
        mu[ik, :] = (np.multiply(elements, data)).sum(axis=0) / n_k
        sigma_sum = 0.
        for i in range(N):
            var = data[i] - mu[ik]
            sigma_sum = sigma_sum + zeta[i, ik] * np.outer(var, var)  # outer product creates the covariance matrix
        sigma[ik, :] = sigma_sum / n_k
    return pi, mu, sigma


def elbo(data, zeta, pi, mu, sigma):
    N = data.shape[0]  # no. of data-points
    K = zeta.shape[1]  # no. of clusters
    d = data.shape[1]  # dim. of each object

    l = 0.
    for i in range(N):
        x = data[i]
        for k in range(K):
            pos_dist = zeta[i, k]  # p(z_i=k|x) = zeta_ik
            log_lik = np.log(multivariate_normal.pdf(x, mean=mu[k, :], cov=sigma[k, :, :]) + 1e-20)  # log p(x|z)
            log_q = np.log(zeta[i, k] + 1e-20)  # log q(z) = log p(z_i=k|x)
            log_pz = np.log(pi[k] + 1e-20)  # log p(z_k =1) =\pi _k
            l = (l + np.multiply(pos_dist, log_pz) + np.multiply(pos_dist, log_lik) +
                 np.multiply(pos_dist, -log_q))
    # print ("check loss: ", loss)
    return l


def train_loop(data, K, tolerance=1e-3, max_iter=500, restart=50):
    N, d = data.shape
    elbo_best = -np.inf  # loss set to the lowest value
    pi_best = None
    mu_best = None
    sigma_best = None
    zeta_f = None
    for _ in range(restart):
        pi = np.ones(K) / K  # if 3 clusters then an array of [.33, .33, .33] # the sum of pi's should be one
        # that's why normalized
        mu = np.random.rand(K, d)  # no condition on
        sigma = np.tile(np.eye(d), (K, 1, 1))  # to initialize sigma we first start with ones only at the diagonals
        # the sigmas are postive semi-definite and symmetric
        last_iter_loss = None
        all_losses = []
        try:
            for i in range(max_iter):
                zeta = E_step(data, pi, mu, sigma)
                pi, mu, sigma = M_step(data, zeta)
                l = elbo(data, zeta, pi, mu, sigma)
                if l > elbo_best:
                    elbo_best = l
                    pi_best = pi
                    mu_best = mu
                    sigma_best = sigma
                    zeta_f = zeta
                if last_iter_loss and abs(
                        (l - last_iter_loss) / last_iter_loss) < tolerance:  # insignificant improvement
                    break
                last_iter_loss = l
                all_losses.append(l)
        except np.linalg.LinAlgError:  # avoid the delta function situation
            pass
    return elbo_best, pi_best, mu_best, sigma_best, all_losses, zeta_f


best_loss, pi_best, mu_best, sigma_best, ls_lst, final_posterior = train_loop(df_X, 5)
'''

############################################################################################################
# **********************************************************************************************************
#                                            HIDDEN MARKOV MODEL 1
# **********************************************************************************************************
# ##########################################################################################################

'''engine_df_A = df_A[df_A['unit'] == engine_unit]
X = df_X[engine_df_A.index[0]:engine_df_A.index[-1] + 1]


class ProbabilityVector:
    def __init__(self, probabilities: dict):
        states = probabilities.keys()
        probs = probabilities.values()

        assert len(states) == len(probs)
        "The probabilities must match the states."

        assert len(states) == len(set(states))
        "The states must be unique."

        assert abs(sum(probs) - 1.0) < 1e-12
        "Probabilities must sum up to 1."

        assert len(list(filter(lambda x: 0 <= x <= 1, probs))) == len(probs), \
            "Probabilities must be numbers from [0, 1] interval."

        self.states = sorted(probabilities)
        self.values = np.array(list(map(lambda x:
                                        probabilities[x], self.states))).reshape(1, -1)

    @classmethod
    def initialize(cls, states: list):
        size = len(states)
        rand = np.random.rand(size) / (size ** 2) + 1 / size
        rand /= rand.sum(axis=0)
        return cls(dict(zip(states, rand)))

    @classmethod
    def from_numpy(cls, array: np.ndarray, states: list):
        return cls(dict(zip(states, list(array))))

    @property
    def dict(self):
        return {k: v for k, v in zip(self.states, list(self.values.flatten()))}

    @property
    def df(self):
        return pd.DataFrame(self.values, columns=self.states, index=['probability'])

    def __repr__(self):
        return "P({}) = {}.".format(self.states, self.values)

    def __eq__(self, other):
        if not isinstance(other, ProbabilityVector):
            raise NotImplementedError
        if (self.states == other.states) and (self.values == other.values).all():
            return True
        return False

    def __getitem__(self, state: str) -> float:
        if state not in self.states:
            raise ValueError("Requesting unknown probability state from vector.")
        index = self.states.index(state)
        return float(self.values[0, index])

    def __mul__(self, other) -> np.ndarray:
        if isinstance(other, ProbabilityVector):
            return self.values * other.values
        elif isinstance(other, (int, float)):
            return self.values * other
        else:
            NotImplementedError

    def __rmul__(self, other) -> np.ndarray:
        return self.__mul__(other)

    def __matmul__(self, other) -> np.ndarray:
        if isinstance(other, ProbabilityMatrix):
            return self.values @ other.values

    def __truediv__(self, number) -> np.ndarray:
        if not isinstance(number, (int, float)):
            raise NotImplementedError
        x = self.values
        return x / number if number != 0 else x / (number + 1e-12)

    def argmax(self):
        index = self.values.argmax()
        return self.states[index]


class ProbabilityMatrix:
    def __init__(self, prob_vec_dict: dict):
        assert len(prob_vec_dict) > 1, \
            "The numebr of input probability vector must be greater than one."
        assert len(set([str(x.states) for x in prob_vec_dict.values()])) == 1, \
            "All internal states of all the vectors must be indentical."
        assert len(prob_vec_dict.keys()) == len(set(prob_vec_dict.keys())), \
            "All observables must be unique."

        self.states = sorted(prob_vec_dict)
        self.observables = prob_vec_dict[self.states[0]].states
        self.values = np.stack([prob_vec_dict[x].values \
                                for x in self.states]).squeeze()

    @classmethod
    def initialize(cls, states: list, observables: list):
        size = len(states)
        rand = np.random.rand(size, len(observables)) \
               / (size ** 2) + 1 / size
        rand /= rand.sum(axis=1).reshape(-1, 1)
        aggr = [dict(zip(observables, rand[i, :])) for i in range(len(states))]
        pvec = [ProbabilityVector(x) for x in aggr]
        return cls(dict(zip(states, pvec)))

    @classmethod
    def from_numpy(cls, array:
    np.ndarray,
                   states: list,
                   observables: list):
        p_vecs = [ProbabilityVector(dict(zip(observables, x)))
                  for x in array]
        return cls(dict(zip(states, p_vecs)))

    @property
    def dict(self):
        return self.df.to_dict()

    @property
    def df(self):
        return pd.DataFrame(self.values,
                            columns=self.observables, index=self.states)

    def __repr__(self):
        return "PM {} states: {} -> obs: {}.".format(
            self.values.shape, self.states, self.observables)

    def __getitem__(self, observable: str) -> np.ndarray:
        if observable not in self.observables:
            raise ValueError("Requesting unknown probability observable from the matrix.")
        index = self.observables.index(observable)
        return self.values[:, index].reshape(-1, 1)


from itertools import product
from functools import reduce


class HiddenMarkovChain:
    def __init__(self, T, E, pi):
        self.T = T  # transmission matrix A
        self.E = E  # emission matrix B
        self.pi = pi
        self.states = pi.states
        self.observables = E.observables

    def __repr__(self):
        return "HML states: {} -> observables: {}.".format(
            len(self.states), len(self.observables))

    @classmethod
    def initialize(cls, states: list, observables: list):
        T = ProbabilityMatrix.initialize(states, states)
        E = ProbabilityMatrix.initialize(states, observables)
        pi = ProbabilityVector.initialize(states)
        return cls(T, E, pi)

    def _create_all_chains(self, chain_length):
        return list(product(*(self.states,) * chain_length))

    def score(self, observations: list) -> float:
        def mul(x, y): return x * y

        score = 0
        all_chains = self._create_all_chains(len(observations))
        for idx, chain in enumerate(all_chains):
            expanded_chain = list(zip(chain, [self.T.states[0]] + list(chain)))
            expanded_obser = list(zip(observations, chain))

            p_observations = list(map(lambda x: self.E.df.loc[x[1], x[0]], expanded_obser))
            p_hidden_state = list(map(lambda x: self.T.df.loc[x[1], x[0]], expanded_chain))
            p_hidden_state[0] = self.pi[chain[0]]

            score += reduce(mul, p_observations) * reduce(mul, p_hidden_state)
        return score


class HiddenMarkovChain_FP(HiddenMarkovChain):
    def _alphas(self, observations: list) -> np.ndarray:
        alphas = np.zeros((len(observations), len(self.states)))
        alphas[0, :] = self.pi.values * self.E[observations[0]].T
        for t in range(1, len(observations)):
            alphas[t, :] = (alphas[t - 1, :].reshape(1, -1)
                            @ self.T.values) * self.E[observations[t]].T
        return alphas

    def score(self, observations: list) -> float:
        alphas = self._alphas(observations)
        return float(alphas[-1].sum())


class HiddenMarkovChain_Simulation(HiddenMarkovChain):
    def run(self, length: int) -> (list, list):
        assert length >= 0, "The chain needs to be a non-negative number."
        s_history = [0] * (length + 1)
        o_history = [0] * (length + 1)

        prb = self.pi.values
        obs = prb @ self.E.values
        s_history[0] = np.random.choice(self.states, p=prb.flatten())
        o_history[0] = np.random.choice(self.observables, p=obs.flatten())

        for t in range(1, length + 1):
            prb = prb @ self.T.values
            obs = prb @ self.E.values
            s_history[t] = np.random.choice(self.states, p=prb.flatten())
            o_history[t] = np.random.choice(self.observables, p=obs.flatten())

        return o_history, s_history


class HiddenMarkovChain_Uncover(HiddenMarkovChain_Simulation):
    def _alphas(self, observations: list) -> np.ndarray:
        alphas = np.zeros((len(observations), len(self.states)))
        alphas[0, :] = list(self.pi.values()) * self.E[observations[0]].T
        for t in range(1, len(observations)):
            alphas[t, :] = (alphas[t - 1, :].reshape(1, -1) @ self.T.values) \
                           * self.E[observations[t]].T
        return alphas

    def _betas(self, observations: list) -> np.ndarray:
        betas = np.zeros((len(observations), len(self.states)))
        betas[-1, :] = 1
        for t in range(len(observations) - 2, -1, -1):
            betas[t, :] = (self.T.values @ (self.E[observations[t + 1]] * betas[t + 1, :].reshape(-1, 1))).reshape(1,
                                                                                                                   -1)
        return betas

    def uncover(self, observations: list) -> list:
        alphas = self._alphas(observations)
        betas = self._betas(observations)
        maxargs = (alphas * betas).argmax(axis=1)
        return list(map(lambda x: self.states[x], maxargs))


class HiddenMarkovLayer(HiddenMarkovChain_Uncover):
    def _digammas(self, observations: list) -> np.ndarray:
        L, N = len(observations), len(self.states)
        digammas = np.zeros((L - 1, N, N))

        alphas = self._alphas(observations)
        betas = self._betas(observations)
        score = self.score(observations)
        for t in range(L - 1):
            P1 = (alphas[t, :].reshape(-1, 1) * self.T.values)
            P2 = self.E[observations[t + 1]].T * betas[t + 1].reshape(1, -1)
            digammas[t, :, :] = P1 * P2 / score
        return digammas


class HiddenMarkovModel:
    def __init__(self, hml: HiddenMarkovLayer):
        self.layer = hml
        self._score_init = 0
        self.score_history = []

    @classmethod
    def initialize(cls, states: list, observables: list):
        layer = HiddenMarkovLayer.initialize(states, observables)
        return cls(layer)

    def update(self, observations: list) -> float:
        alpha = self.layer._alphas(observations)
        beta = self.layer._betas(observations)
        digamma = self.layer._digammas(observations)
        score = alpha[-1].sum()
        gamma = alpha * beta / score

        L = len(alpha)
        obs_idx = [self.layer.observables.index(x) \
                   for x in observations]
        capture = np.zeros((L, len(self.layer.states), len(self.layer.observables)))
        for t in range(L):
            capture[t, :, obs_idx[t]] = 1.0

        pi = gamma[0]
        T = digamma.sum(axis=0) / gamma[:-1].sum(axis=0).reshape(-1, 1)
        E = (capture * gamma[:, :, np.newaxis]).sum(axis=0) / gamma.sum(axis=0).reshape(-1, 1)

        self.layer.pi = ProbabilityVector.from_numpy(pi, self.layer.states)
        self.layer.T = ProbabilityMatrix.from_numpy(T, self.layer.states, self.layer.states)
        self.layer.E = ProbabilityMatrix.from_numpy(E, self.layer.states, self.layer.observables)

        return score

    def train(self, observations: list, epochs: int, tol=None):
        self._score_init = 0
        self.score_history = (epochs + 1) * [0]
        early_stopping = isinstance(tol, (int, float))

        for epoch in range(1, epochs + 1):
            score = self.update(observations)
            print("Training... epoch = {} out of {}, score = {}.".format(epoch, epochs, score))
            if early_stopping and abs(self._score_init - score) / score < tol:
                print("Early stopping.")
                break
            self._score_init = score
            self.score_history[epoch] = score


# HI = ["0.", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1."]
HI = np.arange(0., 1.1, 0.1)
pi = {0.: 0, 0.1: 0, 0.2: 0, 0.3: 0, 0.4: 0, 0.5: 0, 0.6: 0, 0.7: 0, 0.8: 0, 0.9: 0, 1.: 1}
model = HiddenMarkovModel.initialize(HI, df_X[0:100, 0])
model.layer.pi = pi
model.train(df_X[0:100, 0], epochs=100)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.semilogy(model.score_history)
ax.set_xlabel('Epoch')
ax.set_ylabel('Score')
ax.set_title('Training history')
plt.grid()
plt.show()'''

############################################################################################################
# **********************************************************************************************************
#                                            HIDDEN MARKOV MODEL 2
# **********************************************************************************************************
# ##########################################################################################################

'''def forward(V, a, b, initial_distribution):
    alpha = np.zeros((V.shape[0], a.shape[0]))
    alpha[0, :] = initial_distribution * b[:, V[0]]

    for t in range(1, V.shape[0]):
        for j in range(a.shape[0]):
            # Matrix Computation Steps
            #                  ((1x2) . (1x2))      *     (1)
            #                        (1)            *     (1)
            alpha[t, j] = alpha[t - 1].dot(a[:, j]) * b[j, V[t]]

    return alpha


def backward(V, a, b):
    beta = np.zeros((V.shape[0], a.shape[0]))

    # setting beta(T) = 1
    beta[V.shape[0] - 1] = np.ones((a.shape[0]))

    # Loop in backward way from T-1 to
    # Due to python indexing the actual loop will be T-2 to 0
    for t in range(V.shape[0] - 2, -1, -1):
        for j in range(a.shape[0]):
            beta[t, j] = (beta[t + 1] * b[:, V[t + 1]]).dot(a[j, :])

    return beta


def baum_welch(V, a, b, initial_distribution, n_iter=100):
    M = a.shape[0]
    T = len(V)

    for n in range(n_iter):
        alpha = forward(V, a, b, initial_distribution)
        beta = backward(V, a, b)

        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            denominator = np.dot(np.dot(alpha[t, :].T, a) * b[:, V[t + 1]].T, beta[t + 1, :])
            for i in range(M):
                numerator = alpha[t, i] * a[i, :] * b[:, V[t + 1]].T * beta[t + 1, :].T
                xi[i, :, t] = numerator / denominator

        gamma = np.sum(xi, axis=1)
        a = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))

        # Add additional T'th element in gamma
        gamma = np.hstack((gamma, np.sum(xi[:, :, T - 2], axis=0).reshape((-1, 1))))

        K = b.shape[1]
        denominator = np.sum(gamma, axis=1)
        for l in range(K):
            b[:, l] = np.sum(gamma[:, V == l], axis=1)

        b = np.divide(b, denominator.reshape((-1, 1)))

    return (a, b)


def viterbi(V, a, b, initial_distribution):
    T = V.shape[0]
    M = a.shape[0]

    omega = np.zeros((T, M))
    omega[0, :] = np.log(initial_distribution * b[:, V[0]])

    prev = np.zeros((T - 1, M))

    for t in range(1, T):
        for j in range(M):
            # Same as Forward Probability
            probability = omega[t - 1] + np.log(a[:, j]) + np.log(b[j, V[t]])

            # This is our most probable state given previous state at time t (1)
            prev[t - 1, j] = np.argmax(probability)

            # This is the probability of the most probable state (2)
            omega[t, j] = np.max(probability)

    # Path Array
    S = np.zeros(T)

    # Find the most probable last hidden state
    last_state = np.argmax(omega[T - 1, :])

    S[0] = last_state

    backtrack_index = 1
    for i in range(T - 2, -1, -1):
        S[backtrack_index] = prev[i, int(last_state)]
        last_state = prev[i, int(last_state)]
        backtrack_index += 1

    # Flip the path array since we were backtracking
    S = np.flip(S, axis=0)

    # Convert numeric values to actual hidden states
    result = []
    for s in S:
        if s == 0:
            result.append("A")
        else:
            result.append("B")

    return result


engine_df_A = df_A[df_A['unit'] == engine_unit]
V = df_X[engine_df_A.index[0]:engine_df_A.index[-1] + 1]


# Transition Probabilities
a = np.ones((30, 30))
a = a / np.sum(a, axis=1)

# Emission Probabilities
b = np.ones((V.shape[0], V.shape[1]))
b = b / np.sum(b, axis=1).reshape((-1, 1))

# Equal Probabilities for the initial distribution
initial_distribution = np.ones((V.shape[1], V.shape[0]))*0.5

a, b = baum_welch(V, a, b, initial_distribution, n_iter=1000)

predicted_observations = np.array((viterbi(V, a, b, initial_distribution)))

print(predicted_observations)'''

############################################################################################################
# **********************************************************************************************************
#                                  INPUT OUTPUT HIDDEN MARKOV MODEL (LIBRARY)
# **********************************************************************************************************
# ##########################################################################################################

'''import json
from IOHMM import UnSupervisedIOHMM
from IOHMM import OLS, CrossEntropyMNL
from hmmlearn import _hmmc
import logging

logging.getLogger().setLevel(logging.INFO)


def split_X_lengths(X, lengths):
    if lengths is None:
        return [X]
    else:
        cs = np.cumsum(lengths)
        n_samples = len(X)
        if cs[-1] > n_samples:
            raise ValueError("more than {} samples in lengths array {}"
                             .format(n_samples, lengths))
        elif cs[-1] != n_samples:
            warnings.warn(
                "less that {} samples in lengths array {}; support for "
                "silently dropping samples is deprecated and will be removed"
                    .format(n_samples, lengths),
                DeprecationWarning, stacklevel=3)
        return np.split(X, cs)[:-1]'''

############################################################################################################
#                                              NASA-CMAPSS
# ##########################################################################################################

''' num_states = 5

############################################### Training ###################################################

df[['W1', 'W2', 'W3', *range(5, 26)]] = minmax.fit_transform(pd.concat([df_W, df_S], axis=1))
cols_to_drop = df.nunique()[df.nunique() == 1].index
df = df.drop(cols_to_drop, axis=1)
cols_to_drop = df.nunique()[df.nunique() == 2].index
df = df.drop(cols_to_drop, axis=1)

outputs = [[m] for m in df.head() if type(m) == int]

df_obs = PCA(n_components=19).fit_transform(df[np.array(outputs).squeeze()])
df_obs = pd.DataFrame(df_obs)
df_input = PCA(n_components=1).fit_transform(df[['W1', 'W2']])
df_hmm = df_obs
df_hmm['W'] = df_input

outputs = [[m] for m in df_obs.head() if type(m) == int]

num_states = 15
SHMM = UnSupervisedIOHMM(num_states=num_states)
SHMM.set_models(model_emissions=[OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True),
                                 OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True),
                                 OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True),
                                 OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True),
                                 OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True)],
                model_transition=CrossEntropyMNL(solver='lbfgs'),
                model_initial=CrossEntropyMNL(solver='lbfgs'))
# We set operating conditions as the input covariate associate with the sensor output
SHMM.set_inputs(covariates_initial=[], covariates_transition=['W'],
                covariates_emissions=[['W']] * len(outputs))
SHMM.set_outputs(outputs)

lengths = [df_A[df_A['unit'] == i].cycle.max() for i in range(1, df_A['unit'].max() + 1)]
split_data = split_X_lengths(df_hmm, lengths)
SHMM.set_data(split_data)

SHMM.train()

json_dict = SHMM.to_json('../models/UnSupervisedIOHMM/')
with open('../models/UnSupervisedIOHMM/model.json', 'w') as outfile:
    json.dump(json_dict, outfile, indent=4, sort_keys=True)

# transmat = np.empty((num_states, num_states))
# for i in range(num_states):
#     transmat = np.concatenate((transmat, np.exp(SHMM.model_transition[i].predict_log_proba(np.array([[]])))))
# transmat = transmat[num_states:]
# startprob = SHMM.model_initial.coef.T'''

############################################### Loading ####################################################

'''with open('../models/UnSupervisedIOHMM/model.json') as json_file:
    json_dict = json.load(json_file)
SHMM = UnSupervisedIOHMM.from_json(json_dict)

##

lengths = [df_A[df_A['unit'] == i].cycle.max() for i in range(1, df_A['unit'].max() + 1)]
state_sequences = []
for i in range(df_A['unit'].max()):
    for j in range(lengths[i]):
        state_sequences.append(np.argmax(np.exp(SHMM.log_gammas[i])[j]))
pred = [state_sequences[df[df['unit'] == i].index[0]:df[df['unit'] == i].index[-1] + 1] for i in
        range(1, df_A['unit'].max() + 1)]
failure_states = [pred[i][-1] for i in range(df_A['unit'].max())]

HMM_out = [np.exp(SHMM.log_gammas[i]) for i in range(len(SHMM.log_gammas))]

hierarchical = []
h = []
failure = np.unique(failure_states)
for i in range(len(pred)):
    h = []
    for j in range(len(failure)):
        h.append(np.where(np.array(pred[i]) == failure[j]))
    h = np.squeeze(np.sort(np.concatenate(h, 1)))
    hierarchical.append(h)'''

'''def log_mask_zero(a):
    """
    Compute the log of input probabilities masking divide by zero in log.

    Notes
    -----
    During the M-step of EM-algorithm, very small intermediate start
    or transition probabilities could be normalized to zero, causing a
    *RuntimeWarning: divide by zero encountered in log*.

    This function masks this unharmful warning.
    """
    a = np.asarray(a)
    with np.errstate(divide="ignore"):
        return np.log(a)


def _do_viterbi_pass(framelogprob):
    n_samples, n_components = framelogprob.shape
    state_sequence, logprob = _hmmc._viterbi(n_samples, n_components, log_mask_zero(startprob),
                                             log_mask_zero(transmat), framelogprob)
    return logprob, state_sequence


def _decode_viterbi(X):
    framelogprob = SHMM.log_gammas[0]
    return _do_viterbi_pass(framelogprob)


def decode(X, lengths=None):
    decoder = {"viterbi": _decode_viterbi}["viterbi"]
    logprob = 0
    sub_state_sequences = []
    for sub_X in split_X_lengths(X, lengths):
        # XXX decoder works on a single sample at a time!
        sub_logprob, sub_state_sequence = decoder(sub_X)
        logprob += sub_logprob
        sub_state_sequences.append(sub_state_sequence)
    return logprob, np.concatenate(sub_state_sequences)


def predict(X, lengths=None):
    """
    Find most likely state sequence corresponding to ``X``.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix of individual samples.
    lengths : array-like of integers, shape (n_sequences, ), optional
        Lengths of the individual sequences in ``X``. The sum of
        these should be ``n_samples``.

    Returns
    -------
    state_sequence : array, shape (n_samples, )
        Labels for each sample from ``X``.
    """
    _, state_sequence = decode(X, lengths)
    return state_sequence


state_seq = predict(df)'''

############################################################################################################
#                                            HYDRAULIC SYSTEM
# ##########################################################################################################

'''num_states = 5

############################################### Training ###################################################

df_X = minmax.fit_transform(df_X)
df_obs = PCA(n_components=17).fit_transform(df_X)
df_obs = pd.DataFrame(df_obs)
df_hmm = df_obs

outputs = [[m] for m in df_obs.head()]

SHMM = UnSupervisedIOHMM(num_states=num_states)
SHMM.set_models(model_emissions=[OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True),
                                 OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True),
                                 OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True),
                                 OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True),
                                 OLS(est_stderr=True)],
                model_transition=CrossEntropyMNL(solver='lbfgs'),
                model_initial=CrossEntropyMNL(solver='lbfgs'))
# We set operating conditions as the input covariate associate with the sensor output
SHMM.set_inputs(covariates_initial=[], covariates_transition=[],
                covariates_emissions=[[]] * len(outputs))
SHMM.set_outputs(outputs)
SHMM.set_data([df_hmm])
SHMM.train()

json_dict = SHMM.to_json('../models/HydraulicUnSupervisedIOHMM/' + str(num_states) + '/')
with open('../models/HydraulicUnSupervisedIOHMM/' + str(num_states) + '/model.json', 'w') as outfile:
    json.dump(json_dict, outfile, indent=4, sort_keys=True)

# transmat = np.empty((num_states, num_states))
# for i in range(num_states):
#     transmat = np.concatenate((transmat, np.exp(SHMM.model_transition[i].predict_log_proba(np.array([[]])))))
# transmat = transmat[num_states:]
# startprob = SHMM.model_initial.coef.T

############################################### Loading ####################################################

with open('../models/HydraulicUnSupervisedIOHMM/' + str(num_states) + '/model.json') as json_file:
    json_dict = json.load(json_file)
SHMM = UnSupervisedIOHMM.from_json(json_dict)

##

state_sequences = np.argmax(np.exp(SHMM.log_gammas[0]), 1)
pred = state_sequences
HMM_out = np.exp(SHMM.log_gammas)

plt.figure(1)
plt.plot(pred)
plt.xlabel('# Cycles')
plt.ylabel('HMM states')
plt.show()

plt.figure(2)
pca = PCA(n_components=2).fit_transform(df_X)
for class_value in range(num_states):
    row_ix = np.where(pred == class_value)
    plt.scatter(pca[row_ix, 0], pca[row_ix, 1])
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.legend(list(range(0, num_states)), title='HMM states')
plt.show()

df_Y['HMM_state_pred'] = pred

##
df_cond = df_Y.query("Cooler_condition==3 & \
                      Valve_condition!=73 & \
                      Internal_pump_leakage!=2 & \
                      Hydraulic_accumulator!=90 & \
                      stable_flag!=1")
states = df_cond['HMM_state_pred']

plt.figure()
plt.scatter(states.index, states)
plt.xlabel('# Cycles')
plt.ylabel('HMM states')
axes = plt.gca()
axes.yaxis.grid()
plt.legend(title='cooler condition = close to total failure')

##
df_cond = df_Y.query("Cooler_condition!=3 & \
                      Valve_condition==73 & \
                      Internal_pump_leakage!=2 & \
                      Hydraulic_accumulator!=90 & \
                      stable_flag!=1")
states = df_cond['HMM_state_pred']

plt.figure()
plt.scatter(states.index, states)
plt.xlabel('# Cycles')
plt.ylabel('HMM states')
axes = plt.gca()
axes.yaxis.grid()
plt.legend(title='Valve condition = close to total failure')

##
df_cond = df_Y.query("Cooler_condition!=3 & \
                      Valve_condition!=73 & \
                      Internal_pump_leakage==2 & \
                      Hydraulic_accumulator!=90 & \
                      stable_flag!=1")
states = df_cond['HMM_state_pred']

plt.figure()
plt.scatter(states.index, states)
plt.xlabel('# Cycles')
plt.ylabel('HMM states')
axes = plt.gca()
axes.yaxis.grid()
plt.legend(title='Internal pump leakage = severe leakage')

##
df_cond = df_Y.query("Cooler_condition!=3 & \
                      Valve_condition!=73 & \
                      Internal_pump_leakage!=2 & \
                      Hydraulic_accumulator==90 & \
                      stable_flag!=1")
states = df_cond['HMM_state_pred']

plt.figure()
plt.scatter(states.index, states)
plt.xlabel('# Cycles')
plt.ylabel('HMM states')
axes = plt.gca()
axes.yaxis.grid()
plt.legend(title='Hydraulic accumulator = close to total failure')

##
df_cond = df_Y.query("stable_flag==1")
states = df_cond['HMM_state_pred']

plt.figure()
plt.scatter(states.index, states)
plt.xlabel('# Cycles')
plt.ylabel('HMM states')
axes = plt.gca()
axes.yaxis.grid()
plt.legend(title='stable flag = static conditions might not have been reached yet')'''

'''def log_mask_zero(a):
    """
    Compute the log of input probabilities masking divide by zero in log.

    Notes
    -----
    During the M-step of EM-algorithm, very small intermediate start
    or transition probabilities could be normalized to zero, causing a
    *RuntimeWarning: divide by zero encountered in log*.

    This function masks this unharmful warning.
    """
    a = np.asarray(a)
    with np.errstate(divide="ignore"):
        return np.log(a)


def _do_viterbi_pass(framelogprob):
    n_samples, n_components = framelogprob.shape
    state_sequence, logprob = _hmmc._viterbi(n_samples, n_components, log_mask_zero(startprob),
                                             log_mask_zero(transmat), framelogprob)
    return logprob, state_sequence


def _decode_viterbi(X):
    framelogprob = SHMM.log_gammas[0]
    return _do_viterbi_pass(framelogprob)


def decode(X, lengths=None):
    decoder = {"viterbi": _decode_viterbi}["viterbi"]
    logprob = 0
    sub_state_sequences = []
    for sub_X in split_X_lengths(X, lengths):
        # XXX decoder works on a single sample at a time!
        sub_logprob, sub_state_sequence = decoder(sub_X)
        logprob += sub_logprob
        sub_state_sequences.append(sub_state_sequence)
    return logprob, np.concatenate(sub_state_sequences)


def predict(X, lengths=None):
    """
    Find most likely state sequence corresponding to ``X``.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix of individual samples.
    lengths : array-like of integers, shape (n_sequences, ), optional
        Lengths of the individual sequences in ``X``. The sum of
        these should be ``n_samples``.

    Returns
    -------
    state_sequence : array, shape (n_samples, )
        Labels for each sample from ``X``.
    """
    _, state_sequence = decode(X, lengths)
    return state_sequence


state_seq = predict(df)'''

############################################################################################################
#                                                YOKOGAWA
# ##########################################################################################################

'''num_states = 5

############################################### Training ###################################################

df_X = minmax.fit_transform(df_X)
df_obs = PCA(n_components=17).fit_transform(df_X)
df_obs = pd.DataFrame(df_obs)
df_hmm = df_obs

outputs = [[m] for m in df_obs.head()]

SHMM = UnSupervisedIOHMM(num_states=num_states)
SHMM.set_models(model_emissions=[OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True),
                                 OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True),
                                 OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True),
                                 OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True),
                                 OLS(est_stderr=True)],
                model_transition=CrossEntropyMNL(solver='lbfgs'),
                model_initial=CrossEntropyMNL(solver='lbfgs'))
# We set operating conditions as the input covariate associate with the sensor output
SHMM.set_inputs(covariates_initial=[], covariates_transition=[],
                covariates_emissions=[[]] * len(outputs))
SHMM.set_outputs(outputs)
SHMM.set_data([df_hmm])
SHMM.train()

json_dict = SHMM.to_json('../models/HydraulicUnSupervisedIOHMM/' + str(num_states) + '/')
with open('../models/HydraulicUnSupervisedIOHMM/' + str(num_states) + '/model.json', 'w') as outfile:
    json.dump(json_dict, outfile, indent=4, sort_keys=True)

# transmat = np.empty((num_states, num_states))
# for i in range(num_states):
#     transmat = np.concatenate((transmat, np.exp(SHMM.model_transition[i].predict_log_proba(np.array([[]])))))
# transmat = transmat[num_states:]
# startprob = SHMM.model_initial.coef.T

############################################### Loading ####################################################

with open('../models/HydraulicUnSupervisedIOHMM/' + str(num_states) + '/model.json') as json_file:
    json_dict = json.load(json_file)
SHMM = UnSupervisedIOHMM.from_json(json_dict)

##

state_sequences = np.argmax(np.exp(SHMM.log_gammas[0]), 1)
pred = state_sequences
HMM_out = np.exp(SHMM.log_gammas)

plt.figure(1)
plt.plot(pred)
plt.xlabel('# Cycles')
plt.ylabel('HMM states')
plt.show()

plt.figure(2)
pca = PCA(n_components=2).fit_transform(df_X)
for class_value in range(num_states):
    row_ix = np.where(pred == class_value)
    plt.scatter(pca[row_ix, 0], pca[row_ix, 1])
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.legend(list(range(0, num_states)), title='HMM states')
plt.show()

df_Y['HMM_state_pred'] = pred

##
df_cond = df_Y.query("Cooler_condition==3 & \
                      Valve_condition!=73 & \
                      Internal_pump_leakage!=2 & \
                      Hydraulic_accumulator!=90 & \
                      stable_flag!=1")
states = df_cond['HMM_state_pred']

plt.figure()
plt.scatter(states.index, states)
plt.xlabel('# Cycles')
plt.ylabel('HMM states')
axes = plt.gca()
axes.yaxis.grid()
plt.legend(title='cooler condition = close to total failure')

##
df_cond = df_Y.query("Cooler_condition!=3 & \
                      Valve_condition==73 & \
                      Internal_pump_leakage!=2 & \
                      Hydraulic_accumulator!=90 & \
                      stable_flag!=1")
states = df_cond['HMM_state_pred']

plt.figure()
plt.scatter(states.index, states)
plt.xlabel('# Cycles')
plt.ylabel('HMM states')
axes = plt.gca()
axes.yaxis.grid()
plt.legend(title='Valve condition = close to total failure')

##
df_cond = df_Y.query("Cooler_condition!=3 & \
                      Valve_condition!=73 & \
                      Internal_pump_leakage==2 & \
                      Hydraulic_accumulator!=90 & \
                      stable_flag!=1")
states = df_cond['HMM_state_pred']

plt.figure()
plt.scatter(states.index, states)
plt.xlabel('# Cycles')
plt.ylabel('HMM states')
axes = plt.gca()
axes.yaxis.grid()
plt.legend(title='Internal pump leakage = severe leakage')

##
df_cond = df_Y.query("Cooler_condition!=3 & \
                      Valve_condition!=73 & \
                      Internal_pump_leakage!=2 & \
                      Hydraulic_accumulator==90 & \
                      stable_flag!=1")
states = df_cond['HMM_state_pred']

plt.figure()
plt.scatter(states.index, states)
plt.xlabel('# Cycles')
plt.ylabel('HMM states')
axes = plt.gca()
axes.yaxis.grid()
plt.legend(title='Hydraulic accumulator = close to total failure')

##
df_cond = df_Y.query("stable_flag==1")
states = df_cond['HMM_state_pred']

plt.figure()
plt.scatter(states.index, states)
plt.xlabel('# Cycles')
plt.ylabel('HMM states')
axes = plt.gca()
axes.yaxis.grid()
plt.legend(title='stable flag = static conditions might not have been reached yet')'''

'''def log_mask_zero(a):
    """
    Compute the log of input probabilities masking divide by zero in log.

    Notes
    -----
    During the M-step of EM-algorithm, very small intermediate start
    or transition probabilities could be normalized to zero, causing a
    *RuntimeWarning: divide by zero encountered in log*.

    This function masks this unharmful warning.
    """
    a = np.asarray(a)
    with np.errstate(divide="ignore"):
        return np.log(a)


def _do_viterbi_pass(framelogprob):
    n_samples, n_components = framelogprob.shape
    state_sequence, logprob = _hmmc._viterbi(n_samples, n_components, log_mask_zero(startprob),
                                             log_mask_zero(transmat), framelogprob)
    return logprob, state_sequence


def _decode_viterbi(X):
    framelogprob = SHMM.log_gammas[0]
    return _do_viterbi_pass(framelogprob)


def decode(X, lengths=None):
    decoder = {"viterbi": _decode_viterbi}["viterbi"]
    logprob = 0
    sub_state_sequences = []
    for sub_X in split_X_lengths(X, lengths):
        # XXX decoder works on a single sample at a time!
        sub_logprob, sub_state_sequence = decoder(sub_X)
        logprob += sub_logprob
        sub_state_sequences.append(sub_state_sequence)
    return logprob, np.concatenate(sub_state_sequences)


def predict(X, lengths=None):
    """
    Find most likely state sequence corresponding to ``X``.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix of individual samples.
    lengths : array-like of integers, shape (n_sequences, ), optional
        Lengths of the individual sequences in ``X``. The sum of
        these should be ``n_samples``.

    Returns
    -------
    state_sequence : array, shape (n_samples, )
        Labels for each sample from ``X``.
    """
    _, state_sequence = decode(X, lengths)
    return state_sequence


state_seq = predict(df)'''

############################################################################################################
#                                                   TEP
# ##########################################################################################################

'''num_states = 5

############################################### Training ###################################################

df_X = minmax.fit_transform(df_X)
df_obs = PCA(n_components=17).fit_transform(df_X)
df_obs = pd.DataFrame(df_obs)
df_hmm = df_obs

outputs = [[m] for m in df_obs.head()]

SHMM = UnSupervisedIOHMM(num_states=num_states)
SHMM.set_models(model_emissions=[OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True),
                                 OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True),
                                 OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True),
                                 OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True), OLS(est_stderr=True),
                                 OLS(est_stderr=True)],
                model_transition=CrossEntropyMNL(solver='lbfgs'),
                model_initial=CrossEntropyMNL(solver='lbfgs'))
# We set operating conditions as the input covariate associate with the sensor output
SHMM.set_inputs(covariates_initial=[], covariates_transition=[],
                covariates_emissions=[[]] * len(outputs))
SHMM.set_outputs(outputs)
SHMM.set_data([df_hmm])
SHMM.train()

json_dict = SHMM.to_json('../models/HydraulicUnSupervisedIOHMM/' + str(num_states) + '/')
with open('../models/HydraulicUnSupervisedIOHMM/' + str(num_states) + '/model.json', 'w') as outfile:
    json.dump(json_dict, outfile, indent=4, sort_keys=True)

# transmat = np.empty((num_states, num_states))
# for i in range(num_states):
#     transmat = np.concatenate((transmat, np.exp(SHMM.model_transition[i].predict_log_proba(np.array([[]])))))
# transmat = transmat[num_states:]
# startprob = SHMM.model_initial.coef.T

############################################### Loading ####################################################

with open('../models/HydraulicUnSupervisedIOHMM/' + str(num_states) + '/model.json') as json_file:
    json_dict = json.load(json_file)
SHMM = UnSupervisedIOHMM.from_json(json_dict)

##

state_sequences = np.argmax(np.exp(SHMM.log_gammas[0]), 1)
pred = state_sequences
HMM_out = np.exp(SHMM.log_gammas)

plt.figure(1)
plt.plot(pred)
plt.xlabel('# Cycles')
plt.ylabel('HMM states')
plt.show()

plt.figure(2)
pca = PCA(n_components=2).fit_transform(df_X)
for class_value in range(num_states):
    row_ix = np.where(pred == class_value)
    plt.scatter(pca[row_ix, 0], pca[row_ix, 1])
plt.xlabel('PCA Feature 1')
plt.ylabel('PCA Feature 2')
plt.legend(list(range(0, num_states)), title='HMM states')
plt.show()

df_Y['HMM_state_pred'] = pred

##
df_cond = df_Y.query("Cooler_condition==3 & \
                      Valve_condition!=73 & \
                      Internal_pump_leakage!=2 & \
                      Hydraulic_accumulator!=90 & \
                      stable_flag!=1")
states = df_cond['HMM_state_pred']

plt.figure()
plt.scatter(states.index, states)
plt.xlabel('# Cycles')
plt.ylabel('HMM states')
axes = plt.gca()
axes.yaxis.grid()
plt.legend(title='cooler condition = close to total failure')

##
df_cond = df_Y.query("Cooler_condition!=3 & \
                      Valve_condition==73 & \
                      Internal_pump_leakage!=2 & \
                      Hydraulic_accumulator!=90 & \
                      stable_flag!=1")
states = df_cond['HMM_state_pred']

plt.figure()
plt.scatter(states.index, states)
plt.xlabel('# Cycles')
plt.ylabel('HMM states')
axes = plt.gca()
axes.yaxis.grid()
plt.legend(title='Valve condition = close to total failure')

##
df_cond = df_Y.query("Cooler_condition!=3 & \
                      Valve_condition!=73 & \
                      Internal_pump_leakage==2 & \
                      Hydraulic_accumulator!=90 & \
                      stable_flag!=1")
states = df_cond['HMM_state_pred']

plt.figure()
plt.scatter(states.index, states)
plt.xlabel('# Cycles')
plt.ylabel('HMM states')
axes = plt.gca()
axes.yaxis.grid()
plt.legend(title='Internal pump leakage = severe leakage')

##
df_cond = df_Y.query("Cooler_condition!=3 & \
                      Valve_condition!=73 & \
                      Internal_pump_leakage!=2 & \
                      Hydraulic_accumulator==90 & \
                      stable_flag!=1")
states = df_cond['HMM_state_pred']

plt.figure()
plt.scatter(states.index, states)
plt.xlabel('# Cycles')
plt.ylabel('HMM states')
axes = plt.gca()
axes.yaxis.grid()
plt.legend(title='Hydraulic accumulator = close to total failure')

##
df_cond = df_Y.query("stable_flag==1")
states = df_cond['HMM_state_pred']

plt.figure()
plt.scatter(states.index, states)
plt.xlabel('# Cycles')
plt.ylabel('HMM states')
axes = plt.gca()
axes.yaxis.grid()
plt.legend(title='stable flag = static conditions might not have been reached yet')'''

'''def log_mask_zero(a):
    """
    Compute the log of input probabilities masking divide by zero in log.

    Notes
    -----
    During the M-step of EM-algorithm, very small intermediate start
    or transition probabilities could be normalized to zero, causing a
    *RuntimeWarning: divide by zero encountered in log*.

    This function masks this unharmful warning.
    """
    a = np.asarray(a)
    with np.errstate(divide="ignore"):
        return np.log(a)


def _do_viterbi_pass(framelogprob):
    n_samples, n_components = framelogprob.shape
    state_sequence, logprob = _hmmc._viterbi(n_samples, n_components, log_mask_zero(startprob),
                                             log_mask_zero(transmat), framelogprob)
    return logprob, state_sequence


def _decode_viterbi(X):
    framelogprob = SHMM.log_gammas[0]
    return _do_viterbi_pass(framelogprob)


def decode(X, lengths=None):
    decoder = {"viterbi": _decode_viterbi}["viterbi"]
    logprob = 0
    sub_state_sequences = []
    for sub_X in split_X_lengths(X, lengths):
        # XXX decoder works on a single sample at a time!
        sub_logprob, sub_state_sequence = decoder(sub_X)
        logprob += sub_logprob
        sub_state_sequences.append(sub_state_sequence)
    return logprob, np.concatenate(sub_state_sequences)


def predict(X, lengths=None):
    """
    Find most likely state sequence corresponding to ``X``.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix of individual samples.
    lengths : array-like of integers, shape (n_sequences, ), optional
        Lengths of the individual sequences in ``X``. The sum of
        these should be ``n_samples``.

    Returns
    -------
    state_sequence : array, shape (n_samples, )
        Labels for each sample from ``X``.
    """
    _, state_sequence = decode(X, lengths)
    return state_sequence


state_seq = predict(df)'''

'''def compute_R(Y, Y_Thresh):
    R_values = []
    for y in Y:
        if np.all(y == Y_Thresh['Normal'].to_numpy()):
            R = 10
        elif (pd.Series(y).between(Y_Thresh['LO-Alarm'].tolist(), Y_Thresh['HI-Alarm'].tolist())).all():
            # R = -np.sum(abs(y - Y_Thresh['Normal'].to_numpy()))
            R = 0
        else:
            # R = -np.sum(np.square(y - Y_Thresh['Normal'].to_numpy()))
            R = -np.sum(abs(y - Y_Thresh['Normal'].to_numpy()))
        R_values.append(R)
    return R_values'''

############################################################################################################
# **********************************************************************************************************
#                                           ENVIRONMENT MODELING
# **********************************************************************************************************
# ##########################################################################################################

'''
policy = {}
policy_test = {}
'''

'''
class CustomEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Discrete(100)
        self.observation_space = gym.spaces.Box(low=np.array([0]), high=np.array([100]))
        self.reward = 0
        self.cycle = 0
        self.state = np.array([np.array(HI[self.cycle]).astype(np.float32)])
        self.done = False

    def step(self, action):
        if self.cycle > failure_state:
            self.state = np.array([np.array(HI[self.cycle]).astype(np.float32)])
            self.reward = reward_failure
            self.done = True
            print("|cycle reached failure state|:", self.cycle, "reward:", self.reward, '\n')
        elif self.cycle <= failure_state:
            if action == 0:
                print("|hold|:", self.cycle)
                if self.cycle == failure_state:
                    self.state = np.array([np.array(HI[self.cycle]).astype(np.float32)])
                    self.reward = reward_failure
                    self.done = True
                    print("|cycle reached failure state|:", self.cycle, "reward:", self.reward, '\n')
                else:
                    self.cycle += 1
                    if HI[self.cycle] > T:
                        self.reward = reward_hold
                        self.state = np.array([np.array(HI[self.cycle]).astype(np.float32)])
                        self.done = False
                        print("|system running|", "health:", HI[self.cycle], "reward:", self.reward, '\n')
                    elif HI[self.cycle] <= T:
                        self.reward = reward_failure
                        self.state = np.array([np.array(HI[self.cycle]).astype(np.float32)])
                        self.done = True
                        print("|system failed|", "health:", HI[self.cycle], "reward:", self.reward, '\n')
            elif action == 1:
                print("|replace|:", self.cycle, "health:", HI[self.cycle])
                self.reward = reward_replace / (self.cycle + z)
                self.cycle = 0
                self.state = np.array([np.array(HI[self.cycle]).astype(np.float32)])
                self.done = True
                print("reward:", self.reward, '\n')
        info = {}
        return self.state, self.reward, self.done, info

    def reset(self):
        self.cycle = 0
        self.state = np.array([np.array(HI[self.cycle]).astype(np.float32)])
        self.done = False
        return self.state
'''

'''# ******************************** HMM Out ***********************************

HMM_out.insert(0, [])
for i in range(1, len(HMM_out)):
    HMM_out[i] = np.insert(HMM_out[i], 0, np.zeros((1, 15)), axis=0)

# ******************************* IOHMM-DRL **********************************

hierarchical.insert(0, [])
for i in range(1, len(hierarchical)):
    hierarchical[i] = np.insert(hierarchical[i], 0, 1000, axis=0)'''

############################################################################################################
#                                             Maintenance
# ##########################################################################################################

'''c_f = -1000
c_r = -100
do_nothing = 0

class CustomEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, is_training=True, verbose=False, nn='dnn', input_data='raw', is_hierarchical=False):
        self.train = is_training
        self.verbose = verbose
        self.nn = nn
        self.input_data = input_data
        self.is_hierarchical = is_hierarchical
        if self.input_data is 'raw':
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(df_X.shape[1],))
        if self.input_data is 'hmm':
            self.observation_space = gym.spaces.Box(low=0, high=2, shape=(HMM_out[1].shape[1],))
        self.action_space = gym.spaces.Discrete(2)
        self.reward = 0
        self.cycle = 1
        self.done = False
        self.engine_unit = engine_unit
        self.engine_df_A = df_A[df_A['unit'] == self.engine_unit]
        if self.input_data is 'raw':
            self.X = df_X[self.engine_df_A.index[0]:self.engine_df_A.index[-1] + 1]
        if self.input_data is 'hmm':
            self.X = HMM_out[self.engine_unit]
        if self.nn is 'rnn':
            self.state = np.expand_dims(self.X[self.cycle], axis=0)
        if self.nn is 'dnn':
            self.state = self.X[self.cycle]
        self.failure_state = self.engine_df_A['cycle'].max() - 1

    def get_next_engine_data(self):
        self.engine_unit += 1
        if self.train:
            if self.engine_unit > int((df_A['unit'].max() * 0.8)):
                self.engine_unit = 1
        else:
            if self.engine_unit > df_A['unit'].max():
                self.engine_unit = int((df_A['unit'].max() * 0.8) + 1)
        if self.verbose:
            print("********|engine unit|********:", self.engine_unit)
        self.engine_df_A = df_A[df_A['unit'] == self.engine_unit]
        if self.input_data is 'raw':
            self.X = df_X[self.engine_df_A.index[0]:self.engine_df_A.index[-1] + 1]
        if self.input_data is 'hmm':
            self.X = HMM_out[self.engine_unit]
        self.failure_state = self.engine_df_A['cycle'].max() - 1
        return self.X

    def step(self, action):
        if action == 0:
            if self.verbose:
                print("|hold|:", self.cycle)
            if self.cycle == self.failure_state:
                self.reward = (c_r + c_f) / self.cycle
                if self.nn is 'rnn':
                    self.state = np.expand_dims(self.X[self.cycle], axis=0)
                if self.nn is 'dnn':
                    self.state = self.X[self.cycle]
                self.done = True
                if self.train:
                    policy[self.engine_unit] = {'unit': self.engine_unit,
                                                'failure_state': self.failure_state,
                                                'replace_state': self.cycle}
                else:
                    policy_test[self.engine_unit] = {'unit': self.engine_unit,
                                                     'failure_state': self.failure_state,
                                                     'replace_state': self.cycle,
                                                     'reward': self.reward}
                if self.verbose:
                    print("|cycle reached failure state|:", self.cycle, "reward:", self.reward, '\n')
            else:
                self.reward = do_nothing
                self.cycle += 1
                if self.is_hierarchical:
                    if self.cycle in hierarchical[self.engine_unit]:
                        if self.nn is 'rnn':
                            self.state = np.expand_dims(self.X[self.cycle], axis=0)
                        if self.nn is 'dnn':
                            self.state = self.X[self.cycle]
                        self.done = False
                        if self.verbose:
                            print("|system running|", "reward:", self.reward, '\n')
                    else:
                        pass
                else:
                    if self.nn is 'rnn':
                        self.state = np.expand_dims(self.X[self.cycle], axis=0)
                    if self.nn is 'dnn':
                        self.state = self.X[self.cycle]
                    self.done = False
                    if self.verbose:
                        print("|system running|", "reward:", self.reward, '\n')
        elif action == 1:
            if self.verbose:
                print("|replace|:", self.cycle)
            if self.cycle == self.failure_state:
                self.reward = (c_r + c_f) / self.cycle
            else:
                self.reward = c_r / (self.cycle + 0.1)
            if self.nn is 'rnn':
                self.state = np.expand_dims(self.X[self.cycle], axis=0)
            if self.nn is 'dnn':
                self.state = self.X[self.cycle]
            if self.train:
                policy[self.engine_unit] = {'unit': self.engine_unit,
                                            'failure_state': self.failure_state,
                                            'replace_state': self.cycle}
            else:
                policy_test[self.engine_unit] = {'unit': self.engine_unit,
                                                 'failure_state': self.failure_state,
                                                 'replace_state': self.cycle,
                                                 'reward': self.reward}
            self.done = True
        if self.verbose:
            print("reward:", self.reward, '\n')
        info = {}
        return self.state, self.reward, self.done, info

    def reset(self):
        self.X = self.get_next_engine_data()
        self.cycle = 1
        if self.nn is 'rnn':
            self.state = np.expand_dims(self.X[self.cycle], axis=0)
        if self.nn is 'dnn':
            self.state = self.X[self.cycle]
        self.done = False
        return self.state'''

############################################################################################################
#                                           Alarm Management
# ##########################################################################################################
import ast
import matlab
import matlab.engine

from itertools import chain, repeat, islice


def pad_infinite(iterable, padding=None):
    return chain(iterable, repeat(padding))


def pad(iterable, size, padding=None):
    return islice(pad_infinite(iterable, padding), size)


''' run  this command in Matlab '''
# matlab.engine.shareEngine()

# eng = matlab.engine.start_matlab()
name_eng = matlab.engine.find_matlab()
eng1 = matlab.engine.connect_matlab(name_eng[0])
# eng2 = matlab.engine.connect_matlab(name_eng[0])
# # writematrix(xmv,'C:\Users\abbas\Desktop\case_studies\AlarmManagement\temexd_mod\temexd_mod\XMV.xlsx')
# Y = pd.read_excel(r'C:\Users\abbas\Desktop\case_studies\AlarmManagement\temexd_mod\temexd_mod/SIMOUT.xlsx',
#                   header=None).drop([18], axis=1).to_numpy()
# U = pd.read_excel(r'C:\Users\abbas\Desktop\case_studies\AlarmManagement\temexd_mod\temexd_mod/XMV.xlsx',
#                   header=None).drop([4, 8, 11], axis=1).to_numpy()

# np.save(r'C:\Users\abbas\Desktop\case_studies\AlarmManagement\temexd_mod\SIMOUT.npy', Y)
# np.save(r'C:\Users\abbas\Desktop\case_studies\AlarmManagement\temexd_mod\XMV.npy', U)

Y_Thresh = pd.read_csv(r"C:\Users\abbas\Desktop\case_studies\Datasets\TEP\PV_thresh.csv", header=0, delimiter=';')
Y_Thresh = Y_Thresh[:40]
U_Thresh = pd.read_csv(r"C:\Users\abbas\Desktop\case_studies\Datasets\TEP\CV_thresh.csv", header=0, delimiter=',')

init_corr = [0.0]
eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/TE Plant/PID-DRPRL', 'sw', str(1), nargout=0)
eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/TE Plant/xmv_corr', 'Value', str(init_corr), nargout=0)


def get_reward(state):
    """if np.all(state == Y_Thresh['Normal'].to_numpy()):
        reward = 1
    elif (pd.Series(state).between(Y_Thresh['LO-Alarm'].tolist(), Y_Thresh['HI-Alarm'].tolist())).all():
        reward = 0.5
    else:
        if prev_reward <= -np.sum(abs(state - Y_Thresh['Normal'].to_numpy())):
            reward = 0.25
        elif prev_reward > -np.sum(abs(state - Y_Thresh['Normal'].to_numpy())):
            reward = -1
    return reward"""
    return -(abs(state - Y_Thresh['Normal'].to_numpy()[0]))


'''class CustomEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, is_training=False, is_pretraining=False):
        eng.set_param('tesys', 'SimulationCommand', 'start', 'SimulationCommand', 'pause', nargout=0)
        self.observation_space = gym.spaces.Box(low=np.array(Y_Thresh['LO-Alarm']),
                                                high=np.array(Y_Thresh['HI-Alarm']),
                                                shape=(Y.shape[1],))
        self.action_space = gym.spaces.Box(low=np.array(U_Thresh['LO-Alarm']),
                                           high=np.array(U_Thresh['HI-Alarm']),
                                           shape=(U.shape[1],))
        self.reward = 0
        self.T = 0
        self.done = False
        self.is_training = is_training
        self.is_pretraining = is_pretraining
        if is_pretraining:
            self.state = Y[self.T]
        if is_training:
            self.state = eng.python_matlab(str(init_control))

    def step(self, control, is_training=False, is_pretraining=False):
        self.is_training = is_training
        self.is_pretraining = is_pretraining
        if self.is_pretraining:
            self.T += 1
            if self.T >= Y.shape[0]:
                self.done = True
            else:
                self.state = Y[self.T]
            self.reward = get_reward(self.state)
        if self.is_training:
            eng.python_matlab(str(control))
            eng.set_param('tesys', 'SimulationCommand', 'continue', 'SimulationCommand', 'pause', nargout=0)
            if eng.get_param('tesys', 'SimulationStatus') == ('stopped' or 'terminating'):
                self.done = True
            else:
                self.state = eng.python_matlab(str(control))
                self.reward = get_reward(self.state)
        info = {}
        return [np.delete(np.array(self.state), 18)], self.reward, self.done, info

    def reset(self, is_training=False, is_pretraining=False):
        self.is_training = is_training
        self.is_pretraining = is_pretraining
        if self.is_pretraining:
            self.T = 0
            self.state = Y[self.T]
        if self.is_training:
            eng.set_param('tesys', 'SimulationCommand', 'start', 'SimulationCommand', 'pause', nargout=0)
            self.state = eng.TEP_PythonSimulink(str(init_control))
        self.done = False
        return [np.delete(np.array(self.state), 18)], self.done'''

upper_bound = 1.0
scale_factor = 100.0


class CustomEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, test=False, reset_orig=True, run_orig=True, sim_time=10):
        self.dist_stop = None
        self.dist_start = None
        self.reset_orig = reset_orig
        self.run_orig = run_orig
        self.num_alarms = []
        self.true_num_alarms = []
        self.control_action = []
        self.alarms = []
        self.alarms_seq = []
        self.true_alarms = []
        self.errorDRL = []
        self.error = []
        self.test = test
        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent', 'SimulationCommand', 'stop', nargout=0)
        # eng2.set_param('Copy_of_MultiLoop_mode1', 'SimulationCommand', 'stop', nargout=0)
        self.disturbance = [0, 0, 0, 0, 0, 0, 0, 0]
        self.disturbances = self.disturbance + ([0] * 20)
        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/Disturbances', 'Value', str(self.disturbances), nargout=0)
        # eng2.set_param('Copy_of_MultiLoop_mode1/Disturbances', 'Value', str(self.disturbances), nargout=0)
        self.sim_time = sim_time
        if self.test:
            self.sim_time = sim_time
        self.sim_step = 0.00
        self.step_size = 0.01
        self.step_step_size = 0.01
        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent', 'StopTime', str(self.sim_time), nargout=0)
        # eng2.set_param('Copy_of_MultiLoop_mode1', 'StopTime', str(self.sim_time), nargout=0)
        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent', 'EnablePauseTimes', 'on', nargout=0)
        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent', 'PauseTimes', str(self.sim_step), nargout=0)
        # eng2.set_param('Copy_of_MultiLoop_mode1', 'EnablePauseTimes', 'on', nargout=0)
        # eng2.set_param('Copy_of_MultiLoop_mode1', 'PauseTimes', str(self.sim_step), nargout=0)
        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/TE Plant/PID-DRPRL', 'sw', str(1), nargout=0)
        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/TE Plant/xmv_corr', 'Value', str(init_corr), nargout=0)
        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent', 'SimulationCommand', 'start', 'SimulationCommand', 'pause',
                       nargout=0)
        # eng2.set_param('Copy_of_MultiLoop_mode1', 'SimulationCommand', 'start', 'SimulationCommand', 'pause', nargout=0)
        # self.state = list(eng1.workspace['output']._data) + \
        #              (Y_Thresh['Normal'].to_numpy() - np.delete(eng1.workspace['output']._data, 18)).tolist() + \
        #              list(eng1.workspace['input']._data) + \
        #              ast.literal_eval(eng1.get_param('MultiLoop_mode1_DRL_SingleAgent/TE Plant/xmv_corr', 'Value'))
        self.process_history = 2
        self.state = []
        self.pv = []
        self.mv = []
        self.rp = []
        mv = 1
        pv = 1
        dev = 0
        self.feat = mv + pv + dev
        self.observation_space = gym.spaces.Box(low=-np.inf,
                                                high=np.inf,
                                                shape=(self.feat,))
        self.action_space = gym.spaces.Box(low=0,
                                           high=upper_bound,
                                           shape=(mv,))
        self.reward = 0
        self.c_f = -10000
        self.prev_reward = 0
        self.T = 0
        self.done = False
        self.max_episode_steps = int(self.sim_time / self.step_size)

    def step(self, control, disturbance=False, dist_mag=None):
        mu = control * scale_factor
        control = matlab.double(mu.tolist())
        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/TE Plant/xmv_corr', 'Value', str(control), nargout=0)

        self.sim_step += self.step_size
        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent', 'PauseTimes', str(round(self.sim_step, 2)), nargout=0)
        # if self.run_orig:
        #     eng2.set_param('Copy_of_MultiLoop_mode1', 'PauseTimes', str(round(self.sim_step, 2)), nargout=0)
        eng1.set_param("MultiLoop_mode1_DRL_SingleAgent", 'SimulationCommand', 'continue', nargout=0)
        # if self.run_orig:
        #     eng2.set_param("Copy_of_MultiLoop_mode1", 'SimulationCommand', 'continue', nargout=0)

        if disturbance:
            if self.sim_time * self.dist_start < self.sim_step < self.sim_time * self.dist_stop:
                disturbance = [0, 0, 0, 0, 0, dist_mag, 0, 0]
                disturbances = disturbance + ([0] * 20)
                eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/Disturbances', 'Value', str(disturbances), nargout=0)
                # if self.run_orig:
                #     eng2.set_param('Copy_of_MultiLoop_mode1/Disturbances', 'Value', str(disturbances), nargout=0)
            else:
                disturbance = [0, 0, 0, 0, 0, 0, 0, 0]
                disturbances = disturbance + ([0] * 20)
                eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/Disturbances', 'Value', str(disturbances), nargout=0)
                # if self.run_orig:
                #     eng2.set_param('Copy_of_MultiLoop_mode1/Disturbances', 'Value', str(disturbances), nargout=0)

        if self.sim_step >= self.sim_time:
            self.reward = 0
            self.done = True
            self.alarms.append(np.mean(self.num_alarms))
            self.alarms_seq.append(self.num_alarms)
            self.true_alarms.append(np.mean(self.true_num_alarms))
            self.run_orig = False
        elif eng1.get_param('MultiLoop_mode1_DRL_SingleAgent', 'SimulationStatus') == ('stopped' or 'terminating'):
            eng1.set_param("MultiLoop_mode1_DRL_SingleAgent", 'SimulationCommand', 'stop', nargout=0)
            self.reward = self.c_f
            self.done = True
            self.alarms.append(np.mean(self.num_alarms))
            self.alarms_seq.append(self.num_alarms)
            self.true_alarms.append(np.mean(self.true_num_alarms))
            self.run_orig = False
        else:
            self.mv = [eng1.workspace['drprl'] / 100.0]
            self.pv = [eng1.workspace['output']._data.tolist()[0]]
            curr_dev = [eng1.workspace['output']._data.tolist()[0] - Y_Thresh['Normal'].to_numpy()[0]]
            self.state = self.state[self.feat:] + self.mv + self.pv
            self.reward = get_reward((np.delete(eng1.workspace['output']._data, 18)).tolist()[0])

        # if self.run_orig:
        #     true_obs = eng2.workspace['copy_of_output']
        #     true_obs = np.delete(np.array(true_obs), 18)
        #     true_control_room = ~pd.Series(true_obs).between(Y_Thresh['LO-Alarm'].tolist(),
        #                                                      Y_Thresh['HI-Alarm'].tolist())
        #     true_num_alarms = sum(true_control_room)
        #     eng2.set_param('Copy_of_MultiLoop_mode1/TE Plant/num_alarms', 'Value', str(true_num_alarms), nargout=0)
        #     self.error.append(-(abs(
        #         (np.delete(eng2.workspace['copy_of_output']._data, 18)).tolist()[0] - Y_Thresh['Normal'].to_numpy()[
        #             0])))
        #     self.true_num_alarms.append(true_num_alarms)

        control_room = ~pd.Series(np.delete(eng1.workspace['output']._data, 18).tolist()).between(
            Y_Thresh['LO-Alarm'].tolist(), Y_Thresh['HI-Alarm'].tolist())
        num_alarms = sum(control_room)
        op_screen = list(pad(control_room[control_room].index.values, 38, 0))
        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/TE Plant/true_alarms', 'Value', str(op_screen), nargout=0)
        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/TE Plant/num_alarms', 'Value', str(num_alarms), nargout=0)
        self.num_alarms.append(num_alarms)
        self.control_action.append(control[0])
        self.errorDRL.append(
            -(abs((np.delete(eng1.workspace['output']._data, 18)).tolist()[0] - Y_Thresh['Normal'].to_numpy()[0])))
        info = {}
        return self.state, self.reward, self.done, info

    def reset(self):
        self.num_alarms = []
        self.true_num_alarms = []
        self.control_action = []
        self.state = []
        self.pv = []
        self.mv = []
        self.rp = []
        self.dist_start = np.random.uniform(0.0, 0.5)
        self.dist_stop = np.random.uniform(0.6, 1.0)

        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent', 'StopTime', str(self.sim_time), nargout=0)
        # if self.reset_orig:
        #     eng2.set_param('Copy_of_MultiLoop_mode1', 'StopTime', str(self.sim_time), nargout=0)
        self.disturbance = [0, 0, 0, 0, 0, 0, 0, 0]
        self.disturbances = self.disturbance + ([0] * 20)
        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/Disturbances', 'Value', str(self.disturbances), nargout=0)
        # if self.reset_orig:
        #     eng2.set_param('Copy_of_MultiLoop_mode1/Disturbances', 'Value', str(self.disturbances), nargout=0)
        self.sim_step = 0.00
        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent', 'PauseTimes', str(self.sim_step), nargout=0)
        # if self.reset_orig:
        #     eng2.set_param('Copy_of_MultiLoop_mode1', 'PauseTimes', str(self.sim_step), nargout=0)
        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/TE Plant/PID-DRPRL', 'sw', str(1), nargout=0)
        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/TE Plant/xmv_corr', 'Value', str(init_corr), nargout=0)
        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent', 'SimulationCommand', 'start', 'SimulationCommand', 'pause',
                       nargout=0)
        # if self.reset_orig:
        #     eng2.set_param('Copy_of_MultiLoop_mode1', 'SimulationCommand', 'start', 'SimulationCommand', 'pause',
        #                    nargout=0)

        for i in range(self.process_history):
            self.mv = [eng1.workspace['input']._data.tolist()[2] / 100.0]
            self.pv = [eng1.workspace['output']._data.tolist()[0]]
            curr_dev = [eng1.workspace['output']._data.tolist()[0] - Y_Thresh['Normal'].to_numpy()[0]]
            self.state = self.state + self.mv + self.pv

            eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/TE Plant/PID-DRPRL', 'sw', str(1), nargout=0)
            control = np.array([0.0])
            '''control = np.insert([control], 4, 0, axis=1)
            control = np.insert(control, 8, 0, axis=1)
            mu = np.insert(control, 11, 0, axis=1)'''
            mu = control
            control = matlab.double(mu.tolist())
            eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/TE Plant/xmv_corr', 'Value', str(control), nargout=0)

            self.sim_step += self.step_size
            eng1.set_param('MultiLoop_mode1_DRL_SingleAgent', 'PauseTimes', str(round(self.sim_step, 2)), nargout=0)
            # if self.reset_orig:
            #     eng2.set_param('Copy_of_MultiLoop_mode1', 'PauseTimes', str(round(self.sim_step, 2)), nargout=0)
            eng1.set_param("MultiLoop_mode1_DRL_SingleAgent", 'SimulationCommand', 'continue', nargout=0)
            # if self.reset_orig:
            #     eng2.set_param("Copy_of_MultiLoop_mode1", 'SimulationCommand', 'continue', nargout=0)

        self.done = False
        self.reset_orig = False
        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/TE Plant/PID-DRPRL', 'sw', str(0), nargout=0)
        return self.state, self.done


env = CustomEnv(sim_time=5)
eval_env = CustomEnv(test=True, reset_orig=True, run_orig=True, sim_time=5)

############################################################################################################
# **********************************************************************************************************
#                                   DEEP REINFORCEMENT LEARNING ARCHITECTURES
# **********************************************************************************************************
# ##########################################################################################################


############################################################################################################
#                                           Function Approximation
# ##########################################################################################################

'''observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()

featurizer = sklearn.pipeline.FeatureUnion([
    ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
    ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
    ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
    ("rbf4", RBFSampler(gamma=0.5, n_components=100))
])
featurizer.fit(scaler.fit_transform(observation_examples))


class FunctionApproximator:
    def __init__(self):

        self.models = []
        for i in range(env.action_space.n):
            model = SGDRegressor(learning_rate="constant")
            model.partial_fit([self.featurize_state(env.reset())], [0])
            self.models.append(model)

    def featurize_state(self, state):

        scaled = scaler.transform([state])
        features = featurizer.transform(scaled)
        return features[0]

    def predict(self, s, a=None):

        state_features = self.featurize_state(s)
        if not a:
            return np.array([m.predict([state_features])[0] for m in self.models])
        else:
            return self.models[a].predict([state_features])[0]

    def update(self, s, a, y):

        features = self.featurize_state(s)
        self.models[a].partial_fit([features], [y])


def make_epsilon_greedy_policy(estimator, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        q_values = estimator.predict(observation)
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A

    return policy_fn


def q_learning(env, estimator, num_episodes, discount_factor=0.6, epsilon=0.0, epsilon_decay=1.0):
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))

    for i_episode in range(num_episodes):

        policy = make_epsilon_greedy_policy(estimator, epsilon * epsilon_decay ** i_episode, env.action_space.n)
        state = env.reset()

        for t in itertools.count():

            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)

            next_state, reward, end, _ = env.step(action)

            stats.episode_rewards[i_episode] += reward
            stats.episode_lengths[i_episode] = t

            q_values_next = estimator.predict(next_state)
            td_target = reward + discount_factor * np.max(q_values_next)

            estimator.update(state, action, td_target)

            if i_episode % 10 == 0:
                print("\rStep {} @ Episode {}/{} ({})".format(t, i_episode + 1, num_episodes, reward))

            if end:
                break

            state = next_state

    return stats'''

############################################################################################################
# **********************************************************************************************************
#                                        Approximate (Deep) Q Learning
# ##########################################################################################################

'''
n_actions = env.action_space.n
state_dim = env.observation_space.shape

tf.reset_default_graph()
sess = tf.InteractiveSession()
keras.backend.set_session(sess)

############################################## DNN #########################################################

network = keras.models.Sequential()
network.add(keras.layers.InputLayer(state_dim))
# let's create a network for approximate q-learning following guidelines above
network.add(keras.layers.Dense(128, activation='relu'))
network.add(keras.layers.Dense(256, activation='relu'))
network.add(keras.layers.Dense(n_actions, activation='linear'))

############################################## RNN #########################################################

network = keras.models.Sequential()
network.add(keras.layers.LSTM(128, return_sequences=True, input_shape=[None, df_X[0].shape[0]]))
network.add(keras.layers.LSTM(256))
network.add(keras.layers.Dense(n_actions))

network.summary()


def get_action(state, epsilon=0):
    """
    sample actions with epsilon-greedy policy
    recap: with p = epsilon pick random action, else pick action with highest Q(s,a)
    """

    q_values = network.predict(state[None])[0]

    exploration = np.random.random()
    if exploration < epsilon:
        action = np.random.choice(n_actions, 1)[0]
    else:
        action = np.argmax(q_values)
    return action


states_ph = tf.placeholder('float32', shape=(None, None) + state_dim)
actions_ph = tf.placeholder('int32', shape=[None])
rewards_ph = tf.placeholder('float32', shape=[None])
next_states_ph = tf.placeholder('float32', shape=(None, None) + state_dim)
is_done_ph = tf.placeholder('bool', shape=[None])

# get q-values for all actions in current states
predicted_qvalues = network(states_ph)

# select q-values for chosen actions
predicted_qvalues_for_actions = tf.reduce_sum(predicted_qvalues * tf.one_hot(actions_ph, n_actions), axis=1)

gamma = 0.95
alpha = 1e-4

# compute q-values for all actions in next states
predicted_next_qvalues = network(next_states_ph)

# compute V*(next_states) using predicted next q-values
next_state_values = tf.reduce_max(predicted_next_qvalues, axis=1)

# compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
target_qvalues_for_actions = rewards_ph + gamma * next_state_values

# at the last state we shall use simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
target_qvalues_for_actions = tf.where(is_done_ph, rewards_ph, target_qvalues_for_actions)

# mean squared error loss to minimize
loss = (tf.stop_gradient(target_qvalues_for_actions) - predicted_qvalues_for_actions) ** 2
loss = tf.reduce_mean(loss)

# training function that resembles agent.update(state, action, reward, next_state) from tabular agent
train_step = tf.train.AdamOptimizer(alpha).minimize(loss)


def generate_session(t_max=500, epsilon=0, train=False):
    """play env with approximate q-learning agent and train it at the same time"""
    total_reward = 0
    s = env.reset()

    for t in range(t_max):
        a = get_action(s, epsilon=epsilon)
        next_s, r, done, _ = env.step(a)

        if train:
            sess.run(train_step, {
                states_ph: [s], actions_ph: [a], rewards_ph: [r],
                next_states_ph: [next_s], is_done_ph: [done]
            })

        total_reward += r
        s = next_s
        if done:
            break

    return total_reward'''

############################################################################################################
#                                           Double Deep Q Learning
# ##########################################################################################################

'''
class ReplayBuffer:
    def __init__(self, batch_size=32, size=50000):
        """
        batch_size (int): number of data points per batch
        size (int): size of replay buffer.
        """
        self.batch_size = batch_size
        self.memory = deque(maxlen=size)

    def remember(self, s_t, a_t, r_t, s_t_next, d_t):
        """
        s_t (np.ndarray double): state
        a_t (np.ndarray int): action
        r_t (np.ndarray double): reward
        d_t (np.ndarray float): done flag
        s_t_next (np.ndarray double): next state
        """
        self.memory.append((s_t, a_t, r_t, s_t_next, d_t))

    def sample(self):
        """
        random sampling of data from buffer
        """
        # if we don't have enough samples yet
        size = min(self.batch_size, len(self.memory))
        return random.sample(self.memory, size)


class VectorizedEnvWrapper(gym.Wrapper):
    def __init__(self, env, num_envs=1):
        """
        env (gym.Env): to make copies of
        num_envs (int): number of copies
        """
        super().__init__(env)
        self.num_envs = num_envs
        self.envs = [copy.deepcopy(env) for n in range(num_envs)]

    def reset(self):
        """
        Return and reset each environment
        """
        return np.asarray([env.reset() for env in self.envs])

    def step(self, actions):
        """
        Take a step in the environment and return the result.
        actions (np.ndarray int)
        """
        next_states, rewards, dones = [], [], []
        for env, action in zip(self.envs, actions):
            next_state, reward, done, _ = env.step(action)
            if done:
                next_states.append(env.reset())
            else:
                next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
        return np.asarray(next_states), np.asarray(rewards), \
                np.asarray(dones)


class DeepQLearner:
    def __init__(self, env,
                 alpha=0.001, gamma=0.95,
                 epsilon_i=1.0, epsilon_f=0.001, n_epsilon=0.1, nn='dnn', batch_size=8):
        """
        env (VectorizedEnvWrapper): the vectorized gym.Env
        alpha (float): learning rate
        gamma (float): discount factor
        epsilon_i (float): initial value for epsilon
        epsilon_f (float): final value for epsilon
        n_epsilon (float): proportion of timesteps over which to
                           decay epsilon from epsilon_i to
                           epsilon_f
        """
        self.num_envs = env.num_envs
        self.M = env.action_space.n  # number of actions
        self.N = env.observation_space.shape[0]  # dimensionality of state space
        self.epsilon_i = epsilon_i
        self.epsilon_f = epsilon_f
        self.n_epsilon = n_epsilon
        self.epsilon = epsilon_i
        self.gamma = gamma
        if nn is 'rnn':
            self.Q = torch.nn.Sequential(
                torch.nn.LSTM(self.N, 128),
                torch.nn.LSTM(128, 256),
                torch.nn.LSTM(256, 64),
                torch.nn.Linear(64, self.M)
            ).double()
        if nn is 'dnn':
            self.Q = torch.nn.Sequential(
                torch.nn.Linear(self.N, 128),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, self.M)
            ).double()
        self.Q_ = copy.deepcopy(self.Q)
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=alpha)

    def synchronize(self):
        """
        Used to make the parameters of Q_ match with Q.
        """
        self.Q_.load_state_dict(self.Q.state_dict())

    def act(self, s_t):
        """
        Epsilon-greedy policy.
        s_t (np.ndarray): the current state.
        """
        s_t = torch.as_tensor(s_t).double()
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.M, size=self.num_envs)
        else:
            with torch.no_grad():
                return np.argmax(self.Q(s_t).numpy(), axis=1)

    def decay_epsilon(self, n):
        """
        Epsilon decay.
        n (int): proportion of training complete
        """
        self.epsilon = max(
            self.epsilon_f,
            self.epsilon_i - (n / self.n_epsilon) * (self.epsilon_i - self.epsilon_f))
        return self.epsilon

    def update(self, s_t, a_t, r_t, s_t_next, d_t):
        """
        Learning step.
        s_t (np.ndarray double): state
        a_t (np.ndarray int): action
        r_t (np.ndarray double): reward
        d_t (np.ndarray float): done flag
        s_t_next (np.ndarray double): next state
        """

        # make sure everything is torch.Tensor and type-compatible with Q
        s_t = torch.as_tensor(s_t).double()
        a_t = torch.as_tensor(a_t).long()
        r_t = torch.as_tensor(r_t).double()
        s_t_next = torch.as_tensor(s_t_next).double()
        d_t = torch.as_tensor(d_t).double()

        # we don't want gradients when calculating the target y
        with torch.no_grad():
            # taking 0th element because torch.max returns both maximum
            # and argmax
            Q_next = torch.max(self.Q_(s_t_next), dim=1)[0]
            target = r_t + (1 - d_t) * self.gamma * Q_next

        # use advanced indexing on the return to get the predicted
        # Q values corresponding to the actions chosen in each environment.
        Q_pred = self.Q(s_t)[range(self.num_envs), a_t]
        loss = torch.mean((target - Q_pred) ** 2)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def train(env, agent, replay_buffer, T=20000, n_theta=100):
    """
    env (VectorizedEnvWrapper): vectorized gym.Env
    agent (DeepQLearner)
    buffer (ReplayBuffer)
    T (int): total number of training timesteps
    batch_size: number of
    """

    # for plotting
    returns = []
    episode_rewards = 0
    mean_return = []
    running_return = []

    s_t = env.reset()
    for t in range(T):
        # synchronize Q and Q_
        if t % n_theta == 5:
            agent.synchronize()

        a_t = agent.act(s_t)
        s_t_next, r_t, d_t = env.step(a_t)

        # store data into replay buffer
        replay_buffer.remember(s_t, a_t, r_t, s_t_next, d_t)
        s_t = s_t_next

        # learn by sampling from replay buffer
        for batch in replay_buffer.sample():
            agent.update(*batch)

        # for plotting
        episode_rewards += r_t
        for i in range(env.num_envs):
            if d_t[i]:
                returns.append(episode_rewards[i])
                running_return.append(episode_rewards[i])
                episode_rewards[i] = 0

        # epsilon decay
        epsilon = agent.decay_epsilon(t / T)

        if t % 100 == 0:
            mean_return.append(np.mean(returns))
            print('epoch: ', t, '\t mean_return: ', '{:.2f}'.format(round(np.average(returns), 3)), '\t epsilon: ',
                  epsilon)

            returns = []
    plot_returns(running_return)
    plt.plot(mean_return)
    plt.show()
    return agent


sns.set()


def plot_returns(returns, window=10):
    """
    Returns (iterable): list of returns over time
    window: window for rolling mean to smooth plotted curve
    """
    sns.lineplot(
        data=pd.DataFrame(returns).rolling(window=window).mean()[window - 1::window]
    )'''

############################################################################################################
#                                       Double Deep Q Learning (PER)
# ##########################################################################################################

'''import os
import random
import gym
import pylab
import numpy as np
from collections import deque
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Lambda, Add
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras import backend as K


class SumTree(object):
    data_pointer = 0

    # Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
    def __init__(self, capacity):
        # Number of leaf nodes (final nodes) that contains experiences
        self.capacity = capacity

        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema below
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)

        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)

    # Here we define function that will add our priority score in the sumtree leaf and add the experience in data:
    def add(self, priority, data):
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1

        # Update data frame
        self.data[self.data_pointer] = data

        # Update the leaf
        self.update(tree_index, priority)

        # Add 1 to data_pointer
        self.data_pointer += 1

        if self.data_pointer >= self.capacity:  # If we're above the capacity, we go back to first index (we overwrite)
            self.data_pointer = 0

    # Update the leaf priority score and propagate the change through tree
    def update(self, tree_index, priority):
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # then propagate the change through tree
        # this method is faster than the recursive loop in the reference code
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change

    # Here build a function to get a leaf from our tree. So we'll build a function to get the leaf_index,
    # priority value of that leaf and experience associated with that leaf index:
    def get_leaf(self, v):
        parent_index = 0

        # the while loop is faster than the method in the reference code
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If we reach bottom, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]

    @property
    def total_priority(self):
        return self.tree[0]  # Returns the root node


# Now we finished constructing our SumTree object, next we'll build a memory object.
class Memory(object):  # stored as ( state, action, reward, next_state ) in SumTree
    PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.4  # importance-sampling, from initial value increasing to 1

    PER_b_increment_per_sampling = 0.001

    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        # Making the tree
        self.tree = SumTree(capacity)

    # Next, we define a function to store a new experience in our tree.
    # Each new experience will have a score of max_prority (it will be then improved when we use this exp to train our DDQN).
    def store(self, experience):
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        # If the max priority = 0 we can't put priority = 0 since this experience will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)  # set the max priority for new priority

    # Now we create sample function, which will be used to pick batch from our tree memory, which will be used to train our model.
    # - First, we sample a minibatch of n size, the range [0, priority_total] into priority ranges.
    # - Then a value is uniformly sampled from each range.
    # - Then we search in the sumtree, for the experience where priority score correspond to sample values are retrieved from.
    def sample(self, n):
        # Create a minibatch array that will contains the minibatch
        minibatch = []

        b_idx = np.empty((n,), dtype=np.int32)

        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n  # priority segment

        for i in range(n):
            # A value is uniformly sample from each range
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            # Experience that correspond to each value is retrieved
            index, priority, data = self.tree.get_leaf(value)

            b_idx[i] = index

            minibatch.append([data[0], data[1], data[2], data[3], data[4]])

        return b_idx, minibatch

    # Update the priorities on the tree
    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


def OurModel(input_shape, action_space, dueling):
    X_input = Input(input_shape)
    X = X_input

    # 'Dense' is the basic form of a neural network layer
    # Input Layer of state size(4) and Hidden Layer with 512 nodes
    X = Dense(512, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform')(X)

    # Hidden layer with 256 nodes
    X = Dense(256, activation="relu", kernel_initializer='he_uniform')(X)

    # Hidden layer with 64 nodes
    X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

    if dueling:
        state_value = Dense(1, kernel_initializer='he_uniform')(X)
        state_value = Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(action_space,))(state_value)

        action_advantage = Dense(action_space, kernel_initializer='he_uniform')(X)
        action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(action_space,))(
            action_advantage)

        X = Add()([state_value, action_advantage])
    else:
        # Output Layer with # of actions: 2 nodes (left, right)
        X = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(X)

    model = Model(inputs=X_input, outputs=X, name='CartPole D3QN model')
    model.compile(loss="mean_squared_error", optimizer=RMSprop(lr=0.00025, rho=0.95, epsilon=0.01),
                  metrics=["accuracy"])

    model.summary()
    return model


class DQNAgent:
    def __init__(self, env):
        self.env = env
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n

        self.EPISODES = 5000
        memory_size = 1000
        self.MEMORY = Memory(memory_size)
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate

        # EXPLORATION HYPERPARAMETERS for epsilon and epsilon greedy strategy
        self.epsilon = 1.0  # exploration probability at start
        self.epsilon_min = 0.01  # minimum exploration probability
        self.epsilon_decay = 0.0005  # exponential decay rate for exploration prob

        self.batch_size = 32

        # defining model parameters
        self.ddqn = True  # use doudle deep q network
        self.Soft_Update = False  # use soft parameter update
        self.dueling = False  # use dealing netowrk
        self.epsilot_greedy = False  # use epsilon greedy strategy
        self.USE_PER = True

        self.TAU = 0.1  # target network soft update hyperparameter

        self.Save_Path = 'Models'
        if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
        self.scores, self.episodes, self.average = [], [], []

        self.Model_name = os.path.join(self.Save_Path, str(self.env) + "_e_greedy.h5")

        # create main model and target model
        self.model = OurModel(input_shape=(self.state_size,), action_space=self.action_size, dueling=self.dueling)
        self.target_model = OurModel(input_shape=(self.state_size,), action_space=self.action_size,
                                     dueling=self.dueling)

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        if not self.Soft_Update and self.ddqn:
            self.target_model.set_weights(self.model.get_weights())
            return
        if self.Soft_Update and self.ddqn:
            q_model_theta = self.model.get_weights()
            target_model_theta = self.target_model.get_weights()
            counter = 0
            for q_weight, target_weight in zip(q_model_theta, target_model_theta):
                target_weight = target_weight * (1 - self.TAU) + q_weight * self.TAU
                target_model_theta[counter] = target_weight
                counter += 1
            self.target_model.set_weights(target_model_theta)

    def remember(self, state, action, reward, next_state, done):
        experience = state, action, reward, next_state, done
        if self.USE_PER:
            self.MEMORY.store(experience)
        else:
            self.memory.append((experience))

    def act(self, state, decay_step):
        # EPSILON GREEDY STRATEGY
        if self.epsilot_greedy:
            # Here we'll use an improved version of our epsilon greedy strategy for Q-learning
            explore_probability = self.epsilon_min + (self.epsilon - self.epsilon_min) * np.exp(
                -self.epsilon_decay * decay_step)
        # OLD EPSILON STRATEGY
        else:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= (1 - self.epsilon_decay)
            explore_probability = self.epsilon

        if explore_probability > np.random.rand():
            # Make a random action (exploration)
            return random.randrange(self.action_size), explore_probability
        else:
            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            # Take the biggest Q value (= the best action)
            return np.argmax(self.model.predict(state)), explore_probability

    def replay(self):
        if self.USE_PER:
            tree_idx, minibatch = self.MEMORY.sample(self.batch_size)
        else:
            minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        state = np.zeros((self.batch_size, self.state_size))
        next_state = np.zeros((self.batch_size, self.state_size))
        action, reward, done = [], [], []

        # do this before prediction
        # for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for i in range(self.batch_size):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        # do batch prediction to save speed
        # predict Q-values for starting state using the main network
        target = self.model.predict(state)
        target_old = np.array(target)
        # predict best action in ending state using the main network
        target_next = self.model.predict(next_state)
        # predict Q-values for ending state using the target network
        target_val = self.target_model.predict(next_state)

        for i in range(len(minibatch)):
            # correction on the Q value for the action used
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                if self.ddqn:  # Double - DQN
                    # current Q Network selects the action
                    # a'_max = argmax_a' Q(s', a')
                    a = np.argmax(target_next[i])
                    # target Q Network evaluates the action
                    # Q_max = Q_target(s', a'_max)
                    target[i][action[i]] = reward[i] + self.gamma * (target_val[i][a])
                else:  # Standard - DQN
                    # DQN chooses the max Q value among next actions
                    # selection and evaluation of action is on the target Q Network
                    # Q_max = max_a' Q_target(s', a')
                    target[i][action[i]] = reward[i] + self.gamma * (np.amax(target_next[i]))

        if self.USE_PER:
            indices = np.arange(self.batch_size, dtype=np.int32)
            absolute_errors = np.abs(target_old[indices, np.array(action)] - target[indices, np.array(action)])
            # Update priority
            self.MEMORY.batch_update(tree_idx, absolute_errors)

        # Train the Neural Network with batches
        self.model.fit(state, target, batch_size=self.batch_size, verbose=0)

    def load(self, name):
        self.model = load_model(name)

    def save(self, name):
        self.model.save(name)

    pylab.figure(figsize=(18, 9))

    def PlotModel(self, score, episode):
        self.scores.append(score)
        self.episodes.append(episode)
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
        pylab.plot(self.episodes, self.average, 'r')
        pylab.plot(self.episodes, self.scores, 'b')
        pylab.ylabel('Score', fontsize=18)
        pylab.xlabel('Steps', fontsize=18)
        dqn = 'DQN_'
        softupdate = ''
        dueling = ''
        greedy = ''
        PER = ''
        if self.ddqn: dqn = 'DDQN_'
        if self.Soft_Update: softupdate = '_soft'
        if self.dueling: dueling = '_Dueling'
        if self.epsilot_greedy: greedy = '_Greedy'
        if self.USE_PER: PER = '_PER'
        try:
            pylab.savefig(dqn + str(self.env) + softupdate + dueling + greedy + PER + ".png")
        except OSError:
            pass

        return str(self.average[-1])[:5]

    def run(self):
        decay_step = 0
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                # self.env.render()
                decay_step += 1
                action, explore_probability = self.act(state, decay_step)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                if not done:
                    reward = reward
                else:
                    reward = -100
                self.remember(state, action, reward, next_state, done)
                state = next_state
                i += 1
                if done:
                    # every step update target model
                    self.update_target_model()

                    # every episode, plot the result
                    average = self.PlotModel(i, e)

                    print("episode: {}/{}, replacement: {}, e: {:.2}, average: {}".format(e, self.EPISODES, i,
                                                                                          explore_probability, average))
                    # print("Saving trained model to", self.Model_name)
                    break
                self.replay()
        self.env.close()

    def test(self, env):
        self.load(self.Model_name)
        self.env = env
        for e in range(self.EPISODES):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size])
            done = False
            i = 0
            while not done:
                self.env.render()
                action = np.argmax(self.model.predict(state))
                next_state, reward, done, _ = self.env.step(action)
                state = np.reshape(next_state, [1, self.state_size])
                i += 1
                if done:
                    print("episode: {}/{}, replacement: {}".format(e, self.EPISODES, i))
                    break'''

############################################################################################################
# **********************************************************************************************************
#                                              REINFORCE
# ##########################################################################################################


############################################################################################################
# **********************************************************************************************************
#                                              Actor-Critic
# ##########################################################################################################

##************************************** Advantage Actor Critic A2C ****************************************

'''import numpy as np
import torch
import gym
from torch import nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter


def mish(input):
    return input * torch.tanh(F.softplus(input))


class Mish(nn.Module):
    def __init__(self): super().__init__()

    def forward(self, input): return mish(input)


# helper function to convert numpy arrays to tensors
def t(x):
    x = np.array(x) if not isinstance(x, np.ndarray) else x
    return torch.from_numpy(x).float()


class extract_tensor(nn.Module):
    def forward(self, x):
        tensor, _ = x
        return tensor


class Actor(nn.Module):
    def __init__(self, state_dim, n_actions, activation=nn.Tanh):
        super().__init__()
        self.n_actions = n_actions
        self.model = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.LSTM(state_dim, 128, 8, bidirectional=True),
            extract_tensor(),
            nn.ReLU(),
            nn.LSTM(256, 64, 4, bidirectional=True),
            extract_tensor(),
            nn.ReLU(),
            nn.LSTM(128, 32, 2),
            extract_tensor(),
            nn.Linear(32, n_actions)
        )

        logstds_param = nn.Parameter(torch.full((n_actions,), 0.1))
        self.register_parameter("logstds", logstds_param)

    def forward(self, X):
        means = self.model(X)
        stds = torch.clamp(self.logstds.exp(), 1e-3, 50)

        return torch.distributions.Normal(means, stds)


## Critic module
class Critic(nn.Module):
    def __init__(self, state_dim, activation=nn.Tanh):
        super().__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.LSTM(state_dim, 128, 8, bidirectional=True),
            extract_tensor(),
            nn.ReLU(),
            nn.LSTM(256, 64, 4, bidirectional=True),
            extract_tensor(),
            nn.ReLU(),
            nn.LSTM(128, 32, 2),
            extract_tensor(),
            nn.Linear(32, 1),
        )

    def forward(self, X):
        return self.model(X)


def discounted_rewards(rewards, dones, gamma):
    ret = 0
    discounted = []
    for reward, done in zip(rewards[::-1], dones[::-1]):
        ret = reward + ret * gamma * (1 - done)
        discounted.append(ret)

    return discounted[::-1]


def process_memory(memory, gamma=0.95, discount_rewards=True, last_value=None):
    actions = []
    states = []
    next_states = []
    rewards = []
    dones = []

    for action, reward, state, next_state, done in memory:
        actions.append(action)
        rewards.append(reward)
        states.append(state)
        next_states.append(next_state)
        dones.append(done)

    if discount_rewards:
        if False and dones[-1] == 0:
            rewards = discounted_rewards(rewards + [last_value], dones + [0], gamma)[:-1]
        else:
            rewards = discounted_rewards(rewards, dones, gamma)

    actions = torch.squeeze(t(actions), 1)
    states = torch.squeeze(t(states), 1)
    next_states = torch.squeeze(t(next_states), 1)
    rewards = t(rewards).view(-1, 1)
    dones = t(dones).view(-1, 1)
    return actions, rewards, states, next_states, dones


def clip_grad_norm_(module, max_grad_norm):
    nn.utils.clip_grad_norm_([p for g in module.param_groups for p in g["params"]], max_grad_norm)


class A2CLearner:
    def __init__(self, actor, critic, gamma=0.95, entropy_beta=0,
                 actor_lr=1e-6, critic_lr=1e-5, max_grad_norm=0.5):
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.actor = actor
        self.critic = critic
        self.entropy_beta = entropy_beta
        self.actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
        self.critic_optim = torch.optim.Adam(critic.parameters(), lr=critic_lr)
        self.last_value = 0

    def learn(self, memory, steps, discount_rewards=True):
        actions, rewards, states, next_states, dones = process_memory(memory, self.gamma, discount_rewards,
                                                                      self.last_value)

        if discount_rewards:
            td_target = rewards
        else:
            td_target = rewards + self.gamma * critic(next_states) * (1 - dones)
        value = critic(states)
        advantage = td_target - value
        self.last_value = value

        # actor
        norm_dists = self.actor(states)
        logs_probs = norm_dists.log_prob(actions)
        entropy = norm_dists.entropy().mean()

        actor_loss = (-logs_probs * advantage.detach()).mean() - entropy * self.entropy_beta
        self.actor_optim.zero_grad()
        actor_loss.backward()

        clip_grad_norm_(self.actor_optim, self.max_grad_norm)
        writer.add_histogram("gradients/actor",
                             torch.cat([p.grad.view(-1) for p in self.actor.parameters()]), global_step=steps)
        writer.add_histogram("parameters/actor",
                             torch.cat([p.data.view(-1) for p in self.actor.parameters()]), global_step=steps)
        self.actor_optim.step()

        # critic
        critic_loss = F.mse_loss(td_target, value)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic_optim, self.max_grad_norm)
        writer.add_histogram("gradients/critic",
                             torch.cat([p.grad.view(-1) for p in self.critic.parameters()]), global_step=steps)
        writer.add_histogram("parameters/critic",
                             torch.cat([p.data.view(-1) for p in self.critic.parameters()]), global_step=steps)
        self.critic_optim.step()

        # reports
        writer.add_scalar("losses/log_probs", -logs_probs.mean(), global_step=steps)
        writer.add_scalar("losses/entropy", entropy, global_step=steps)
        writer.add_scalar("losses/entropy_beta", self.entropy_beta, global_step=steps)
        writer.add_scalar("losses/actor", actor_loss, global_step=steps)
        writer.add_scalar("losses/advantage", advantage.mean(), global_step=steps)
        writer.add_scalar("losses/critic", critic_loss, global_step=steps)


class Runner:
    def __init__(self, env):
        self.env = env
        self.state = None
        self.done = True
        self.steps = 0
        self.episode_reward = 0
        self.episode_rewards = []
        self.true_alarms_per_episode = []
        self.alarms_per_episode = []
        self.test_alarms = []
        self.true_alarm_seq = []
        self.alarm_seq = []
        self.control_seq = []

    def reset(self):
        self.done = False
        self.episode_reward = 0
        self.state, _ = self.env.reset()

    def run(self, max_steps, memory=None, test=False):
        if not memory: memory = []

        for i in range(max_steps):
            if self.done: self.reset()

            dists = actor(t(self.state))
            actions = dists.sample().detach().data.numpy()
            actions_clipped = np.clip(actions, self.env.action_space.low.min(), env.action_space.high.max())

            control = np.insert(actions_clipped, 4, 0, axis=1)
            control = np.insert(control, 8, 0, axis=1)
            control = np.insert(control, 11, 0, axis=1)
            control = matlab.double(control.tolist())

            next_state, reward, self.done, info = self.env.step(control)
            memory.append(([np.squeeze(actions)], reward, self.state, next_state, self.done))

            self.state = next_state
            self.steps += 1
            self.episode_reward += reward

            if self.done:
                self.episode_rewards.append(self.episode_reward)
                self.true_alarms_per_episode.append(np.mean(env.true_num_alarms))
                self.alarms_per_episode.append(np.mean(env.num_alarms))
                self.true_alarm_seq.append(env.true_num_alarms)
                self.alarm_seq.append(env.num_alarms)
                self.control_seq.append(env.control_action)
                runner.run_test()
                self.test_alarms.append(np.mean(env.num_alarms))
                if len(self.episode_rewards) % 1 == 0:
                    print("episode:", len(self.episode_rewards), "\t\t episode reward:", round(self.episode_rewards[-1]),
                          "\t\t alarms:", self.alarms_per_episode[-1], '\t\t test alarms:', np.mean(env.num_alarms))
                writer.add_scalar("episode_reward", self.episode_rewards[-1], global_step=self.steps)
        return memory

    def run_test(self):
        env.test = True
        self.reset()
        while env.test:
            dists = actor(t(self.state))
            actions = dists.sample().detach().data.numpy()
            actions_clipped = np.clip(actions, self.env.action_space.low.min(), env.action_space.high.max())
            control = np.insert(actions_clipped, 4, 0, axis=1)
            control = np.insert(control, 8, 0, axis=1)
            control = np.insert(control, 11, 0, axis=1)
            control = matlab.double(control.tolist())
            self.state, reward, self.done, info = self.env.step(control)
            if self.done: env.test = False
        return


writer = SummaryWriter("runs/mish_activation")'''

##************************** Twin Delayed Deep Deterministic Policy Gradients (TD3) ************************

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e5)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, env.process_history, env.feat))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, env.process_history, env.feat))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


class extract_tensor(nn.Module):
    def forward(self, x):
        tensor, _ = x
        return tensor


class keep_sequence(nn.Module):
    def forward(self, x):
        tensor, _ = x
        return tensor[:, -1, :]


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.LSTM(env.feat, 64, batch_first=True)
        self.l2 = nn.LSTM(64, 32, batch_first=True)
        self.l3 = nn.Linear(32, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = self.l1(state)[0]
        a = self.l2(a)[0][:, -1, :]
        af = self.l3(a)
        return self.max_action * torch.sigmoid(af)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # self.l1 = nn.LSTM(env.feat + action_dim, 64, batch_first=True)
        # self.l2 = nn.LSTM(64, 32, batch_first=True)
        # self.l3 = nn.Linear(32, action_dim)
        #
        # self.l4 = nn.LSTM(env.feat + action_dim, 64, batch_first=True)
        # self.l5 = nn.LSTM(64, 32, batch_first=True)
        # self.l6 = nn.Linear(32, action_dim)
        self.l1 = nn.Linear(env.process_history * state_dim + action_dim, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, 1)

        self.l4 = nn.Linear(env.process_history * state_dim + action_dim, 64)
        self.l5 = nn.Linear(64, 32)
        self.l6 = nn.Linear(32, 1)

    def forward(self, state, action):
        sa = torch.cat([torch.reshape(state, (-1, env.process_history * env.feat)), action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([torch.reshape(state, (-1, env.process_history * env.feat)), action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3(object):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            min_action,
            batch_size,
            replay_buffer,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2
    ):

        """def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.uniform_(m.weight, -0.003, 0.003)
                m.bias.data.fill_(0.001)"""

        self.max_action = max_action
        self.min_action = min_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise * max_action
        self.noise_clip = noise_clip * max_action
        self.policy_freq = policy_freq

        self.total_it = 0
        self.burn_in = 0
        self.warm_up = 10000
        self.burn_period = 10000
        self.actor_bc_loss_hist = []
        self.critic_bc_loss_hist = []

        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.bc_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-3)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-3, weight_decay=1e-2)
        self.actor_target = self.train_actor_bc(replay_buffer, batch_size=batch_size)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4, weight_decay=1e-2)
        self.critic_target = self.train_critic_bc(replay_buffer, batch_size=batch_size)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train_actor_bc(self, replay_buffer, batch_size=None):
        xmv = np.expand_dims(
            np.load(r"C:\Users\abbas\Desktop\case_studies\AlarmManagement\temexd_mod\XMV_RPL_big.npy")[:, 2] / 100.0, 1)
        xmeas = np.expand_dims(
            np.load(r"C:\Users\abbas\Desktop\case_studies\AlarmManagement\temexd_mod\XMEAS_RPL_big.npy")[:, 0], 1)

        error_signal = []
        for i in range(len(xmeas)):
            error_signal.append(xmeas[i] - Y_Thresh['Normal'].to_numpy()[0])
        error_signal = np.array(error_signal)

        samples = list()
        length = env.process_history
        n = len(xmeas)
        for i in range(0, n):
            if i == n - length:
                break
            sample = np.concatenate((xmv[i:i + length], xmeas[i:i + length]), 1)
            samples.append(sample)
        samples = np.array(samples)

        expert_traj = xmv[length:]

        for i, (state, action) in enumerate(zip(samples, expert_traj)):
            if i >= len(samples) - 1:
                break
            next_state = samples[i + 1]
            reward = get_reward(xmeas[length + i])
            done_bool = 0
            replay_buffer.add(state, action, next_state, reward, done_bool)

        print("----------------------- Behaviour Cloning -------------------------")
        for i in range(self.warm_up):
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
            bc_loss = F.mse_loss(self.actor(state), action)
            self.bc_optimizer.zero_grad()
            bc_loss.backward()
            self.bc_optimizer.step()
            self.actor_bc_loss_hist.append(bc_loss.item())
        print("\nactor_loss", round(bc_loss.item(), 5))
        return copy.deepcopy(self.actor)

    def train_critic_bc(self, replay_buffer, batch_size=None):
        for i in range(self.burn_period):
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            with torch.no_grad():
                noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                next_action = (self.actor_target(next_state) + noise).clamp(self.min_action, self.max_action)

                target_Q1, target_Q2 = self.critic(next_state, next_action)
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = torch.layer_norm(reward, reward.shape) + (not_done * self.discount * target_Q)

            current_Q1, current_Q2 = self.critic(state, action)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            self.critic_bc_loss_hist.append(critic_loss.item())
        print("\ncritic_loss", round(critic_loss.item(), 5))
        print("\n-------------------------------------------------------------------")
        return copy.deepcopy(self.critic)

    def train(self, replay_buffer, batch_size=None):
        self.total_it += 1
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(self.min_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = torch.layer_norm(reward, reward.shape) + (not_done * self.discount * target_Q)

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


'''    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)
        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)'''

############################################################################################################
# **********************************************************************************************************
#                                     Offline DRL (Behavioral Cloning)
# ##########################################################################################################

##************************************** Stable Baselines (Library) ****************************************

'''import gym
from stable_baselines import DQN
from stable_baselines.gail import generate_expert_traj

model = DQN('MlpPolicy', 'CartPole-v1', verbose=1)
# Train a DQN agent for 1e5 timesteps and generate 10 trajectories
# data will be saved in a numpy archive named `expert_cartpole.npz`
generate_expert_traj(model, 'expert_cartpole', n_timesteps=int(1e5), n_episodes=10)


env = gym.make("CartPole-v1")


# Here the expert is a random agent
# but it can be any python function, e.g. a PID controller
def dummy_expert(_obs):
    """
    Random agent. It samples actions randomly
    from the action space of the environment.

    :param _obs: (np.ndarray) Current observation
    :return: (np.ndarray) action taken by the expert
    """
    return env.action_space.sample()


# Data will be saved in a numpy archive named `expert_cartpole.npz`
# when using something different than an RL expert,
# you must pass the environment object explicitly
generate_expert_traj(dummy_expert, 'dummy_expert_cartpole', env, n_episodes=10)'''

'''from stable_baselines import PPO2
from stable_baselines.gail import ExpertDataset

# Using only one expert trajectory
# you can specify `traj_limitation=-1` for using the whole dataset
dataset = ExpertDataset(expert_path='expert_cartpole.npz',
                        traj_limitation=1, batch_size=128)

model = PPO2('MlpPolicy', 'CartPole-v1', verbose=1)
# Pretrain the PPO2 model
model.pretrain(dataset, n_epochs=1000)

# As an option, you can train the RL agent
# model.learn(int(1e5))

# Test the pre-trained model
env = model.get_env()
obs = env.reset()

reward_sum = 0.0
for _ in range(1000):
    action, _ = model.predict(obs)
    obs, reward, done, _ = env.step(action)
    reward_sum += reward
    env.render()
    if done:
        print(reward_sum)
        reward_sum = 0.0
        obs = env.reset()

env.close()'''

##******************************************** Open AI Gym *************************************************

'''n_actions = env.action_space.shape
state_dim = env.observation_space.shape


def create_model():
    """
    Creates the model.
    """
    state_ph = tf.placeholder(tf.float32, shape=[None, state_dim[0]])

    # # Hidden neurons
    with tf.variable_scope("layer1"):
        hidden = tf.layers.dense(state_ph, 128, activation=tf.nn.relu)
    with tf.variable_scope("layer2"):
        hidden = tf.layers.dense(hidden, 256, activation=tf.nn.relu)
    with tf.variable_scope("layer3"):
        hidden = tf.layers.dense(hidden, 128, activation=tf.nn.relu)
    with tf.variable_scope("layer4"):
        hidden = tf.layers.dense(hidden, 64, activation=tf.nn.relu)
    # Make output layers
    with tf.variable_scope("output"):
        action = tf.layers.dense(hidden, n_actions[0])

    return state_ph, action


def create_training(action):
    """
    Creates the model.
    """
    with tf.variable_scope("loss"):
        actions_ph = tf.placeholder(tf.float32, shape=[None, n_actions[0]])
        loss = tf.reduce_mean(tf.square(actions_ph - action))
        tf.summary.scalar('loss', loss)

    with tf.variable_scope("training"):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        train_op = optimizer.minimize(loss=loss)

    return train_op, loss, actions_ph


# Create the environment with specified arguments
state_data, action_data = Y, U

y, model = create_model()
train, loss, u = create_training(model)

saver = tf.train.Saver()
sess = tf.Session()

# Create summaries
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('./logs/' + str(time.time()), sess.graph)

##************************************************ Training ************************************************

sess.run(tf.global_variables_initializer())
rewards = []
tick = 0

for e in range(3):
    obs, done = env.reset()
    while not done:
        # Get a random batch from the data
        batch_index = np.random.choice(len(state_data), 64)  # Batch size
        state_batch, action_batch = state_data[batch_index], action_data[batch_index]

        # Train the model.
        _, cur_loss, cur_summaries = sess.run([train, loss, merged], feed_dict={y: state_data, u: action_data})

        print("Loss: {}".format(cur_loss))
        train_writer.add_summary(cur_summaries, tick)

        # Handle the toggling of different application states
        action = sess.run(model, feed_dict={y: [obs.flatten()]})[0]

        obs, reward, done, info = env.step(action)

        rewards.append(reward)
        tick += 1

    saver.save(sess, "./tf_ckpt/model")
    fig, ax = plt.subplots()

    ax.plot(AlarmCounts, color='red', label='Alarm Counts')
    ax.tick_params(axis='y', labelcolor='red')
    ax.legend(loc="upper left")

    ax2 = ax.twinx()
    ax2.plot(rewards[:-205], color='green', label='Rewards')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.legend(loc="upper right")

##************************************************* Loading ************************************************

# Restore the model.
saver.restore(sess, "./tf_ckpt/model")

##********************************************** Control Params ********************************************

control_params = []
for state_param in Y:
    control_params.append(sess.run(model, feed_dict={y: [state_param.flatten()]})[0])
control_params = np.array(control_params)'''

############################################################################################################
# **********************************************************************************************************
#                          Offline + Online DRL (Cycle of Learning - Actor Critic)
# ##########################################################################################################

'''from collections import deque
import random

tf.enable_eager_execution()

state_dim = env.observation_space.shape
n_actions = env.action_space.shape


class ReplayBuffer:
    def __init__(self, batch_size=None, size=None):
        self.batch_size = batch_size
        self.memory = deque(maxlen=size)

    def remember(self, s_t, a_t, r_t, s_t_next, d_t):
        self.memory.append((s_t, a_t, r_t, s_t_next, d_t))

    def sample(self, batch_size_agent):
        self.batch_size = batch_size_agent
        size = min(self.batch_size, len(self.memory))
        return random.sample(self.memory, size)


# ***********************************************************************************************

gamma = 0.98
alpha = 1e-5
beta = 1e-4
epsilon = 1.0
tau = 0.05

class Actor(tf.keras.Model):
    def __init__(self, name):
        super(Actor, self).__init__()
        self.net_name = name
        self.dense_0 = tf.keras.layers.Dense(64, activation='relu')
        self.dense_1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(128, activation='relu')
        self.policy = tf.keras.layers.Dense(n_actions[0],
                                            kernel_initializer=
                                            tf.keras.initializers.random_uniform(minval=-0.001, maxval=0.001),
                                            activation='tanh')

    def call(self, states_ph):
        layer_norm = tf.cast(tf.contrib.layers.layer_norm(states_ph), tf.float32)
        x = self.dense_0(layer_norm)
        policy = self.dense_1(x)
        policy = self.dense_2(policy)
        policy = self.policy(policy)
        return policy * upper_bound


class Critic(tf.keras.Model):
    def __init__(self, name):
        super(Critic, self).__init__()
        self.net_name = name
        self.dense_0 = tf.keras.layers.Dense(64, activation='relu')
        self.dense_1 = tf.keras.layers.Dense(128, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(128, activation='relu')
        self.q_value = tf.keras.layers.Dense(1, activation=None)

    def call(self, state, action):
        layer_norm = tf.cast(tf.contrib.layers.layer_norm(tf.concat([state, action], axis=1)), tf.float32)
        state_action_value = self.dense_0(layer_norm)
        state_action_value = self.dense_1(state_action_value)
        state_action_value = self.dense_2(state_action_value)
        q_value = self.q_value(state_action_value)
        return q_value


class Agent:
    def __init__(self):
        self.actor_lr = alpha
        self.critic_lr = beta

        self.actor = Actor(name='actor')
        self.critic = Critic(name='critic')
        self.target_actor = Actor(name='target_actor')
        self.target_critic = Critic(name='target_critic')

        self.actor.compile(optimizer=tf.train.AdamOptimizer(learning_rate=alpha))
        self.critic.compile(optimizer=tf.train.AdamOptimizer(learning_rate=beta))
        self.target_actor.compile(optimizer=tf.train.AdamOptimizer(learning_rate=alpha))
        self.target_critic.compile(optimizer=tf.train.AdamOptimizer(learning_rate=beta))

        actor_weights = self.actor.get_weights()
        critic_weights = self.critic.get_weights()

        self.target_actor.set_weights(actor_weights)
        self.target_critic.set_weights(critic_weights)

    def update_target_networks(self, tau):
        actor_weights = self.actor.weights
        target_actor_weights = self.target_actor.weights
        for index in range(len(actor_weights)):
            target_actor_weights[index] = tau * actor_weights[index] + (1 - tau) * target_actor_weights[index]

        self.target_actor.set_weights(target_actor_weights)

        critic_weights = self.critic.weights
        target_critic_weights = self.target_critic.weights

        for index in range(len(critic_weights)):
            target_critic_weights[index] = tau * critic_weights[index] + (1 - tau) * target_critic_weights[index]

        self.target_critic.set_weights(target_critic_weights)


agent = Agent()'''

# ******************************************************************************************************

'''def create_training(Q_value, rewards_ph, Q_value1_ph, done_ph):
    with tf.variable_scope("loss_a"):
        loss_a = -tf.reduce_mean(Q_value)

    with tf.variable_scope("loss_c"):
        # rewards_ph = tf.placeholder(tf.float32, shape=[None], name='rewards')
        # Q_value1_ph = tf.placeholder(tf.float32, shape=[None], name='total_returns')
        # done_ph = tf.placeholder(tf.float32, shape=[None], name='dones')
        loss_q = tf.reduce_mean(tf.square((rewards_ph + gamma * Q_value1_ph * (1 - done_ph)) - Q_value))

    with tf.variable_scope("train_a"):
        with tf.GradientTape as tape1:
            actor_gradient = tape1.gradients(loss_a, tf.trainable_variables('actor'))
        train_actor = tf.train.AdamOptimizer(learning_rate=alpha).\
            apply_gradients(zip(actor_gradient, tf.trainable_variables('actor')))

    with tf.variable_scope("train_c"):
        critic_gradient = tf.keras.backend.gradients(loss_q, tf.trainable_variables('critic'))
        train_critic = tf.train.AdamOptimizer(learning_rate=beta).\
            apply_gradients(zip(critic_gradient, tf.trainable_variables('critic')))

    return train_actor, train_critic, loss_a, loss_q, rewards_ph, Q_value1_ph, done_ph'''

'''s_t = tf.placeholder(tf.float32, shape=[None, state_dim[0]], name='state')
actor = create_actor(s_t=None, 'actor')
critic = create_critic(s_t=None, actor, 'critic')
target_actor = create_actor(s_t=None, 'target_actor')
target_critic = create_critic(s_t, target_actor, 'target_critic')
train_a, train_c, loss_a, loss_c, r_t, q_t1, d_t = create_training(critic)'''

############################################################################################################
# **********************************************************************************************************
#                                               TRAINING
# **********************************************************************************************************
# ##########################################################################################################

# Define and Train the agent
print('###################################################################')
print("                            Training                               ")
print('###################################################################', '\n')

############################################################################################################
# **********************************************************************************************************
#                                         Predictive Maintenance
# ##########################################################################################################

# *************************** Tabular Q Learning *****************************

'''q_table = np.zeros([env.observation_space.n, env.action_space.n])

# Hyperparameters
alpha = 0.01
gamma = 0.9
epsilon = 0.2

all_epochs = []
episode_reward = []
mean_episode_reward = []

for i in range(1, 1000000):
    state = env.reset()

    epochs, reward = 0, 0
    done = False
    total_reward = 0

    while not done:
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(q_table[state])  # Exploit learned values
            # action = np.random.choice(np.where(q_table[int(state)] == q_table[int(state)].max())[0])

        next_state, reward, done, info = env.step(action)

        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])

        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        state = next_state
        epochs += 1
        total_reward += reward

    episode_reward.append(total_reward)
    if i % 100 == 0:
        print(f"Episode: {i}")
        epsilon *= 0.999
        mean_episode_reward.append(np.mean(episode_reward))
        episode_reward = []

plt.plot(mean_episode_reward)'''

# ************************** Function Approximation **************************

'''estimator = FunctionApproximator()
stats = q_learning(env, estimator, 100, epsilon=0.1)

# plotting.plot_episode_stats(stats, smoothing_window=25)

'''

# ************************** Approximate Q Learning **************************

'''env = CustomEnv()
initial_epsilon = 0.5
epsilon_decay = 0.99
epsilon = initial_epsilon
total_returns = []
for i in range(1000):
    session_rewards = [generate_session(epsilon=epsilon, train=True) for _ in range(int(df_A['unit'].max() * 0.8))]
    print("epoch #{}\tmean reward = {:.3f}\tepsilon = {:.3f}".format(i, np.mean(session_rewards), epsilon))
    total_returns.append(np.mean(session_rewards))
    epsilon *= epsilon_decay
    # Make sure epsilon is always nonzero during training
    if epsilon <= 1e-4:
        break
plt.plot(total_returns)
plt.show()'''

# ************************ Double Deep Q Learning ****************************

'''env = CustomEnv(nn='dnn', input_data='raw', hierarchical=False)
envv = VectorizedEnvWrapper(env, num_envs=32)
replay_buffer = ReplayBuffer(batch_size=8)
agent = DeepQLearner(envv, alpha=1e-3, gamma=0.95, nn=env.nn, batch_size=replay_buffer.batch_size)
agent = train(envv, agent, replay_buffer, T=50000)'''

# ******************** Double Deep Q Learning (library) **********************

'''model = DQN(LnMlpPolicy, env, verbose=1, tensorboard_log="./dqn_nasa_tensorboard/").learn(total_timesteps=10000)
# tensorboard --logdir ./dqn_nasa_tensorboard/
model.save("deepq_nasa")'''

# ********************** Double Deep Q Learning (PER) ************************

'''env = CustomEnv(nn='dnn', input_data='raw', hierarchical=False)
agent = DQNAgent(env)
agent.run()'''

############################################################################################################
# **********************************************************************************************************
#                                          Alarm Management
# ##########################################################################################################

# ************************** Behavioral Cloning ******************************

'''sess.run(tf.global_variables_initializer())
rewards = []
tick = 0

for e in range(3):
    obs, done = env.reset()
    while not done:
        # Get a random batch from the data
        batch_index = np.random.choice(len(state_data), 64)  # Batch size
        state_batch, action_batch = state_data[batch_index], action_data[batch_index]

        # Train the model.
        _, cur_loss, cur_summaries = sess.run([train, loss, merged], feed_dict={y: state_batch, u: action_batch})

        print("Loss: {}".format(cur_loss))
        train_writer.add_summary(cur_summaries, tick)

        # Handle the toggling of different application states
        action = sess.run(model, feed_dict={y: [obs.flatten()]})[0]
        obs, reward, done, info = env.step(action)

        rewards.append(reward)
        tick += 1

    saver.save(sess, "./tf_ckpt/model")
    fig, ax = plt.subplots()

    ax.plot(AlarmCounts, color='red', label='Alarm Counts')
    ax.tick_params(axis='y', labelcolor='red')
    ax.legend(loc="upper left")

    ax2 = ax.twinx()
    ax2.plot(rewards[:-205], color='green', label='Rewards')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.legend(loc="upper right")

print("Training finished.\n")'''

# ******************** Cycle of Learning (Actor Critic) **********************

'''def memoryBufferCoL(batch_size=512, ratio=0.75):
    batch_size_agent = int(batch_size * ratio)
    batch_size_expert = batch_size - batch_size_agent

    batch_index = np.random.choice(len(expert_state_data), batch_size_expert)
    batch1_index = np.array([x + 1 for x in batch_index])
    state_batch_e, action_batch_e, rewards_batch_e, next_state_batch_e = expert_state_data[batch_index], \
                                                                         expert_action_data[batch_index], \
                                                                         np.array(expert_reward_data)[batch_index], \
                                                                         expert_next_state_data[batch1_index]

    replay = np.array(agent_replay_buffer.sample(batch_size_agent))
    s, a, r, s_, d = np.stack(replay[:, 0], 1)[0], np.stack(replay[:, 1]), np.stack(replay[:, 2]), \
                     np.stack(replay[:, 3], 1)[0], np.stack(replay[:, 4])

    states = np.concatenate((s, state_batch_e))
    actions = np.concatenate((a, action_batch_e))
    rewards = np.concatenate((r, rewards_batch_e))
    next_states = np.concatenate((s_, next_state_batch_e))

    return states, actions, rewards, next_states'''

"""agent_replay_buffer = ReplayBuffer(batch_size=512, size=5000)

total_returns_pretrain = []
total_returns = []

total_alarms_pretrain = []
total_alarms = []
copy_of_total_alarms = []

sim_time = []
alarms_seq = []

def TrainUpdate(epsilon=1.0):
    agent_rewards = []
    time_axis = []
    alarms = []
    copy_of_alarms = []
    corr_factor = []
    obs, done = env.reset()
    tout = 0
    while not done:
        '''replay = np.array(agent_replay_buffer.sample())
        s, a, r, s_, d = np.stack(replay[:, 0], 1)[0], np.stack(replay[:, 1]), np.stack(replay[:, 2])[0], \
                         np.stack(replay[:, 3], 1), np.stack(replay[:, 4])'''
        if len(agent_replay_buffer.memory) > 2 * agent_replay_buffer.batch_size:
            for i in range(5):
                replay = np.array(agent_replay_buffer.sample(512), dtype=object)
                s, a, r, s_, d = np.stack(replay[:, 0], 1)[0], np.stack(replay[:, 1]), np.stack(replay[:, 2]), \
                                 np.stack(replay[:, 3], 1)[0], np.stack(replay[:, 4])
                '''_, _, l_a, l_q = sess.run([train_a, train_c, loss_a, loss_c],
                                                 feed_dict={s_t: s, r_t: r, d_t: d,
                                                            q_t1: sess.run(target_critic,
                                                                           feed_dict={s_t: s_})[:, 0]})'''

                states = tf.convert_to_tensor(s, dtype=tf.float32)
                new_states = tf.convert_to_tensor(s_, dtype=tf.float32)
                rewards = tf.convert_to_tensor(r, dtype=tf.float32)
                actions = tf.convert_to_tensor(a, dtype=tf.float32)

                with tf.GradientTape() as tape1:
                    target_actions = tf.cast(agent.target_actor(new_states), tf.float32)
                    target_critic_values = tf.squeeze(agent.target_critic(new_states, target_actions), 1)
                    critic_value = tf.squeeze(agent.critic(states, actions), 1)
                    target = standard.fit_transform([rewards]) + gamma * target_critic_values * (1 - d)
                    critic_loss = tf.keras.losses.Huber()(target, critic_value)
                critic_gradient = tape1.gradient(critic_loss, agent.critic.trainable_variables)
                clipped_critic_gradient, _ = tf.clip_by_global_norm(critic_gradient, 0.5)
                agent.critic.optimizer.apply_gradients(zip(clipped_critic_gradient, agent.critic.trainable_variables))

                with tf.GradientTape() as tape2:
                    policy_actions = tf.cast(agent.actor(states), tf.float32)
                    actor_loss = -agent.critic(states, policy_actions)
                    actor_loss = tf.math.reduce_mean(actor_loss)
                actor_gradient = tape2.gradient(actor_loss, agent.actor.trainable_variables)
                clipped_actor_gradient, _ = tf.clip_by_global_norm(actor_gradient, 0.5)
                agent.actor.optimizer.apply_gradients(zip(clipped_actor_gradient,  agent.actor.trainable_variables))

                agent.update_target_networks(tau)
                print("Loss: {}".format(actor_loss + critic_loss))

        if env.sim_time * 0.25 < tout < env.sim_time * 0.8:
            disturbance = [0, 0, 0, 0, 0, 0, 0, 1]
            disturbances = disturbance + ([0] * 20)
            eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/Disturbances', 'Value', str(disturbances), nargout=0)
            eng2.set_param('Copy_of_MultiLoop_mode1/Disturbances', 'Value', str(disturbances), nargout=0)
        else:
            disturbance = [0, 0, 0, 0, 0, 0, 0, 0]
            disturbances = disturbance + ([0] * 20)
            eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/Disturbances', 'Value', str(disturbances), nargout=0)
            eng2.set_param('Copy_of_MultiLoop_mode1/Disturbances', 'Value', str(disturbances), nargout=0)

        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            noise = np.random.normal(scale=1, size=n_actions) if epsilon > 0 else 0
            action = agent.actor(np.array(obs)) + noise
            action = np.clip(action, env.action_space.low, env.action_space.high)[0]
        control = np.insert(action, 4, 0, axis=0)
        control = np.insert(control, 8, 0, axis=0)
        control = np.insert(control, 11, 0, axis=0)
        control = matlab.double(control.tolist())

        next_obs, reward, done, tout, info = env.step(control)
        agent_replay_buffer.remember(obs, action, reward, next_obs, done)

        agent_rewards.append(reward)
        if not done:
            eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/TE Plant/state_rewards', 'Value', str(reward), nargout=0)

        control_room = ~pd.Series(obs[0][:40]).between(Y_Thresh['LO-Alarm'].tolist(), Y_Thresh['HI-Alarm'].tolist())
        num_alarms = sum(control_room)
        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/TE Plant/num_alarms', 'Value', str(num_alarms), nargout=0)
        true_alarms = list(pad(control_room[control_room].index.values, 38, 0))
        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/TE Plant/true_alarms', 'Value', str(true_alarms), nargout=0)

        copy_of_obs = eng2.workspace['copy_of_output']
        copy_of_obs = [np.delete(np.array(copy_of_obs), 18)]
        copy_of_control_room = ~pd.Series(copy_of_obs[0]).between(Y_Thresh['LO-Alarm'].tolist(),
                                                                  Y_Thresh['HI-Alarm'].tolist())
        copy_of_num_alarms = sum(copy_of_control_room)
        eng2.set_param('Copy_of_MultiLoop_mode1/TE Plant/num_alarms', 'Value', str(copy_of_num_alarms), nargout=0)
        copy_of_true_alarms = list(pad(copy_of_control_room[copy_of_control_room].index.values, 38, 0))
        eng2.set_param('Copy_of_MultiLoop_mode1/TE Plant/true_alarms', 'Value', str(copy_of_true_alarms), nargout=0)

        obs = next_obs
        corr_factor.append(action)
        alarms.append(num_alarms)
        copy_of_alarms.append(copy_of_num_alarms)
        time_axis.append(tout)
    total_returns.append(np.mean(agent_rewards[:-1]))
    total_alarms.append(np.mean(alarms))
    alarms_seq.append(alarms)
    pid_alarms_seq = copy_of_alarms.copy()
    copy_of_total_alarms.append(np.mean(copy_of_alarms))
    sim_time.append(tout)
# plt.plot(time_axis[:250], alarms_seq[])
# plt.plot(time_axis[:250], pid_alarms_seq)


for epochs in range(10000):
    if epsilon > 0:
        epsilon -= 0.01
        noise = np.random.normal(scale=1, size=n_actions)
    else:
        epsilon = 0
        noise = 0
    TrainUpdate(epsilon=epsilon)"""

# ********************************** A2C *************************************

'''state_dim = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]
actor = Actor(state_dim, n_actions, activation=Mish)
critic = Critic(state_dim, activation=Mish)

learner = A2CLearner(actor, critic)
runner = Runner(env)

steps_on_memory = 25
episodes = 50000
episode_length = 100
total_steps = (episode_length * episodes) // steps_on_memory

for i in range(total_steps):
    memory = runner.run(steps_on_memory)
    learner.learn(memory, runner.steps, discount_rewards=False)

alarms = runner.alarms_per_episode
plt.plot(pd.DataFrame(runner.episode_rewards).rolling(50).mean())
plt.plot(pd.DataFrame(runner.alarms_per_episode).rolling(50).mean())'''

# ********************************** TD3 *************************************
eval_reward = []
def eval_policy(policy, eval_episodes=1, dist_mag=None):
    eval_env.reset_orig = True
    eval_env.run_orig = True
    eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/TE Plant/Scope 5', 'commented', 'on', nargout=0)
    eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/TE Plant/Scope 6', 'commented', 'on', nargout=0)
    avg_reward = 0.
    for i in range(eval_episodes):
        eval_DRL_action = []
        state, done = eval_env.reset()
        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/TE Plant/PID-DRPRL', 'sw', str(0), nargout=0)
        disturbance = True
        eval_timesteps = 0
        while not done:
            eval_timesteps += 1
            PID = eng1.workspace['input']._data[2] / 100.0
            DRPRL = policy.select_action(np.reshape(state, (1, env.process_history, env.feat)))
            action_mix = DRPRL + PID
            action = DRPRL
            eval_DRL_action.append(action)
            state, reward, done, _ = eval_env.step(action_mix, disturbance=disturbance, dist_mag=dist_mag)
            if eval_timesteps > 70:
                avg_reward += reward
    avg_reward /= eval_episodes
    eval_reward.append(avg_reward)
    print("--------------------------------------------------------------------")
    print("eval_reward:", round(avg_reward, 2), "\t\t\t\t eval_alarms:", round(eval_env.alarms[-1], 2))
    print("--------------------------------------------------------------------")
    eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/TE Plant/Scope 5', 'commented', 'on', nargout=0)
    eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/TE Plant/Scope 6', 'commented', 'on', nargout=0)
    return round(avg_reward, 2), round(eval_env.alarms[-1], 2)


state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
min_action = float(env.action_space.low[0])

total_warmup = 1e5
expl_noise = 0.01
batch_size = 128
epsilon_action = 0.2
alpha_action = 0.8
beta_mix = 0.5
gamma_mix = 0.3
replay_buffer = ReplayBuffer(state_dim, action_dim)
policy = TD3(state_dim,
             action_dim,
             max_action,
             min_action,
             batch_size,
             replay_buffer,
             discount=0.99,
             tau=0.005,
             policy_noise=0.02,
             noise_clip=0.05,
             policy_freq=2)

state, done = env.reset()
episode_reward = 0
episode_timesteps = 0
episode_num = 0
episode_rewards = []
c = 0
for t in range(int(env.max_episode_steps * 1e7)):
    episode_timesteps += 1
    disturbance = True

    z = np.random.uniform(0, 1)
    if z < epsilon_action:
        action_mix = np.array([np.random.uniform(0, 1)])
        action = action_mix
        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/TE Plant/PID-DRPRL', 'sw', str(0), nargout=0)
    else:
        PID = eng1.workspace['input']._data[2] / 100
        DRPRL = (policy.select_action(np.reshape(state, (1, env.process_history, env.feat))) +
                 np.random.normal(0, max_action * expl_noise, size=action_dim)).clip(min_action, max_action)
        action_mix = DRPRL + PID
        action = DRPRL
        eng1.set_param('MultiLoop_mode1_DRL_SingleAgent/TE Plant/PID-DRPRL', 'sw', str(0), nargout=0)

    next_state, reward, done, _ = env.step(action_mix, disturbance=disturbance, dist_mag=0.65)
    done_bool = float(done) if episode_timesteps < env.max_episode_steps else 0

    replay_buffer.add(np.reshape(state, (env.process_history, env.feat)), action,
                      np.reshape(next_state, (env.process_history, env.feat)), reward, done_bool)

    state = next_state
    if episode_timesteps > 70:
        episode_reward += reward

    policy.train(replay_buffer, batch_size=batch_size)

    if done:
        episode_rewards.append(episode_reward)
        print("episode:", episode_num,
              "\t\t episode reward:", round(episode_rewards[-1], 2),
              "\t\t alarms:", round(env.alarms[-1], 2))
        if episode_num % 5 == 0:
            evaluations = eval_policy(policy, dist_mag=0.65, eval_episodes=1)
            if evaluations[1] == 0.0:
                pass
        episode_reward = 0
        episode_timesteps = 0
        episode_num += 1
        state, done = env.reset()

plt.plot(pd.DataFrame(episode_rewards).rolling(50).mean())
plt.plot(pd.DataFrame(env.alarms).rolling(50).mean())

# np.save(r'C:\Users\abbas\Desktop\case_studies\AlarmManagement\temexd_mod\temexd_mod/DRPRL_DRL_0.65.npy', np.array(eval_env.control_action).reshape(1,-1)[0])
# np.save(r'C:\Users\abbas\Desktop\case_studies\AlarmManagement\temexd_mod\temexd_mod/REWARD_DRL_0.65.npy', pd.read_excel(r'C:\Users\abbas\Desktop\case_studies\AlarmManagement\temexd_mod\DRL_0.65.xlsx', header=None).to_numpy()[:,0])

print("Training finished.\n")

############################################################################################################
# **********************************************************************************************************
#                                              EVALUATION
# **********************************************************************************************************
# ##########################################################################################################

print('###################################################################################################')
print("Testing")
print('###################################################################################################', '\n')

############################################################################################################
# **********************************************************************************************************
#                                         Predictive Maintenance
# ##########################################################################################################

# *************************** Tabular Q Learning *****************************

'''"""Evaluate agent's performance after Q-learning"""

total_epochs = 0
episodes = 100

for _ in range(episodes):
    state = env.reset()
    epochs, reward = 0, 0

    done = False

    while not done:
        action = np.argmax(q_table[int(state)])
        state, reward, done, info = env.step(action)
        epochs += 1
    total_epochs += epochs

print(f"Results after {episodes} episodes:")
print(f"Average timesteps per episode: {total_epochs / episodes}")'''

# ************************** Function Approximation **************************

'''state = env.reset()
while True:
    q_values = estimator.predict(state)
    best_action = np.argmax(q_values)
    next_state, reward, done, _ = env.step(best_action)
    if done:
        break
    state = next_state'''

# ************************** Approximate Q Learning **************************

'''total_returns = []
for i in range(10):
    session_rewards = [generate_session(epsilon=0, train=False) for _ in range(100)]
    total_returns.append(np.mean(session_rewards))
env.close()
plt.plot(total_returns)
plt.show()'''

'''df_test = pd.read_csv(dir_path + '/CMAPSSData/test_FD001.txt', sep=" ", header=None, skipinitialspace=True).dropna(
    axis=1)
df = df_test.rename(columns={0: 'unit', 1: 'cycle', 2: 'W1', 3: 'W2', 4: 'W3'})
df_A = df[df.columns[[0, 1]]]
df_W = df[df.columns[[2, 3, 4]]]
df_X = df[df.columns[list(range(5, 26))]]'''

'''engine_unit = int((df_A['unit'].max() * 0.8) + 1)
env = CustomEnv(is_training=False)

session_rewards = [generate_session(epsilon=0, train=False) for _ in range(int(df_A['unit'].max() * 0.2))]
total_returns_test = np.mean(session_rewards)
print("mean reward = ", total_returns_test)

policy = pd.DataFrame.from_dict(policy).T
policy['remaining_cycles'] = policy['failure_state'] - policy['replace_state']
policy_test = pd.DataFrame.from_dict(policy_test).T
policy_test['remaining_cycles'] = policy_test['failure_state'] - policy_test['replace_state']

M = int(df_A['unit'].max() * 0.2)
IMC = (M * c_r) / np.sum(policy_test['failure_state'] - 1)
CMC = (M * (c_r + c_f)) / np.sum(policy_test['failure_state'])
print("Average remaining cycles", np.average(policy_test['remaining_cycles']))

np.savetxt(r'./output/sensor_data_train_HMM' + ' ' + dataset + ' ' + str(IMC) + ' ' + str(CMC) + ' ' + str(alpha) + ' '
           + str(gamma) + ' ' + str(initial_epsilon) + ' ' + str(epsilon_decay) + ' ' + str(c_r) + ' ' + str(c_f) + ' '
           + str(do_nothing) + ' ' + str(network.count_params()) + ' ' + str(num_states) + ' '
           + str(np.round(np.mean(total_returns_test), 2))
           + '.txt', policy.values, fmt='%d')
np.savetxt(r'./output/sensor_data_test_HMM' + ' ' + dataset + ' ' + str(IMC) + ' ' + str(CMC) + ' ' + str(alpha) + ' '
           + str(gamma) + ' ' + str(initial_epsilon) + ' ' + str(epsilon_decay) + ' ' + str(c_r) + ' ' + str(c_f) + ' '
           + str(do_nothing) + ' ' + str(network.count_params()) + ' ' + str(num_states) + ' '
           + str(np.round(np.mean(total_returns_test), 2))
           + '.txt', policy_test.values, fmt='%d')

env.close()'''

# ************************** Double Deep Q Learning **************************

'''engine_unit = int((df_A['unit'].max() * 0.8) + 1)
env = CustomEnv(is_training=False, nn='dnn', input_data='raw', hierarchical=False)
envv = VectorizedEnvWrapper(env, num_envs=1)
episodes = int(df_A['unit'].max() * 0.2)
episode_return = []
s_t = envv.reset()

for _ in range(episodes):
    d_t = False
    r_t = 0
    total_return = []
    while not d_t:
        a_t = agent.act(s_t)
        s_t_next, r_t, d_t = envv.step(a_t)
        s_t = s_t_next
        total_return.append(r_t)
        # store data into replay buffer
        # replay_buffer.remember(s_t, a_t, r_t, s_t_next, d_t)
        # s_t = s_t_next
        # learn by sampling from replay buffer
        # for batch in replay_buffer.sample():
        # agent.update(*batch)

policy = pd.DataFrame.from_dict(policy).T
policy['remaining_cycles'] = policy['failure_state'] - policy['replace_state']
policy_test = pd.DataFrame.from_dict(policy_test).T
policy_test['remaining_cycles'] = policy_test['failure_state'] - policy_test['replace_state']

M = int(df_A['unit'].max() * 0.2)
IMC = (M * c_r) / np.sum(policy_test['failure_state'] - 1)
CMC = (M * (c_r + c_f)) / np.sum(policy_test['failure_state'])
print("mean reward = ", np.mean(policy_test['reward']))
print("Average remaining cycles", np.average(policy_test['remaining_cycles']))'''

# ******************** Double Deep Q Learning (library) **********************

'''env.reset()
for obs in HI:
    obs = np.array([np.array(obs).astype(np.float32)])
    action, _states = model.predict(obs)
    print("action:", action)
    state, reward, done, info = env.step(action)
    print("HI:", state, "G:", reward, "is_done:", done, '\n')
    print('####################################################', '\n')'''

# ********************** Double Deep Q Learning (PER) ************************

'''env = CustomEnv(is_training=False, nn='dnn', input_data='raw', hierarchical=False)
agent.test(env)
'''

'''policy = pd.DataFrame.from_dict(policy).T
policy['remaining_cycles'] = policy['failure_state'] - policy['replace_state']
policy_test = pd.DataFrame.from_dict(policy_test).T
policy_test['remaining_cycles'] = policy_test['failure_state'] - policy_test['replace_state']

M = int(df_A['unit'].max() * 0.2)
IMC = (M * c_r) / np.sum(policy_test['failure_state'] - 1)
CMC = (M * (c_r + c_f)) / np.sum(policy_test['failure_state'])
print("mean reward = ", np.mean(policy_test['reward']))
print("Average remaining cycles", np.average(policy_test['remaining_cycles']))

env.close()
print("done")'''

############################################################################################################
# **********************************************************************************************************
#                                          Alarm Management
# ##########################################################################################################

# ************************** Behavioral Cloning ******************************

'''# Restore the model.
saver.restore(sess, "./tf_ckpt/model")

##  Control Params 

control_params = []
for state_param in Y:
    control_params.append(sess.run(model, feed_dict={y: [state_param.flatten()]})[0])
control_params = np.array(control_params)

# matlab.engine.shareEngine (type in MATLAB command window)

import matlab
import matlab.engine

eng = matlab.engine.start_matlab()
eng = matlab.engine.connect_matlab()

controls = np.insert(U, [4, 7], 0, axis=1)
controls = np.insert(controls, 11, 100, axis=1)

init_control = matlab.double(controls[0].tolist())
eng.workspace['xmv'] = init_control  # Initialize state

control = init_control
eng.set_param('tesys', 'SimulationCommand', 'start', 'SimulationCommand', 'pause', nargout=0)
while eng.get_param('tesys', 'SimulationStatus') != ('stopped' or 'terminating'):
    # Pause the Simulation for each timestep
    eng.set_param('tesys', 'SimulationCommand', 'continue', 'SimulationCommand', 'pause', nargout=0)
    if eng.get_param('tesys', 'SimulationStatus') == ('stopped' or 'terminating'):
        print('Episode Done')
    else:
        # Get current state based on previous control action
        next_state = eng.TEP_PythonSimulink(str(control))
        # Generate the control action based on the current state
        control = sess.run(model, feed_dict={y: [np.delete(np.array(next_state), 18)]})[0]
        control = np.insert(control, [4, 7], 0, axis=0)
        control = np.insert(control, 11, 100, axis=0)
        control = matlab.double(control.tolist())
        '''

# ******************** Cycle of Learning (Actor Critic) **********************

'''# Restore the model.
saver.restore(sess, "./tf_ckpt/model")

##  Control Params 

control_params = []
for state_param in Y:
    control_params.append(sess.run(model, feed_dict={y: [state_param.flatten()]})[0])
control_params = np.array(control_params)

# matlab.engine.shareEngine (type in MATLAB command window)'''

# ********************************* A2C **************************************

# ********************************* TD3 **************************************

evaluations = eval_policy(policy)

eng1.quit()
# eng2.quit()
env.close()
print("done")
