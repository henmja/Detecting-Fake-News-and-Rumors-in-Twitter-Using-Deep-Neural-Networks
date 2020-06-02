import pandas as pd

#CNN_Accuracies = pd.read_pickle('../CNN/CNN_accuracies.pkl')
#CNN_Accuracies = CNN_Accuracies*100
#print(CNN_Accuracies)

#LSTM_Accuracies = pd.read_pickle('../LSTM/LSTM_accuracies.pkl')
#LSTM_Accuracies = LSTM_Accuracies*100
#print(LSTM_Accuracies)
BiLSTM_Accuracies = pd.read_pickle('BiLSTM_smote_accuracies.pkl')
C_LSTM_Accuracies = pd.read_pickle('C_LSTM__smote_accuracies.pkl') 
CNN_Accuracies = pd.read_pickle('CNN_smote_accuracies.pkl')
Khan_LSTM_Accuracies = pd.read_pickle('Khan_BiLSTM_smote_accuracies.pkl')
Khan_C_LSTM_Accuracies = pd.read_pickle('Khan_C_LSTM_smote_accuracies.pkl')
#SVM_Accuracies = pd.read_pickle('SVM_accuracies.pkl')

#Khan_LSTM_Smote_Accuracies = pd.read_pickle('Khan_BiLSTM_smote_accuracies.pkl')
import numpy as np
#print(Khan_LSTM_Accuracies)
#print(Khan_LSTM_Smote_Accuracies.shape)
from math import sqrt
from numpy.random import seed
from numpy.random import randn
from numpy import mean
BiLSTM_Accuracies = np.mean(BiLSTM_Accuracies,axis=1)
C_LSTM_Accuracies = np.mean(C_LSTM_Accuracies,axis=1)
CNN_Accuracies = np.mean(CNN_Accuracies,axis=1)
Khan_LSTM_Accuracies = np.mean(Khan_LSTM_Accuracies,axis=1)
Khan_C_LSTM_Accuracies = np.mean(Khan_C_LSTM_Accuracies,axis=1)
#SVM_Accuracies = np.mean(SVM_Accuracies,axis=1)
print(BiLSTM_Accuracies)
print(C_LSTM_Accuracies)
print(CNN_Accuracies)
print(Khan_LSTM_Accuracies)
print(Khan_C_LSTM_Accuracies)
#print(np.mean(Khan_LSTM_Accuracies,axis=1))
#print(np.mean(Khan_LSTM_Smote_Accuracies,axis=1))
#print(max(Khan_LSTM_Accuracies,axis=1))
#print(max(Khan_LSTM_Smote_Accuracies,axis=1))
from scipy.stats import sem
from scipy.stats import t
mean1_glob = 0
mean2_glob = 0
# function for calculating the t-test for two independent samples
def independent_ttest(data1, data2, alpha):
	# calculate means
    mean1, mean2 = mean(data1), mean(data2)
    print(mean1)
    print(mean2)
    mean1_glob = mean1
    mean2_glob = mean2
	# calculate standard errors
    se1, se2 = sem(data1), sem(data2)
	# standard error on the difference between the samples
    sed = sqrt(se1**2.0 + se2**2.0)
	# calculate the t statistic
    t_stat = (mean1 - mean2) / sed
	# degrees of freedom
    df = len(data1) + len(data2) - 2
	# calculate the critical value
    cv = t.ppf(1.0 - alpha, df)
	# calculate the p-value
    p = (1.0 - t.cdf(abs(t_stat), df)) * 2.0
	# return everything
    return t_stat, df, cv, p
 
# seed the random number generator
seed(1)

alpha = 0.01
t_stat, df, cv, p = independent_ttest(C_LSTM_Accuracies, BiLSTM_Accuracies, alpha)
#data1 = 5 * randn(100) + 50
#data2 = 5 * randn(100) + 51
#t_stat, df, cv, p = independent_ttest(data1, data2, alpha)
print('t=%.3f, df=%d, cv=%.3f, p=%.3f' % (t_stat, df, cv, p))
# interpret via critical value
if abs(t_stat) <= cv:
    print(abs(t_stat))
    print(cv)
    print('Accept null hypothesis that the means are equal.')
else:
    print(t_stat)
    print(cv)
    print('Reject the null hypothesis that the means are equal.')
# interpret via p-value
if p > alpha:
    print(p)
    print('Accept null hypothesis that the means are equal.')
else:
    print(p)
    print('Reject the null hypothesis that the means are equal.')

t_stat, df, cv, p = independent_ttest(CNN_Accuracies, BiLSTM_Accuracies, alpha)
#data1 = 5 * randn(100) + 50
#data2 = 5 * randn(100) + 51
#t_stat, df, cv, p = independent_ttest(data1, data2, alpha)
print('t=%.3f, df=%d, cv=%.3f, p=%.3f' % (t_stat, df, cv, p))
# interpret via critical value
if abs(t_stat) <= cv:
    print(abs(t_stat))
    print(cv)
    print('Accept null hypothesis that the means are equal.')
else:
    print(t_stat)
    print(cv)
    print('Reject the null hypothesis that the means are equal.')
# interpret via p-value
if p > alpha:
    print(p)
    print('Accept null hypothesis that the means are equal.')
else:
    print(p)
    print('Reject the null hypothesis that the means are equal.')

t_stat, df, cv, p = independent_ttest(Khan_LSTM_Accuracies, BiLSTM_Accuracies, alpha)
#data1 = 5 * randn(100) + 50
#data2 = 5 * randn(100) + 51
#t_stat, df, cv, p = independent_ttest(data1, data2, alpha)
print('t=%.3f, df=%d, cv=%.3f, p=%.3f' % (t_stat, df, cv, p))
# interpret via critical value
if abs(t_stat) <= cv:
    print(abs(t_stat))
    print(cv)
    print('Accept null hypothesis that the means are equal.')
else:
    print(t_stat)
    print(cv)
    print('Reject the null hypothesis that the means are equal.')
# interpret via p-value
if p > alpha:
    print(p)
    print('Accept null hypothesis that the means are equal.')
else:
    print(p)
    print('Reject the null hypothesis that the means are equal.')

t_stat, df, cv, p = independent_ttest(Khan_C_LSTM_Accuracies, BiLSTM_Accuracies, alpha)
#data1 = 5 * randn(100) + 50
#data2 = 5 * randn(100) + 51
#t_stat, df, cv, p = independent_ttest(data1, data2, alpha)
print('t=%.3f, df=%d, cv=%.3f, p=%.3f' % (t_stat, df, cv, p))
# interpret via critical value
if abs(t_stat) <= cv:
    print(abs(t_stat))
    print(cv)
    print('Accept null hypothesis that the means are equal.')
else:
    print(t_stat)
    print(cv)
    print('Reject the null hypothesis that the means are equal.')
# interpret via p-value
if p > alpha:
    print(p)
    print('Accept null hypothesis that the means are equal.')
else:
    print(p)
    print('Reject the null hypothesis that the means are equal.')

#t_stat, df, cv, p = independent_ttest(SVM_Accuracies, BiLSTM_Accuracies, alpha)
#print('t=%.3f, df=%d, cv=%.3f, p=%.3f' % (t_stat, df, cv, p))
# interpret via critical value
#if abs(t_stat) <= cv:
#    print(abs(t_stat))
#    print(cv)
#    print('Accept null hypothesis that the means are equal.')
#else:
#    print(t_stat)
#    print(cv)
#    print('Reject the null hypothesis that the means are equal.')
# interpret via p-value
#if p > alpha:
#    print(p)
#    print('Accept null hypothesis that the means are equal.')
#else:
#    print(p)
#    print('Reject the null hypothesis that the means are equal.')
