import pandas as pd
import numpy as np
try:
    BiLSTM_Accuracies = pd.read_pickle('/local/home/henrikm/Fakenews_Classification/T_Test/BiLSTM_smote_accuracies.pkl')
    BiLSTM_Accuracies = np.mean(BiLSTM_Accuracies,axis=1)
except FileNotFoundError:
    pass
try:
    C_LSTM_Accuracies = pd.read_pickle('/local/home/henrikm/Fakenews_Classification/T_Test/C-LSTM_smote_accuracies.pkl')
    C_LSTM_Accuracies = np.mean(C_LSTM_Accuracies,axis=1)
except FileNotFoundError:
    pass
try:
    C_LSTM_Func_Accuracies = pd.read_pickle('/local/home/henrikm/Fakenews_Classification/T_Test/CNN_LSTM_smote_accuracies.pkl')  
    C_LSTM_Func_Accuracies = np.mean(C_LSTM_Func_Accuracies,axis=1)
except FileNotFoundError:
    pass
try:
    CNN_Accuracies = pd.read_pickle('/local/home/henrikm/Fakenews_Classification/T_Test/CNN_smote_accuracies.pkl')
    CNN_Accuracies = np.mean(CNN_Accuracies,axis=1) 
except FileNotFoundError:
    pass
try:
    Khan_LSTM_Accuracies = pd.read_pickle('/local/home/henrikm/Fakenews_Classification/T_Test/Khan_BiLSTM_smote_accuracies.pkl')
    Khan_LSTM_Accuracies = np.mean(Khan_LSTM_Accuracies,axis=1)
except FileNotFoundError:
    pass
try:
    Khan_C_LSTM_Accuracies = pd.read_pickle('/local/home/henrikm/Fakenews_Classification/T_Test/Khan_C_LSTM_smote_accuracies.pkl')
    Khan_C_LSTM_Accuracies = np.mean(Khan_C_LSTM_Accuracies,axis=1)
except FileNotFoundError:
    pass
import numpy as np
from math import sqrt
from numpy.random import seed
from numpy.random import randn
from numpy import mean
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
try:
    t_stat, df, cv, p = independent_ttest(C_LSTM_Func_Accuracies, BiLSTM_Accuracies, alpha)
    print('T-test for CNN+LSTM and BiLSTM')
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
except NameError:
    pass
try:
    t_stat, df, cv, p = independent_ttest(C_LSTM_Accuracies, BiLSTM_Accuracies, alpha)
    print('T-test for C-LSTM and BiLSTM')
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
except NameError:
    pass
try:
    t_stat, df, cv, p = independent_ttest(CNN_Accuracies, BiLSTM_Accuracies, alpha)
    print('T-test for CNN and BiLSTM')
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
except NameError:
    pass
try:
    t_stat, df, cv, p = independent_ttest(Khan_LSTM_Accuracies, BiLSTM_Accuracies, alpha)
    print('T-test for BiLSTM (Khan) and BiLSTM')
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
except NameError:
    pass
try:
    t_stat, df, cv, p = independent_ttest(Khan_C_LSTM_Accuracies, BiLSTM_Accuracies, alpha)
    print('T-test for C-LSTM (Khan) and BiLSTM')
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
except NameError:
    pass
