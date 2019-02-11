#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
process_signal.py: Python script that processes signals after they have been aligned
to events.
"""


__author__ = "DM Brady"
__datewritten__ = "06 Feb 2019"
__lastmodified__ = "06 Feb 2019"


import sys
import numpy as np
import pandas as pd
import seaborn as sns
from imaging_analysis.signal_processing import DeltaFOverF, PolyfitWindow, SmoothSignalWithPeriod, ZScoreCalculator
from imaging_analysis.utils import PrintNoNewLine
import matplotlib.pyplot as plt
from scipy import stats

sns.set_style('darkgrid')
#######################################################################
# VARIABLES TO ALTER

analysis_blocks = [
    {
        'load_file': 'correct',
        'save_file_as': 'correct_processed',
        'z_score_window': [-8, -3],
        'to_csv': True,
        'downsample': 10,
        'quantification': 'mean', # options are AUC, median, and mean
        'baseline_window': [-5, -2],
        'response_window': [1, 4]
    },
    {
        'load_file': 'iti_start',
        'save_file_as': 'iti_start_processed',
        'z_score_window': [-10, -5],
        'to_csv': True,
        'downsample': 10,
        'quantification': 'AUC', # options are AUC, median, and mean
        'baseline_window': [-6, -3],
        'response_window': [0, 3]
    }
]
# Checks if a directory path to the data is provided, if not, will
# use what is specified in except
try:
    dpath = sys.argv[1]
except IndexError:
    #dpath = '/Users/DB/Development/Monkey_frog/data/social/TDT-LockinRX8-22Oct2014_20-4-15_DT1_041718'
    dpath = '/Users/DB/Development/Monkey_frog/data/KN_newRigData/RS/12/FirstFibPho-180817-160254/'

for block in analysis_blocks:
    # Extract analysis block params
    load_file = block['load_file']
    save_file_as = block['save_file_as']
    to_csv = block['to_csv']
    downsample = block['downsample']
    z_score_window = block['z_score_window']
    quantification = block['quantification']
    baseline_window = block['baseline_window']
    response_window = block['response_window']

    print('Processing signals for {}\n'.format(load_file))

    # Load data
    signal = pd.read_csv(dpath + load_file + '_all_traces.csv', index_col=0)
    reference = pd.read_csv(dpath + load_file + '_reference_all_traces.csv', index_col=0)

    # Down sample data
    if downsample > 0:
        signal.reset_index(inplace=True)
        reference.reset_index(inplace=True)
        sample = (signal.index.to_series() / downsample).astype(int)
        signal = signal.groupby(sample).mean()
        reference = reference.groupby(sample).mean()
        signal = signal.set_index('index')
        reference = reference.set_index('index')

    # Get plotting read
    figure = plt.figure(figsize=(12, 12))
    figure.subplots_adjust(hspace=1.3)
    ax1 = plt.subplot2grid((6, 2), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((6, 2), (2, 0), rowspan=2)
    ax3 = plt.subplot2grid((6, 2), (4, 0), rowspan=2)
    ax4 = plt.subplot2grid((6, 2), (0, 1), rowspan=3)
    ax5 = plt.subplot2grid((6, 2), (3, 1), rowspan=3)
    # fig, axs = plt.subplots(2, 2, sharex=False, sharey=False)
    # fig.set_size_inches(12, 12)

############################### PLOT AVERAGE EVOKED RESPONSE ######################
    PrintNoNewLine('Calculating average filtered responses...')
    signal_mean = signal.mean(axis=1)
    reference_mean = reference.mean(axis=1)

    signal_se = signal.sem(axis=1)
    reference_se = reference.sem(axis=1)

    signal_dc = signal_mean.mean()
    reference_dc = reference_mean.mean()

    signal_avg_response = signal_mean - signal_dc 
    reference_avg_response = reference_mean - reference_dc

    # Plotting signal
    # current axis
    #curr_ax = axs[0, 0]
    curr_ax = ax1
    curr_ax.plot(signal_avg_response.index, signal_avg_response.values, color='b', linewidth=2)
    curr_ax.fill_between(signal_avg_response.index, (signal_avg_response - signal_se).values, 
        (signal_avg_response + signal_se).values, color='b', alpha=0.05)

    # Plotting reference
    curr_ax.plot(reference_avg_response.index, reference_avg_response.values, color='g', linewidth=2)
    curr_ax.fill_between(reference_avg_response.index, (reference_avg_response - reference_se).values, 
        (reference_avg_response + reference_se).values, color='g', alpha=0.05)

    # Plot event onset
    curr_ax.axvline(0, color='black', linestyle='--')
    curr_ax.set_ylabel('Voltage')
    curr_ax.set_xlabel('')
    curr_ax.legend(['465 nm', '405 nm', load_file])
    curr_ax.set_title('Average Lowpass Signal $\pm$ SEM: {} Trials'.format(signal.shape[1]))
    print('Done!')
############################# Calculate detrended signal #################################
    # Detrending
    PrintNoNewLine('Detrending signal...')
    fits = np.array([np.polyfit(reference.values[:, i],signal.values[:, i],1) for i in xrange(signal.shape[1])])
    Y_fit_all = np.array([np.polyval(fits[i], reference.values[:,i]) for i in np.arange(reference.values.shape[1])]).T
    Y_df_all = signal.values - Y_fit_all
    detrended_signal = pd.DataFrame(Y_df_all, index=signal.index)

    detrended_signal_mean = detrended_signal.mean(axis=1)

    detrended_signal_sem = detrended_signal.sem(axis=1)

    # Plotting signal
    # current axis
    curr_ax = ax2
    # # curr_ax = axs[1, 0]
    #curr_ax = plt.axes()
    zscore_start = detrended_signal[z_score_window[0]:z_score_window[1]].index[0]
    zscore_end = detrended_signal[z_score_window[0]:z_score_window[1]].index[-1]
    zscore_height = detrended_signal[z_score_window[0]:z_score_window[1]].mean(axis=1).min() - 2

    curr_ax.plot([zscore_start, zscore_end], [zscore_height, zscore_height], color='.1', linewidth=3)


    curr_ax.plot(detrended_signal_mean.index, detrended_signal_mean.values, color='b', linewidth=2)
    curr_ax.fill_between(detrended_signal_mean.index, (detrended_signal_mean - detrended_signal_sem).values, 
        (detrended_signal_mean + detrended_signal_sem).values, color='b', alpha=0.05)

    # Plot event onset
    curr_ax.legend(['z-score window'])
    curr_ax.axvline(0, color='black', linestyle='--')
    curr_ax.set_ylabel('Voltage')
    curr_ax.set_xlabel('')
    curr_ax.set_title('465 nm Average Detrended Signal $\pm$ SEM')

    print('Done!')
########### Calculate z-scores ###############################################
    PrintNoNewLine('Calculating Z-Scores and making rasters...')
    # calculate z_scores
    zscores = ZScoreCalculator(detrended_signal, baseline_start=z_score_window[0], 
        baseline_end=z_score_window[1])


############################ Make rasters #######################################
    # indice that is closest to event onset
    # curr_ax = axs[0, 1]
    curr_ax = ax4
    # curr_ax = plt.axes()
    zero = np.where(zscores.index == np.abs(zscores.index).min())[0][0]
    for_hm = zscores.T.copy()
    for_hm.index = for_hm.index + 1
    for_hm.columns = np.round(for_hm.columns, 1)
    try:
        sns.heatmap(for_hm.iloc[::-1], center=0, robust=True, ax=curr_ax, cmap='bwr',
            xticklabels=int(for_hm.shape[1]*.15), yticklabels=int(for_hm.shape[0]*.15))
    except:
        sns.heatmap(for_hm.iloc[::-1], center=0, robust=True, ax=curr_ax, cmap='bwr')
    curr_ax.axvline(zero, linestyle='--', color='black', linewidth=2)
    curr_ax.set_ylabel('Trial');
    curr_ax.set_xlabel('Time (s)');
    curr_ax.set_title('Z-Score Heat Map \n Baseline Window: {} to {} Seconds'.format(z_score_window[0], z_score_window[1]));
    print('Done!')
########################## Plot Z-score waveform ##########################
    PrintNoNewLine('Plotting Z-Score waveforms...')
    zscores_mean = zscores.mean(axis=1)

    zscores_sem = zscores.sem(axis=1)

    # Plotting signal
    # current axis
    # curr_ax = axs[1, 1]
    curr_ax = ax3
    #curr_ax = plt.axes()
    # Plot baseline and response
    baseline_start = zscores[baseline_window[0]:baseline_window[1]].index[0]
    baseline_end = zscores[baseline_window[0]:baseline_window[1]].index[-1]
    response_start = zscores[response_window[0]:response_window[1]].index[0]
    response_end = zscores[response_window[0]:response_window[1]].index[-1]
    baseline_height = zscores[baseline_window[0]:baseline_window[1]].mean(axis=1).min() - 0.5
    response_height = zscores[response_window[0]:response_window[1]].mean(axis=1).max() + .5

    curr_ax.plot([baseline_start, baseline_end], [baseline_height, baseline_height], color='.6', linewidth=3)
    curr_ax.plot([response_start, response_end], [response_height, response_height], color='r', linewidth=3)

    curr_ax.plot(zscores_mean.index, zscores_mean.values, color='b', linewidth=2)
    curr_ax.fill_between(zscores_mean.index, (zscores_mean - zscores_sem).values, 
        (zscores_mean + zscores_sem).values, color='b', alpha=0.05)

    # Plot event onset
    curr_ax.axvline(0, color='black', linestyle='--')

    curr_ax.set_ylabel('Z-Score')
    curr_ax.set_xlabel('Time (s)')
    curr_ax.legend(['baseline window', 'response window'])
    curr_ax.set_title('465 nm Average Z-Score Signal $\pm$ SEM')
    print('Done!')
##################### Quantification #################################
    PrintNoNewLine('Performing statistical testing on baseline vs response periods...')
    # Generating summary statistics
    if quantification == 'AUC':
        base = np.trapz(zscores[baseline_window[0]:baseline_window[1]], axis=0)
        resp = np.trapz(zscores[response_window[0]:response_window[1]], axis=0)
        ylabel = 'AUC'
    elif quantification == 'mean':
        base = np.mean(zscores[baseline_window[0]:baseline_window[1]], axis=0)
        resp = np.mean(zscores[response_window[0]:response_window[1]], axis=0)
        ylabel = 'Z-Score'
    elif quantification == 'median':
        base = np.median(zscores[baseline_window[0]:baseline_window[1]], axis=0)
        resp = np.median(zscores[response_window[0]:response_window[1]], axis=0)
        ylabel = 'Z-Score'

    base_sem = np.mean(base)/np.sqrt(base.shape[0])
    resp_sem = np.mean(resp)/np.sqrt(resp.shape[0])

    # Testing for normality (D'Agostino's K-Squared Test)
    normal_alpha = 0.05
    base_normal = stats.normaltest(base)
    resp_normal = stats.normaltest(resp)

    difference_alpha = 0.05
    if (base_normal[1] <= normal_alpha) or (resp_normal[1] <= normal_alpha):
        test = 'Wilcoxon Signed-Rank Test'
        stats_results = stats.wilcoxon(base, resp)
    else:
        test = 'Paired Sample T-Test'
        stats_results = stats.ttest_rel(base, resp)

    if stats_results[1] <= difference_alpha:
        sig = '**'
    else:
        sig = 'ns'

    #curr_ax = plt.axes() 
    curr_ax = ax5
    ind = np.arange(2)
    labels = ['baseline', 'response']
    bar_kwargs = {'width': 0.7,'color': ['.6', 'r'],'linewidth':2,'zorder':5}
    err_kwargs = {'zorder':0,'fmt': 'none','linewidth':2,'ecolor':'k'}
    curr_ax.bar(ind, [base.mean(), resp.mean()], tick_label=labels, **bar_kwargs)
    curr_ax.errorbar(ind, [base.mean(), resp.mean()], yerr=[base_sem, resp_sem],capsize=5, **err_kwargs)
    x1, x2 = 0, 1
    y, h, col = np.max([base.mean(), resp.mean()]) + np.max([base_sem, resp_sem])*1.3, 10, 'k'
    curr_ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    curr_ax.text((x1+x2)*.5, y+h, sig, ha='center', va='bottom', color=col)
    curr_ax.set_ylabel(ylabel)
    curr_ax.set_title('Baseline vs. Response Changes in Z-Score Signal \n {} of {}s'.format(test, quantification))

    print('Done!')
################# Save Stuff ##################################
    PrintNoNewLine('Saving everything...')
    save_path = dpath + save_file_as
    figure.savefig(save_path + '.png')
    plt.close()
    print('Done!')
