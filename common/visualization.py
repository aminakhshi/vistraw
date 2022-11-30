#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 05:05:08 2022

@author: amin
"""

import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.preprocessing import LabelEncoder
import os
import pickle
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.utils import compute_class_weight
from sklearn.metrics import f1_score
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(y_true, y_pred, output_class, kw=None):
    labels = y_true.index.get_level_values(level=output_class).categories
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = plt.figure(num=1, figsize=(12, 10))
    ax = fig.add_subplot(111)
    disp = sns.heatmap(cm, annot=True, fmt='g', ax=ax, cbar=False, square=True)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    plt.show()
    if hasattr(kw, 'save_output'):
        try:
            plt.savefig(os.path.join(kw.results_path,'_fold-'+kw.fold+'_confusionMatrix.png'), dpi=300, facecolor='w', bbox_inches='tight')
        except:
            plt.savefig(os.path.join(kw.results_path, '_confusionMatrix.png'), dpi=300, facecolor='w', bbox_inches='tight')
        plt.close()



def plot_confusion_matrix_ts(y_true, y_pred, output_class, kw=None):
    labels = y_true.index.get_level_values(level=output_class).categories
    # labels = list(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = plt.figure(num=1, figsize=(12, 10))
    ax = fig.add_subplot(111)
    disp = sns.heatmap(cm, annot=True, fmt='g', ax=ax, cbar=False, square=True)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    plt.show()
    if hasattr(kw, 'save_output'):
        try:
            plt.savefig(os.path.join(kw.results_path, kw.experiment+'_fold-'+kw.fold+'_confusionMatrix.png'), dpi=300, facecolor='w', bbox_inches='tight')
        except:
            plt.savefig(os.path.join(kw.results_path, kw.experiment+'_confusionMatrix.png'), dpi=300, facecolor='w', bbox_inches='tight')
        plt.close()

def plot_ovr_roc_curves_ts(results, y_prob, le, output_class, kw=None):
    labels = results.index.get_level_values(level=output_class).categories
    testY = le.transform(results['True Values'])
    auc = roc_auc_score(testY, y_prob, multi_class='ovr', average='weighted')

    fpr = {}
    tpr = {}
    thresh = {}

    n_class = len(labels)
    transformed_order = {}

    for i in range(n_class):
        fpr[i], tpr[i], thresh[i] = roc_curve(testY, y_prob[:, i], pos_label=i)
        transformed_order[le.inverse_transform([i])[0]] = i

    fig = plt.figure()

    for peptide in labels:
        plt.plot(fpr[transformed_order[peptide]], tpr[transformed_order[peptide]], linestyle='--',
                 label=peptide + ' vs Rest')
    try:
        file_name = kw.experiment[kw.experiment.rindex('MW'):]
        if hasattr(kw, 'filter'):
            file_name = file_name+'_'+kw.filter[1]
        plt.title('Multiclass ROC curve for {exp} (AUC = {auc:.3f})'.format(exp=file_name, auc=auc))
    except:
        plt.title('Multiclass ROC curve (AUC = {auc:.3f})'.format(auc=auc))
        
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.show()
    if hasattr(kw, 'save_output'):
        try:  
            plt.savefig(os.path.join(kw.results_path, kw.experiment+'_fold-'+kw.fold+'_MulticlassROC.png'), dpi=300, facecolor='w', bbox_inches='tight')
        except:
            plt.savefig(os.path.join(kw.results_path, kw.experiment+'_MulticlassROC.png'), dpi=300, facecolor='w', bbox_inches='tight')
        plt.close()

def plot_ovr_roc_curves(results, y_prob, le, output_class, kw=None):
    labels = results.index.get_level_values(level=output_class).categories
    testY = le.transform(results['True Values'])
    y_probs = np.vstack((1-y_prob, y_prob)).T
    auc = roc_auc_score(testY, y_prob)
    fpr, tpr, thresholds = roc_curve(testY, y_prob)
    fig = plt.figure()
    plt.plot(fpr, tpr, linestyle='--')
    plt.title('Multiclass ROC curve (AUC = {auc:.3f})'.format(auc=auc))    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    # plt.legend(loc='best')
    plt.show()
    if hasattr(kw, 'save_output'):
        try:  
            plt.savefig(os.path.join(kw.results_path, '_fold-'+kw.fold+'_ROC.png'), dpi=300, facecolor='w', bbox_inches='tight')
        except:
            plt.savefig(os.path.join(kw.results_path, '_ROC.png'), dpi=300, facecolor='w', bbox_inches='tight')
        plt.close()


def plot_prediction_distributions(results, output_class, miss_elects=None, kw=None):
    labels = results.index.get_level_values(level=output_class).categories
    disp = sns.displot(data=results.reset_index(), x='Predicted Values', col='True Values', col_wrap=2,
                       hue='Predicted Values', fill=True)
    plt.show()
    if hasattr(kw, 'save_output'):
        try: 
            plt.savefig(os.path.join(kw.results_path, '_fold-'+kw.fold+'_predictionDistributions.png'), dpi=300, facecolor='w', bbox_inches='tight')
        except:
            plt.savefig(os.path.join(kw.results_path, '_predictionDistributions.png'), dpi=300, facecolor='w', bbox_inches='tight')
        plt.close()
    
    temp = results.reset_index()
    # accurate_predictions = results[(results['True Values'] == results['Predicted Values'])]
    # not_accurate = results[~(results['True Values'] == results['Predicted Values'])]
    # temp_accurate = accurate_predictions.reset_index()
    vars = list(results.index.names)
    vars.remove(output_class)
    vars.remove('trial_id')
    for i, col in enumerate(vars):
        all_counts = temp[col].value_counts(sort=False)
        all_counts = temp[col].unique()
        percent_correct = np.zeros_like(all_counts, dtype=np.float16)
        for index, val_index in enumerate(all_counts): #percent_correct.items():
            acc_counts = temp.loc[temp[col]==val_index]
            percent_correct[index] = f1_score(acc_counts['True Values'],acc_counts['Predicted Values'], average='macro')
        
        percent_correct = pd.Series(percent_correct, index=all_counts, name=col)
        # accurate_counts = temp_accurate[col].value_counts(sort=False)
        # percent_correct = accurate_counts / all_counts
        fig = plt.figure(num=i, figsize=(20, 8))
        ax = fig.add_subplot(111)
        sns.barplot(x=percent_correct.index, y=percent_correct.values, ax=ax)
        ax.set_title(percent_correct.name)
        ax.set_ylabel('% Predicted Correctly')
        ax.set_ylim(0, 1)
        ax.bar_label(ax.containers[0], rotation=90, fmt='%.2f')
        plt.tight_layout()
        plt.show()
        if hasattr(kw, 'save_output'):
            try:
                plt.savefig(os.path.join(kw.results_path, '_fold-'+kw.fold+f'_{col}_predictionDistributions.png'), dpi=300, facecolor='w', bbox_inches='tight')
            except:
                plt.savefig(os.path.join(kw.results_path, f'_{col}_predictionDistributions.png'), dpi=300, facecolor='w', bbox_inches='tight')
            plt.close()
    miss_elects = [0,56,78,88,90,99]
    if miss_elects:
        for i in miss_elects:
            percent_correct[i] = 0
    percent_correct = percent_correct.sort_index()
    plx_map = percent_correct.to_numpy().reshape([10, 10])
    plx_map_b = [0,73,74,75,92,93,94,79,96,0]
    plx_map_l = [0,65,66,67,68,69,70,71,72,0]
    plx_map_r = [58,25,27,28,29,30,31,32,80,0]
    plx_map_t = [0,17,1,2,18,19,20,21,23,58]

from mpl_toolkits.axes_grid1 import host_subplot

    fig = plt.figure(num=1, figsize=(12, 10))
    ax = fig.add_subplot(111)
    # ax.set_xlabel('Predicted labels')
    # ax.set_ylabel('True labels')
    disp = sns.heatmap(plx_map, annot=True, fmt='.2f', ax=ax, vmin=0, vmax=1,cbar=False, square=True, zorder=1)
    ax.xaxis.set_ticklabels(plx_map_b)
    ax.yaxis.set_ticklabels(plx_map_l)
    ax2 = ax.twinx()
    ax3 =ax.twiny()
    ax2.set_xticklabel(xp, plx_map_t)
    ax3.set_ytick(yp, plx_map_r)
    # disp0 = sns.heatmap(np.zeros_like(plx_map), ax=ax2,cbar=False, square=True, zorder=0, alpha=0)
    # ax2.yaxis_set_label(ax.yaxis_get_label())
    # ax2.set_xlim(ax.get_xlim())

    secax = plt.gca().secondary_xaxis('top')
    secay = plt.gca().secondary_yaxis('right')
    secax.set_xlabel('Extraverted')
    xb = ax.get_xticklabels(minor=False, which=None)
    xp = [t.get_position() for t in ax.get_xticklabels()]
    xp = [(t[0], 10) for t in xp]
    yp = [t.get_position() for t in ax.get_yticklabels()]
    yp = [(10, t[1]) for t in yp]
    secax.set_xtick(xp, plx_map_t)

    secax = ax.secondary_xaxis('top', functions=(xp, plx_map_t))
    for i, val in xb:
        print (i, val)
    ax.get_yticklabels(minor=False, which=None)
    secay.set_xticklabels(minor=False, which=None)

    secay.get_xticklabels(minor=False, which=None)
    ax.tick_params(labeltop=True, labelright=True)
    ax.xaxis.set_ticklabels(plx_map_r)
    ax.yaxis.set_ticklabels(plx_map_t)
    xlbl = ax.xaxis.get_label()
    xlbl.get_position()


    # ax1 = ax.twinx()
    # ax1.yaxis.set_ticklabels(plx_map_r)
    plt.show()
    if hasattr(kw, 'save_output'):
        try:
            plt.savefig(os.path.join(kw.results_path, kw.experiment+'_fold-'+kw.fold+'_confusionMatrix.png'), dpi=300, facecolor='w', bbox_inches='tight')
        except:
            plt.savefig(os.path.join(kw.results_path, kw.experiment+'_confusionMatrix.png'), dpi=300, facecolor='w', bbox_inches='tight')
        plt.close()

    return 
# def plot_summary_plot(shap_values, samples, plot_type=None, class_names= None, le=None, kw=None):
#     if not class_names:
#         class_names = le.inverse_transform(range(len(set(y_test)))).tolist()
#     # if plot_type:
        
#     fig = plt.figure(num=1, figsize=(12, 10))
#     ax = fig.add_subplot(111)
#     disp = shap.summary_plot(shap_values, plot_type='bar', feature_names=samples.columns, show=False, class_names=class_names)
#     # ax.set_xlabel('Predicted labels')
#     # ax.set_ylabel('True labels')
#     # ax.xaxis.set_ticklabels(labels)
#     # ax.yaxis.set_ticklabels(labels)
#     plt.show()
#     if hasattr(kw, 'save_output'):
#         try:
#             plt.savefig(os.path.join(kw.results_path, kw.experiment+'_fold-'+kw.fold+'_summary_plot_all.png'), dpi=300, facecolor='w', bbox_inches='tight')
#         except:
#             plt.savefig(os.path.join(kw.results_path, kw.experiment+'_summary_plot_all.png'), dpi=300, facecolor='w', bbox_inches='tight')
#         plt.close()
#         shap.plots.beeswarm(shap_values[i], samples, plot_size=0.9, show=False, color_bar=False)
# shap.plots.heatmap(shap_values[:100,:,0], max_display=20)
# shap.plots.waterfall(shap_values[:,:,0]) # For the first observation

# shap.plots.bar(shap_values[:100,:,:]) # default is max_display=12
# shap.plots.bar(shap_values[:,:,0].cohorts(2).abs.mean(0))
#     orderedPeptides = ['N4', 'A2', 'Y3', 'Q4', 'T4', 'V4', 'G4', 'E1']
#     fig, axes = plt.subplots(2, 4, figsize=(30,10))
#     for i in range(8):
#         x = orderedPeptides.index(le.inverse_transform([i])[0])
#         ax = axes.flat[x]
#         plt.sca(ax)
#         shap.plots.beeswarm(shap_values[i], samples, plot_size=0.9, show=False, color_bar=False)
#         ax.set_xlabel('')
#         ax.set_title(le.inverse_transform([i])[0])

# plt.subplots_adjust(left=0.1, right=0.9, top=0.9)
# plt.tight_layout()
# plt.savefig('/Users/wahlstenml/Documents/Neural Network/MW_008/MW_008/MW_008_NoTreatment/shap_summaryplot_perNode.png', facecolor='w', dpi=300, bbox_inches='tight')
#     fig, ax = plt.subplots(, figsize=(12, 10))
#     # ax = fig.add_subplot(111)
#     for ind, val in enumerate(class_names):
#         shap.summary_plot(shap_values[ind], samples.values, feature_names=samples.columns, use_log_scale=True, plot_type='violin', plot_size=0.9, show=False)
#     for i in range(8):
#     p = shap.summary_plot(shap_values[i], samples.drop('Predicted Values', axis=1).values, feature_names=testX.columns, use_log_scale=True, plot_size=0.9, show=False)
#     plt.savefig('/Users/wahlstenml/Documents/Neural Network/MW_008/MW_008/MW_008_NoTreatment/SHAP/shap_summaryDotPlot_Pulsed_node{X}.png'.format(X=le.inverse_transform([i])[0]), facecolor='w', dpi=300, bbox_inches='tight')
#     plt.close()
#     print(le.inverse_transform([i])[0])
    
#     if hasattr(kw, 'class_names'):
#         fig = plt.figure(num=i, figsize=(10, 8))
#         ax = fig.add_subplot(111)
#         sns.barplot(x=percent_correct.index, y=percent_correct.values, ax=ax)
#         ax.set_title(percent_correct.name)
#         ax.set_ylabel('% Predicted Correctly')
#         ax.bar_label(ax.containers[0])
#         plt.show()
#         if hasattr(kw, 'save_output'):
#             try:
#                 plt.savefig(os.path.join(kw.results_path, kw.experiment+'_fold-'+kw.fold+'_{level}_predictionDistributions.png'), dpi=300, facecolor='w', bbox_inches='tight')
#             except:
#                 plt.savefig(os.path.join(kw.results_path, kw.experiment+'_{level}_predictionDistributions.png'), dpi=300, facecolor='w', bbox_inches='tight')
#             plt.close()
#         feature_names=X_test.columns, show=False, class_names=class_names,
