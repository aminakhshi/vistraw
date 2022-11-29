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
    plt.show()
    if hasattr(kw, 'save_output'):
        try:  
            plt.savefig(os.path.join(kw.results_path, '_fold-'+kw.fold+'_ROC.png'), dpi=300, facecolor='w', bbox_inches='tight')
        except:
            plt.savefig(os.path.join(kw.results_path, '_ROC.png'), dpi=300, facecolor='w', bbox_inches='tight')
        plt.close()




def plot_prediction_distributions(results, output_class, kw=None):
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

    accurate_predictions = results[(results['True Values'] == results['Predicted Values'])]
    not_accurate = results[~(results['True Values'] == results['Predicted Values'])]

    temp = results.reset_index()
    temp_accurate = accurate_predictions.reset_index()
    vars = list(results.index.names)
    vars.remove(output_class)
    vars.remove('trial_id')
    for i, col in enumerate(vars):
        all_counts = temp[col].value_counts(sort=False)
        accurate_counts = temp_accurate[col].value_counts(sort=False)
        percent_correct = accurate_counts / all_counts
        fig = plt.figure(num=i, figsize=(20, 8))
        ax = fig.add_subplot(111)
        sns.barplot(x=percent_correct.index, y=percent_correct.values, ax=ax)
        ax.set_title(percent_correct.name)
        ax.set_ylabel('% Predicted Correctly')
        ax.bar_label(ax.containers[0], rotation=90, fmt='%.2f')
        plt.tight_layout()
        plt.show()
        if hasattr(kw, 'save_output'):
            try:
                plt.savefig(os.path.join(kw.results_path, '_fold-'+kw.fold+f'_{col}_predictionDistributions.png'), dpi=300, facecolor='w', bbox_inches='tight')
            except:
                plt.savefig(os.path.join(kw.results_path, f'_{col}_predictionDistributions.png'), dpi=300, facecolor='w', bbox_inches='tight')
            plt.close()