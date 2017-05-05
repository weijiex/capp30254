 ### code is modified from https://github.com/rayidghani/magicloops/blob/master/magicloops.py


from __future__ import division
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.grid_search import ParameterGrid, GridSearchCV
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import random
import pylab as pl
import matplotlib.pyplot as plt
from scipy import optimize
import time
import seaborn as sns
import sys

def define_clfs_params(grid_size):

    clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        'DT': DecisionTreeClassifier(),
        'KNN': KNeighborsClassifier(n_neighbors=3),
        'BAG': BaggingClassifier(DecisionTreeClassifier(max_depth=10), n_estimators=5, max_samples=0.6, max_features=1)
            }

    large_grid = { 
    'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']},
    'BAG': {'n_estimators': [5,10,20], 'max_samples':[0.4,0.5,0.6]}
           }
    
    small_grid = {
    'RF':{'n_estimators': [10,100], 'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.001,0.1,1,10]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']},
    'BAG': {'n_estimators': [5,10], 'max_samples':[0.4,0.6]}
           }
    
    test_grid = { 
    'RF':{'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'LR': { 'penalty': ['l1'], 'C': [0.01]},
    'AB': { 'algorithm': ['SAMME'], 'n_estimators': [1]},
    'DT': {'criterion': ['gini'], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
    'SVM' :{'C' :[0.01],'kernel':['linear']},
    'KNN' :{'n_neighbors': [5],'weights': ['uniform'],'algorithm': ['auto']},
    'BAG':{'n_estimators': [5]}
           }
    
    if grid_size == 'large':
        return clfs, large_grid
    elif grid_size == 'small':
        return clfs, small_grid
    elif grid_size == 'test':
        return clfs, test_grid
    else:
        return 0, 0

        
        
def generate_binary_at_k(y_pred_probs, k):
    '''
    Turn probability estimates into binary at level k
    '''

    cutoff_index = int(len(y_pred_probs) * (k / 100.0))
    y_pred_binary = [1 if x < cutoff_index else 0 for x in range(len(y_pred_probs))]
    return y_pred_binary

def precision_at_k(y_true, y_pred_probs, k):
    '''
    Calculate precision score for probability estimates at level k
    '''

    preds_at_k = generate_binary_at_k(y_pred_probs, k)
    precision = precision_score(y_true, preds_at_k)
    return precision

def recall_at_k(y_true, y_pred_probs, k):
    '''
    Calculate recall score for probability estimates at level k
    '''

    preds_at_k = generate_binary_at_k(y_pred_probs, k)
    recall = recall_score(y_true, preds_at_k)
    return recall

def plot_precision_recall_n(y_true, y_pred_probs, model_name):
    '''
    Plot precision recall curve
    '''

    from sklearn.metrics import precision_recall_curve
    y_score = y_pred_probs
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    plt.figure()
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')

    plt.title(model_name)
    plt.savefig('Graphs from evaluation/'+model_name)
    plt.close()
        
        
def clf_loop(models_to_run, clfs, grid, X, y, test_size, output=True):
    '''
    Run over certain clfs and grid, and evaluate each classifier by certain metrics
    Inputs:
        models_to_run: list of strings, classifiers to test
        clfs: dictionary of classifiers 
        grid: dictionary of parameters grid 
        X: pandas DataFrame, features
        y: pandas DataFrame
        test_size: float [0.0, 1.0], proportion of the total data used as test dataset
        output: boolean, save graphs and save result dataframe to csv file
    Output:
        pandas DataFrame, including classifiers, parameters, runtime, and evaluation scores
    '''

    results_df =  pd.DataFrame(columns=('model_type','clf', 'parameters', 'train_time', 'test_time',
                                        'accuracy','f1_score', 'precision', 'recall', 'auc', 
                                        'p_at_5', 'p_at_10', 'p_at_20',
                                        'r_at_5', 'r_at_10', 'r_at_20'))
    for n in range(1, 2):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=0)
        for index,clf in enumerate([clfs[x] for x in models_to_run]):
            print (models_to_run[index])
            parameter_values = grid[models_to_run[index]]
            for p in ParameterGrid(parameter_values):
                try:
                    clf.set_params(**p)
                    train_start = time.time()
                    clf.fit(X_train, y_train)
                    train_end = time.time()
                    train_time = train_end - train_start

                    test_start = time.time()
                    y_pred = clf.predict(X_test)
                    test_end = time.time()
                    test_time = test_end - test_start

                    y_pred_probs = clf.predict_proba(X_test)[:,1]
                    y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test), reverse=True))
                    results_df.loc[len(results_df)] = [models_to_run[index],clf, p, train_time, test_time,
                                                       accuracy_score(y_test, y_pred),
                                                       f1_score(y_test, y_pred),
                                                       precision_score(y_test, y_pred),
                                                       recall_score(y_test, y_pred),
                                                       roc_auc_score(y_test, y_pred),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                                                       precision_at_k(y_test_sorted,y_pred_probs_sorted,20.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                                                       recall_at_k(y_test_sorted,y_pred_probs_sorted,20.0)]

                    if output:
                        model_name = models_to_run[index] + str(len(results_df))
                        plot_precision_recall_n(y_test, y_pred_probs,model_name)
                except IndexError as e:
                    print ('Error:',e)
                    continue
    if output:
        results_df.to_csv('clf_evaluations.csv')
    return results_df
 
