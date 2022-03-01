import numpy as np
from sklearn.metrics import auc
import pandas

def get_val_metrics(labels, pred, probs=None, thresh=None):
    '''
    Calculates
    
    tpr: true positive rate
    fdr: false detection rate
    fpr: false positive rate
    nRGC: number of found GCs
    nGC: total number of GCs
    nFGC: number of false positives
    nS: total number of sources
    
    and AUC ROC for prediction 'pred' and according ground-truth labels.
    '''
    tpr, fdr, fpr, nRGC, nFGC, nGC, nS, _, _, _ = get_metrics(pred, labels)
    if probs is not None:
        auc_score_fdr, xvals, yvals = eval_with_auc_fdr(labels, probs, thresh)
        auc_score_fpr, xvals, yvals = eval_with_auc_fpr(labels, probs, thresh)
    else:
        auc_score_fdr, auc_score_fpr = '--', '--'
    stats_validation = pandas.DataFrame({'TPR': [tpr],
                                         'FDR': [fdr],
                                         'FPR': [fpr],
                                         'AUC(FDR,TPR)': [auc_score_fdr],
                                         'AUC(FPR,TPR)': [auc_score_fpr],
                                         '# found GCs': [nRGC],
                                         '# total GCs': [nGC],
                                         '# fake GCs': [nFGC],
                                         '# sources': [nS]})

    return stats_validation

def get_test_metrics(test_galaxies, galaxy_list, IDs, labels, pred, probs=None, thresh=None):
    '''
    Calculates and returns all metrics for testing (TPR, FPR, FDR, AUC ROC, ROC curves) as 
    well as lists of false positives, false negatives and true positives.
    '''
    # define variables for storing results
    true_pos, false_pos, false_det, auc_values, auc_values_fpr = [], [], [], [], []
    num_found_GCs, num_fake_GCs, num_GCs, num_sources = [],[],[],[]
    false_positives, false_negatives, auc_curve_fdr, auc_curve_fpr = {},{},{},{}
    found_GCs = {}

    # first go through all galaxies separately
    for gal in test_galaxies:
        # filter for single galaxy
        galactic_filter = galaxy_list==gal
        filtered_labels = labels[galactic_filter]
        filtered_pred = pred[galactic_filter]

        # also filter probability scores if given
        if probs is not None:
            filtered_probs = probs[galactic_filter]

        # calculate evaluation metrics
        tpr, fdr, fpr, nRGC, nFGC, nGC, nS, list_of_GCs_not_found, list_of_wrong_GCs, list_of_found_GCs = get_metrics(filtered_pred, filtered_labels)

        # store them
        false_positives[gal] = IDs[galactic_filter][list_of_wrong_GCs]
        false_negatives[gal] = IDs[galactic_filter][list_of_GCs_not_found]
        found_GCs[gal] = IDs[galactic_filter][list_of_found_GCs]
        true_pos.append(tpr)
        false_pos.append(fpr)
        false_det.append(fdr)
        num_found_GCs.append(nRGC)
        num_fake_GCs.append(nFGC)
        num_GCs.append(nGC)
        num_sources.append(nS)

        # calculate AUC if probability scores are given
        if probs is not None:
            auc_score, xvals, yvals = eval_with_auc_fdr(filtered_labels, filtered_probs, thresh)
            auc_values.append(auc_score)
            auc_curve_fdr[gal] = {'fdr': xvals, 'tpr': yvals}

            auc_score, xvals, yvals = eval_with_auc_fpr(filtered_labels, filtered_probs, thresh)
            auc_values_fpr.append(auc_score)
            auc_curve_fpr[gal] = {'fpr': xvals, 'tpr': yvals}
        else:
            auc_values.append('--')
            auc_values_fpr.append('--')

    # save results in dataframe
    stats_galaxies = pandas.DataFrame({'Galaxy': test_galaxies,
                                       'TPR': true_pos,
                                       'FDR': false_det,
                                       'FPR': false_pos,
                                       'AUC(FDR,TPR)': auc_values,
                                       'AUC(FPR,TPR)': auc_values_fpr,
                                       '# found GCs': num_found_GCs,
                                       '# total GCs': num_GCs,
                                       '# fake GCs': num_fake_GCs,
                                       '# sources': num_sources})

    # calculate scorese over all galaxies
    tpr, fdr, fpr, nRGC, nFGC, nGC, nS, _, _, _ = get_metrics(pred, labels)
    # if probability score is given, calculate AUC
    if probs is not None:
        auc_score, xvals, yvals = eval_with_auc_fdr(labels, probs, thresh)
        auc_curve_fdr['ALL'] = {'fdr': xvals, 'tpr': yvals}
        auc_score_fpr, xvals, yvals = eval_with_auc_fpr(labels, probs, thresh)
        auc_curve_fpr['ALL'] = {'fpr': xvals, 'tpr': yvals}
    else:
        auc_score, auc_score_fpr = '--', '--'

    # save results in dataframe
    stats_all = pandas.DataFrame({'Galaxy': ['ALL'],
                                             'TPR': [tpr],
                                             'FDR': [fdr],
                                             'FPR': [fpr],
                                             'AUC(FDR,TPR)': [auc_score],
                                             'AUC(FPR,TPR)': [auc_score_fpr],
                                             '# found GCs': [nRGC],
                                             '# total GCs': [nGC],
                                             '# fake GCs': [nFGC],
                                             '# sources': [nS]})

    return stats_galaxies, stats_all, auc_curve_fdr, auc_curve_fpr, false_positives, false_negatives, found_GCs

def get_metrics(prediction, labels):
    '''
    Calculate all test metrics for predictions given ground-truth labels.
    '''
    true_pos_rate = np.mean(prediction[labels==1]==1)
    false_det_rate = np.mean(labels[prediction==1]==0)
    false_pos_rate = np.mean(prediction[labels==0]==1)
    list_of_found_GCs = np.where((labels==1)*(prediction==1))[0]
    list_of_GCs_not_found = np.where((labels==1)*(prediction==0))[0]
    list_of_wrong_GCs = np.where((prediction==1)*(labels==0))[0]
    num_GCs = np.sum(labels)
    num_found_GCs = np.sum(prediction[labels==1]==1)
    num_fake_GCs = np.sum(prediction[labels==0]==1)
    num_sources = len(labels)

    return true_pos_rate, false_det_rate, false_pos_rate, num_found_GCs, num_fake_GCs, num_GCs, num_sources, list_of_GCs_not_found, list_of_wrong_GCs, list_of_found_GCs

def quick_eval_fpr(labels, prediction):
    '''
    Returns the true positive and false positive rate.
    '''
    true_pos_rate = np.mean(prediction[labels==1]==1)
    false_pos_rate = np.mean(prediction[labels==0]==1)

    return true_pos_rate, false_pos_rate

def quick_eval_fdr(labels, prediction):
    '''
    Returns the true positive and false detection rate.
    '''
    true_pos_rate = np.mean(prediction[labels==1]==1)
    false_det_rate = np.mean(labels[prediction==1]==0)

    return true_pos_rate, false_det_rate

def eval_with_auc_fdr(labels, pred, thresh):
    '''
    Returns rescaled AUC ROC and ROC curve with FDR instead of FPR.
    '''
    x,y = [], []
    for p in thresh:
        tpr, fdr = quick_eval_fdr(labels, pred>=p)
        y.append(tpr)
        x.append(fdr)
    x[-1] = 0.
    x = np.nan_to_num(x)
    
    order = np.argsort(x)
    x = np.array(x)[order]
    y = np.array(y)[order]

    return auc(x,y)/(2*auc([0,x[-1]], [0,y[-1]])), x, y

def eval_with_auc_fpr(labels, pred, thresh):
    '''
    Returns AUC ROC and ROC curve.
    '''
    x,y = [], []
    for p in thresh:
        tpr, fdr = quick_eval_fpr(labels, pred>=p)
        y.append(tpr)
        x.append(fdr)

    order = np.argsort(x)
    x = np.array(x)[order]
    y = np.array(y)[order]

    return auc(x,y), x, y
