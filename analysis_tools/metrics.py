import numpy as np

def accuracy_metrics(labels, preds):

    from sklearn.metrics import log_loss, accuracy_score

    cce = log_loss(labels, preds[:, -1])
    avg_accuracy = accuracy_score(labels, np.round(preds[:, -1]))

    print(f'Cross Entropy Loss: {cce}')
    print(f'Average Accuracy: {avg_accuracy}')

def roc_metrics(labels, preds):

    from sklearn.metrics import roc_auc_score, roc_curve

    fpr, tpr, thresholds = roc_curve(labels, preds[:, -1])
    auc = roc_auc_score(labels, preds[:, -1])
    
    return fpr, tpr, auc

def bkg_rej(labels, preds, eff = 0.5):

    fpr, tpr, auc = roc_metrics(labels, preds)
    
    idx = next(idx for idx, v in enumerate(tpr) if v>eff)
    rej = 1/fpr[idx]

    print(f'Background Rejection at {eff*100}% Signal Efficiency: {rej}')

    return rej

def confusion_matrices(labels, preds):
    
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true=labels, y_pred = np.round(preds[:, -1]))

    return cm