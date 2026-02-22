import numpy as np
import matplotlib.pyplot as plt
import mplhep as mh
plt.style.use(mh.style.CMS)

def accuracy_metrics(labels, preds):

    from sklearn.metrics import log_loss, accuracy_score

    cce = log_loss(labels, preds[:, -1], labels=[0, 1])
    avg_accuracy = accuracy_score(labels, np.round(preds[:, -1]))

    return cce, avg_accuracy

def roc_metrics(labels, preds):

    from sklearn.metrics import roc_auc_score, roc_curve

    fpr, tpr, thresholds = roc_curve(labels, preds[:, -1], pos_label=1)
    auc = roc_auc_score(labels, preds[:, -1], labels=[0, 1])
    
    return fpr, tpr, auc

def bkg_rej(labels, preds, eff = 0.5):

    fpr, tpr, auc = roc_metrics(labels, preds, labels=[0, 1])
    
    idx = next(idx for idx, v in enumerate(tpr) if v>eff)
    rej = 1/fpr[idx]


    return rej

def confusion_matrices(labels, preds):
    
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true=labels, y_pred = np.round(preds[:, -1]), labels=[0, 1])

    return cm