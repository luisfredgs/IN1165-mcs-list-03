from math import sqrt
from deslib.util.instance_hardness import kdn_score
from deslib.util.diversity import double_fault
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
import numpy as np

def get_validation_data(X, y, threshold = 0.5, hardness='hard'):
    score, kdn_neighbors = kdn_score(X, y, k=5)
    if hardness == 'hard':
        indices = np.where(score>threshold)
    elif hardness == 'easy':
        indices = np.where(score<=threshold)
    else:
        # original data with all their instances
        indices = np.where(score>=0.0) 
    return X[indices], y[indices]


def voting(X, pool_classifiers):
    preds = np.array([estimator.predict(X) for estimator in pool_classifiers.estimators_])

    # also one can try: stats.mode(preds)[0][0]
    maj_votes = np.apply_along_axis(lambda x:
                        np.argmax(np.bincount(x,weights=None)),
                        axis=0,
                        arr=preds.astype('int'))
    return maj_votes

# Metrics

def get_accuracy_score(y, predictions):
    accuracy = accuracy_score(y, predictions)
    return accuracy

def get_f1_score(y_true, predictions):
    return f1_score(y_true, predictions, average='macro')

def get_g1_score(y, predictions, average):
	precision = precision_score(y, predictions, average=average)
	recall = recall_score(y, predictions, average=average)
	return sqrt(precision*recall)