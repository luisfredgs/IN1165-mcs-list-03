from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import Perceptron
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from deslib.util.diversity import double_fault

# Dynamic classifier selection OLA, LCA, MCB
from deslib.dcs.ola import OLA
from deslib.dcs.lca import LCA
from deslib.dcs.mcb import MCB

# Dynamic ensemble selection KNORA-E e KNORA-U
from deslib.des.knora_e import KNORAE
from deslib.des.knora_u import KNORAU

from sklearn.metrics import roc_auc_score
import seaborn as sns
import numpy as np
import time
from utils import *
from datasets import *
import argparse

seed = 100000
np.random.seed(seed)
n_splits = 10

def run(args):
    
    hardnesses = args.hardnesses # easy, all_instances    
    n_estimators = args.n_estimators
    base_learner = Perceptron(max_iter=100)
    print("Using %d classifiers." % n_estimators)
    if args.dataset == 'kc2':
        ds_name, X, Y = dataset_kc2()
    else:
        ds_name, X, Y = dataset_pc1()


    skf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
    pool_classifiers = BaggingClassifier(base_estimator=base_learner, n_estimators=n_estimators, random_state=seed)

    diversity_matrices = []
    results = {'accuracy': [], 'f1_score': [], 'g1_score': [], 'roc_auc':[]}

    def train(train_index, test_index):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]        

        # train classifiers
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, 
                                            test_size=0.2, random_state=seed)
        pool_classifiers.fit(X_train, y_train)    
        
        validation_data, validation_labels = get_validation_data(X_valid, y_valid, 0.5, hardnesses=hardnesses)


        dynamic_selection_algorithm = None
        try:        
            if args.dynamic_selection == True and args.dynamic_algorithm is None:
                raise ValueError('Dynamic selection requires you provide an algorithm.')
            elif args.dynamic_selection == True and args.dynamic_algorithm is not None:
                if args.dynamic_algorithm == 'ola':
                    dynamic_selection_algorithm = OLA(pool_classifiers, random_state=seed)                    
                elif args.dynamic_algorithm == 'lca':
                    dynamic_selection_algorithm = LCA(pool_classifiers, random_state=seed)
                elif args.dynamic_algorithm == 'mcb':
                    dynamic_selection_algorithm = MCB(pool_classifiers, random_state=seed)
                elif args.dynamic_algorithm == 'knorau':
                    dynamic_selection_algorithm = KNORAU(pool_classifiers, random_state=seed)
                elif args.dynamic_algorithm == 'kne':
                    dynamic_selection_algorithm = KNORAE(pool_classifiers, random_state=seed)
                
                dynamic_selection_algorithm.fit(validation_data, validation_labels)
                preds = dynamic_selection_algorithm.predict(X_test)
            else:
                # Static combination by voting
                preds = voting(X_test, pool_classifiers)
        except Exception as error:
            raise error

        
        acc = get_accuracy_score(y_test, preds)
        g1 = get_g1_score(y_test, preds, average='macro')
        f1 = get_f1_score(y_test, preds)
        roc = roc_auc_score(y_test, preds, average='macro')

        return dict(f1=f1, g1=g1, 
                acc=acc,
                roc=roc)    

    output = Parallel(n_jobs=-1, verbose=100, pre_dispatch='1.5*n_jobs')(
        delayed(train)(train_index, test_index) for train_index, test_index in skf.split(X, Y))

    if len(output) == n_splits:
        print(len(output))
        results['accuracy'] = [out['acc'] for out in output]
        results['f1_score'] = [out['f1'] for out in output]
        results['g1_score'] = [out['g1'] for out in output]
        results['roc_auc'] = [out['roc'] for out in output]

        # Results       
        metric_results = pd.DataFrame(results)
        #metric_results.to_csv("results/%s_summary_metrics_all_folds_%s_%s.csv" % (ds_name, hardnesses, label_pruning_save), index=False)

        print(metric_results.mean())
        #metric_results.mean().to_csv("results/%s_summary_metrics_average_folds_%s_%s.csv" % (ds_name, hardnesses, label_pruning_save))
        

if __name__ == '__main__':        

    parser = argparse.ArgumentParser(description='Pruning classifiers')

    parser.add_argument('--hardnesses', dest='hardnesses',
                    default='all_instances', help='Instance hardness of validation set')

    parser.add_argument('--dynamic-selection', dest='dynamic_selection',
                    action='store_true', help='When using dynamic combining')

    parser.add_argument('--dynamic-algorithm', dest='dynamic_algorithm',
                    default=None, help='dynamic classifier selection algorithms (DCS).')
    
    parser.add_argument('--dataset', dest='dataset',
                    default="kc2", help='Dataset')
    
    parser.add_argument('--n_estimators', dest='n_estimators',
                    default=100, type=int, help='Number of base classifiers')

    args = parser.parse_args()

    run(args)