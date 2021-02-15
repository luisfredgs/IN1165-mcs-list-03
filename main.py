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
    
    hardness = args.hardness # easy, all_instances    
    n_estimators = args.n_estimators
    base_learner = Perceptron(max_iter=100)
    print("Using %d classifiers." % n_estimators)
    if args.dataset == 'kc2':
        ds_name, X, Y = dataset_kc2()
    else:
        ds_name, X, Y = dataset_jm1()


    skf = StratifiedKFold(n_splits=n_splits, random_state=seed, shuffle=True)
    pool_classifiers = BaggingClassifier(base_estimator=base_learner, n_estimators=n_estimators, random_state=seed)

    diversity_matrices = []
    results = {'accuracy': [], 'f1_score': [], 'g1_score': [], 'roc_auc':[]}

    def train(train_index, test_index):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]        

        #train_perc = 0.7
        #split_point = int(train_perc*len(train_index))
        # valid_index = train_index[split_point:]
        # train_index = train_index[:split_point]
        # X_train, X_valid, X_test = X[train_index], X[valid_index], X[test_index]
        # y_train, y_valid, y_test = Y[train_index], Y[valid_index], Y[test_index]
        #print("TRAIN:", train_index, "VALID:", valid_index, "TEST:", test_index)

        
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, 
                                            test_size=0.3, random_state=seed)
        pool_classifiers.fit(X_train, y_train)    
        
        validation_data, validation_labels = get_validation_data(X_valid, y_valid, 0.5, hardness=hardness)

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

    output = Parallel(n_jobs=-1, verbose=0, pre_dispatch='1.5*n_jobs')(
        delayed(train)(train_index, test_index) for train_index, test_index in skf.split(X, Y))

    if len(output) == n_splits:
        print(len(output))
        results['accuracy'] = [out['acc'] for out in output]
        results['f1_score'] = [out['f1'] for out in output]
        results['g1_score'] = [out['g1'] for out in output]
        results['roc_auc'] = [out['roc'] for out in output]

        # Results       
        metric_results = pd.DataFrame(results)
        metric_results.to_csv("results/all_folds_%s_%s_%s.csv" % (ds_name, hardness, str(args.dynamic_algorithm)), index=False)

        print(metric_results.mean())
        new_df = metric_results.mean()
        new_df['dataset'] = args.dataset
        new_df['validation'] = hardness
        if args.dynamic_selection == True and args.dynamic_algorithm is not None:
            new_df['dynamic_algorithm'] = args.dynamic_algorithm
        else:
            new_df['dynamic_algorithm'] = '-'
        
        new_df = pd.DataFrame(dict(zip(new_df.index, [[i] for i in new_df.values])))
        print(new_df)
        new_df.to_csv("results/average_folds_%s_%s_%s.csv" % (ds_name, hardness, str(args.dynamic_algorithm)))
        

if __name__ == '__main__':        

    parser = argparse.ArgumentParser(description='Dynamic classifiers selecion')

    parser.add_argument('--hardness', dest='hardness',
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