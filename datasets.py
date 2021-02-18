import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy.io import arff
from sklearn.preprocessing import StandardScaler

def dataset_kc2():
    """KC2/software defect prediction
    Data comes from McCabe and Halstead features extractors of source code.  These features were defined in the 70s in an attempt
    to objectively characterize code features that are associated with software quality.  The nature of association is under dispute.    
    """
    data = arff.loadarff("data/kc2.arff")
    df = pd.DataFrame(data[0])
    #print(df.info())

    le = preprocessing.LabelEncoder()
    le.fit(df['problems'])

    X = df.drop(['problems'], axis=1).to_numpy()

    sc = StandardScaler()
    X = sc.fit_transform(X)
    Y = le.transform(df['problems']) 
    return 'kc2', X, Y


def dataset_jm1():
    """PC1 Software defect prediction"""

    data = arff.loadarff("data/jm1.arff")
    df = pd.DataFrame(data[0])    
    
    #df.fillna(0.0, inplace=True)    
    df['uniq_Op'].fillna(value=df['uniq_Op'].mean(), inplace=True)
    df['uniq_Opnd'].fillna(value=df['uniq_Opnd'].mean(), inplace=True)
    df['total_Op'].fillna(value=df['total_Op'].mean(), inplace=True)
    df['total_Opnd'].fillna(value=df['total_Opnd'].mean(), inplace=True)
    df['branchCount'].fillna(value=df['branchCount'].mean(), inplace=True)

    le = preprocessing.LabelEncoder()
    le.fit(df['defects'])

    X = df.drop(['defects'], axis=1).to_numpy()

    sc = StandardScaler()
    X = sc.fit_transform(X)
    Y = le.transform(df['defects']) 
    return 'jm1', X, Y