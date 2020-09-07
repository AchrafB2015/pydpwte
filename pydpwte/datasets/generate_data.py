import os
import pandas as pd
import numpy  as np
from utils.preprocess import normalize_input_data

def generate_data_SUPPORT():

    """ Generate and preprocess SUPPORT2 dataset
    """
    np.random.seed(31415)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path = os.path.abspath(os.path.join(dir_path, '', 'support2.csv'))
    print("path:{}".format(path))
    data                   = pd.read_csv(path, index_col=0)
    quantiles, Y_quantiles = [], []
    for col in list(data.columns):
        if(data[col].isna().sum()>300):
            data = data.drop([col],axis=1)
    col_time    = 'd.time'
    col_event   = 'death'
    data        = data.dropna(axis=0, how='any')
    data        = pd.get_dummies(data, columns=['dnr', 'ca', 'dzclass', 'dzgroup', 'sex', 'race'])
    corr_matrix = data.corr().abs()
    upper       = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.8
    to_drop        = [column for column in upper.columns if (any(upper[column] > 0.8)) ]
    data           = data.drop(data[to_drop], axis=1)
    mx             = max(data['d.time'])
    data           = data.sample(frac=1)
    data['d.time'] = data['d.time'].apply(lambda x: x/mx)
    X              = np.array(data.drop(['d.time', 'death'], axis = 1)).astype(np.float32)
    n_cols         = X.shape[1]
    X              = normalize_input_data(X)
    Y              = np.array(data[['d.time', 'death']]).astype(np.float32)

    return(X, Y)

def generate_METABRIC_Data():
    """ Generate METABRIC dataset"""
    np.random.seed(31416)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    path_label = os.path.abspath(os.path.join(dir_path, '', 'label.csv'))
    path_features = os.path.abspath(os.path.join(dir_path, '', 'cleaned_features_final.csv'))
    print("path of label :{}".format(path_label))
    print("path of features :{}".format(path_features))
    data  = pd.read_csv(path_features, index_col=0)
    Y     = pd.read_csv(path_label, index_col=0)
    data[['event_time','label']] = Y


    mx          = max(data['event_time'])
    data['event_time'] = data['event_time'].apply(lambda x : x/mx)

    X, Y  = np.array(data.drop(['event_time', 'label'], axis = 1)).astype(np.float32), np.array(data[['event_time', 'label']]).astype(np.float32)
    n_cols = X.shape[1]

    size       = len(DATA)
    train_size = int(0.8*size)

    train_indices  = np.random.choice(range(size),train_size, replace=False)
    X        = X[train_indices,:]
    Y        = Y[train_indices,:]

    return(X,Y)