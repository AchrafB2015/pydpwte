def generate_data_SUPPORT():

    """ Generate and preprocess SUPPORT2 dataset
    """

    DATA                   = pd.read_csv('support2.csv')
    quantiles, Y_quantiles = [], []
    for col in list(DATA.columns):
        if(DATA[col].isna().sum()>300):
            DATA = DATA.drop([col],axis=1)
    col_time    = 'd.time'
    col_event   = 'death'
    DATA        = DATA.dropna(axis=0, how='any')
    DATA        = pd.get_dummies(DATA, columns=['dnr', 'ca', 'dzclass', 'dzgroup', 'sex', 'race'])
    corr_matrix = DATA.corr().abs()
    upper       = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find index of feature columns with correlation greater than 0.8
    to_drop        = [column for column in upper.columns if (any(upper[column] > 0.8)) ]
    DATA           = DATA.drop(DATA[to_drop], axis=1)
    mx             = max(DATA['d.time'])
    DATA           = DATA.sample(frac=1)
    DATA['d.time'] = DATA['d.time'].apply(lambda x: x/mx)
    X              = np.array(DATA.drop(['d.time', 'death'], axis = 1)).astype(np.float32)
    n_cols         = X.shape[1]
    X              = Normalize_InputData(X)
    Y              = np.array(DATA[['d.time', 'death']]).astype(np.float32)

    return(X, Y)

def generate_METABRIC_Data():
    Y                            = pd.read_csv('label.csv')
    DATA                         = pd.read_csv('cleaned_features_final.csv')
    DATA[['event_time','label']] = Y


    mx          = max(DATA['event_time'])
    DATA['event_time'] = DATA['event_time'].apply(lambda x : x/mx)

    X, Y  = np.array(DATA.drop(['event_time', 'label'], axis = 1)).astype(np.float32), np.array(DATA[['event_time', 'label']]).astype(np.float32)
    n_cols = X.shape[1]

    size       = len(DATA)
    train_size = int(0.8*size)

    train_indices  = np.random.choice(range(size),train_size, replace=False)
    X        = X[train_indices,:]
    Y        = Y[train_indices,:]

    return(X,Y)