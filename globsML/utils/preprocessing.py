import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

### DEFAULT PARAMETERS

# dictionary summarising how features are rescaled
default_rescaling = {
    'orientation': 'maxdiv',
    'min_value_z': 'meandiv',
    'min_value': 'meandiv',
    'm3_g': 'norm',
    'CI_z': 'norm',
    'CI_g': 'norm',
    'm3_z': 'norm',
    'm4_g': 'norm',
    'CI4_z': 'norm',
    'CI4_g': 'norm',
    'm4_z': 'norm',
    'm5_g': 'norm',
    'CI5_z': 'norm',
    'CI5_g': 'norm',
    'm5_z': 'norm',
    'max_value': 'p95div',
    'max_value_z': 'p95div',
    'segment_flux': 'p90div',
    'segment_flux_z': 'p90div',
    'orientation_z': 'maxdiv',
    'semiminor_sigma_z': 'maxdiv',
    'semiminor_sigma': 'maxdiv',
    'colour': 'norm',
    'area_z': 'meandiv',
    'area': 'meandiv',
    'semimajor_sigma': 'p99div',
    'semimajor_sigma_z': 'p99div',
    'eccentricity': 'nothing',
    'eccentricity_z': 'nothing'}

# set of tabular features that are dropped from the data
default_columns_to_drop = set(['fwhm',
                               'e_fwhm',
                                'cluster',
                               'kron_flux',
                               'kron_flux_z',
                               'Unnamed: 0',
                               'pGC',
                               'label_z',
                               'matched',
                               'HST_ID',
                               'galaxy',
                               'ID',
                               'label',
                               'xcentroid',
                               'ycentroid',
                               'sky_centroid.ra',
                               'sky_centroid.dec',
                               'xcentroid_z',
                               'ycentroid_z',
                               'sky_centroid.ra_z',
                               'sky_centroid.dec_z',
                               'type'])

# the following features are investigated when checking for NaN values
default_feat_with_NaN_list = ['CI4_g', 'CI4_z', 'm4_g', 'm4_z', 'CI5_g', 'CI5_z', 'm5_g', 'm5_z', 'colour', 'm3_z', 'm3_g', 'CI_g', 'CI_z']

#### CLASS DEFINITIONS

class Scaler:
    '''
    Scaler object for rescaling tabular features.
    
    Methods: fit, transform, reverse_transform.
    '''
    def __init__(self, mode):
        self.scale = None
        self.subtract = None
        self.mode = mode

    def fit(self, data):
        '''
        Gets parameters for rescaling from 
        the training data.
        '''
        if self.mode == 'maxdiv':
            self.scale = np.max(data)
        elif self.mode == 'meandiv':
            self.scale = np.mean(data)
        elif self.mode == 'norm':
            self.scale = np.std(data)
            self.subtract = np.mean(data)
        elif self.mode[0] == 'p' and len(self.mode) == 6:
            self.scale = np.percentile(data, int(self.mode[1:3]))

    def transform(self, data):
        '''
        Rescales data.
        '''
        if self.mode == 'norm':
            data = data - self.subtract
        data = data / self.scale
        return data

    def reverse_transform(self, data):
        '''
        Undoes the rescaling.
        '''
        data = data * self.scale
        if self.mode == 'norm':
            data = data + self.subtract
        return data


#### FUNCTION DEFINITIONS

def replace_NaN(df_train, dfs_test, feat_list = default_feat_with_NaN_list):
    '''
    Deals with NaN values in the training and testing dataframe.
    
    For training data (df_train), sources that contain NaN as an entry of 
    one of the features in feat_list are dropped.
    For testing data (dfs_test), NaN values are replaced by the according 
    median value from the training data.
    '''
    # drop sources with NaN entries
    num_sources = len(df_train)
    for feat in feat_list:
        df_train = df_train[df_train[feat].notna()]
    num_sources_after_drop = len(df_train)
    print('Number of sources in training split after dropping rows with NaN as CI/m/color: {}'.format(num_sources_after_drop))
    print('{} sources have been dropped.'.format(num_sources-num_sources_after_drop))

    # get median value of tabular features from the training data
    median_values = {}
    for feat in feat_list:
        median_values[feat] = df_train[feat].median()
    
    # replace NaN values in the test dataframe with median values from the training data
    for feat in feat_list:
        dfs_test[feat][dfs_test[feat].isna()] = median_values[feat]
    print('NaN values in testing data have been replaced with the correspnding median value observed in the training split')

    return df_train, dfs_test

def select_galaxies(data, test_galaxies = set(['FCC47', 'FCC119'])):
    '''
    Splits a dataframe with tabular features into a training and test dataframe by
    selecting different galaxies for training and testing.
    
    test_galaxies: set of galaxies used for testing only.
    '''
    galaxies = set(data.galaxy.unique())
    remaining_galaxies = galaxies.difference(test_galaxies)

    df_train = data[data['galaxy'].isin(remaining_galaxies)]
    dfs_test = data[data['galaxy'].isin(test_galaxies)]

    return df_train, dfs_test

def select_sources(data, test_size=0.2, random_seed = 42424242):
    '''
    Splits a dataframe with tabular features randomly into a training and test dataframe.
    
    test_size: fraction of data that will be used for testing.
    '''
    df_train, dfs_test = train_test_split(data, test_size=test_size, random_state=random_seed)

    return df_train, dfs_test

def create_data_dict(data, df_train, dfs_test, eval_size = 0.05, seed = 42424242):
    '''
    Turns the dataframes into dictionaries containing data in a uniform format.
    
    data: original dataframe (before splitting)
    df_train: processed training dataframe
    dfs_test: processed testing dataframe
    eval_size: percentage of training data that will be used for validation only.
    
    Format of the created dataset:
    df: dictionary with keys test, train and eval, containing the testing,
    training and validation data, respectively.
    
    df[key][x]:
    x=inputs: array containing tabular features, with rows corresponding to sources,
    x=labels: list with labels (source is GC = 1, source is non-GC = 0) for each source,
    x=probs: pGC of sources,
    x=feature_name: name of feature for each column in inputs,
    x=galaxy: galaxy names of sources,
    x=ID: ID of source.
    
    '''
    # drop unwanted tabular features
    columns_to_drop = default_columns_to_drop
    columns_to_keep = np.sort(list(set(data.columns).difference(columns_to_drop)))

    # final dictionary containing trianing, validation and testing split
    df = {'test': {}, 'train': {}, 'eval': {}}

    # split the training data further into training and validation split
    train, val = train_test_split(df_train, test_size=eval_size, random_state=seed)
    traindata = train[columns_to_keep]
    evaldata = val[columns_to_keep]

    # fill in all data for training
    df['train']['inputs'] = traindata.values
    df['train']['labels'] = np.array(train['pGC'].values >= 0.5, dtype=int)
    df['train']['probs'] = train['pGC'].values
    df['train']['feature_name'] = list(traindata.columns.values)
    df['train']['galaxy'] = train['galaxy'].values
    df['train']['ID'] = train['ID'].values

    # fill in all data for validation
    df['eval']['inputs'] = evaldata.values
    df['eval']['labels'] =  np.array(val['pGC'].values >= 0.5, dtype=int)
    df['eval']['probs'] = val['pGC'].values
    df['eval']['feature_name'] = list(evaldata.columns.values)
    df['eval']['galaxy'] = val['galaxy'].values
    df['eval']['ID'] = val['ID'].values

    # fill in all data for testing
    testdata = dfs_test[columns_to_keep]
    df['test']['inputs'] = testdata.values
    df['test']['feature_name'] = list(testdata.columns.values)
    df['test']['probs'] = dfs_test['pGC'].values
    df['test']['labels'] = np.array(dfs_test['pGC'].values >= 0.5, dtype=int)
    df['test']['galaxy'] = np.array(dfs_test['galaxy'].values)
    df['test']['ID'] = np.array(dfs_test['ID'].values)

    return df

def rescale_data(df, howto = default_rescaling):
    '''
    Convenience function for rescaling data.
    Uses the data dictionary returned by create_data_dict().
    
    Returns both the rescaled data dictionary as well as the scaler object.
    '''
    scalers = {}
    # for all tabular features
    for i in tqdm(range(len(df['train']['feature_name']))):
        # get name of tabular feature
        variable = df['train']['feature_name'][i]
        # get rescale method for feature
        method = howto[variable]
        if method == 'nothing':
            print('{} will not be transformed. Skipped.'.format(variable))
        else:
            sc = Scaler(method)
            # fit scaler on training data
            sc.fit(df['train']['inputs'][:,i])
            # use the fit to rescale all data splits
            df['train']['inputs'][:,i] = sc.transform(df['train']['inputs'][:,i])
            df['eval']['inputs'][:,i] = sc.transform(df['eval']['inputs'][:,i])
            df['test']['inputs'][:,i] = sc.transform(df['test']['inputs'][:,i])
            scalers[variable] = sc

    return df, scalers