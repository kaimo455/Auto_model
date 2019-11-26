import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold
from sklearn.model_selection import LeaveOneOut, LeavePOut, LeaveOneGroupOut, LeavePGroupsOut
from sklearn.model_selection import ShuffleSplit, GroupShuffleSplit, StratifiedShuffleSplit


class Preprocessor:
    """
    Data preprocessor:
    
    - Loading data & parse dates -> data
    - Drop specific columns/features -> data
    - Drop columns/features by null ratio -> data_dropped 
    - Drop rows/samples by null ratio -> data_dropped 
    - Filling null values -> data_fillna
    - Encoding features -> data_feat_enc
    - PCA dimensionality reduction: -> data_pca
      what features to PCA and evr thresholdto choose number of components
    - Scaling
    """

    def __init__(self):
        """
        Initialize Preprocessor instance
        """
        self.feature_encoders = []
        print('Preprocessor initialized.')

    @staticmethod
    def load_data(data_path, data_type, parse_dates=None, **kwargs):
        """
        function to load different type of data. Supporting "csv", "xlsx", "xls".
        - parse datetime
        - convert columns to corresponding dtype
        """
        print(f'Loading data from \'{data_path}\'...', end=' ')
        # load csv data
        if data_type == 'csv':
            data = pd.read_csv(data_path, **kwargs)
        if data_type == 'xlsx' or data_type == 'xls':
            data = pd.read_excel(data_path, **kwargs)
        # parse datetime
        if parse_dates:
            print('Parsing datetime...', end=' ')
            for col in parse_dates:
                data[col] = pd.to_datetime(data[col], infer_datetime_format=True)
        print('Finished.')
        return data

    @staticmethod
    def _get_null_stats(features, axis, nulls=None):
        """
        Calculate null values statistic
        if nulls are specified, then elements in nulls list are replace with np.NaN, then calculate stats
        Args:
            features (DataFrame): feature DataFrame
            axis (int): 0 - column stats, 1 - row stats
            nulls (list or None): what value(s) is null value

        Returns: null statistics series

        """
        if nulls:
            for null_val in nulls:
                features.replace(null_val, np.nan)
        null_stats = features.isna().sum(axis=axis) / features.shape[axis]
        return null_stats

    @staticmethod
    def drop_cols(features, cols_to_drop):
        """
        Drop some specific columns
        """
        print('Dropping specific column(s)...', end=" ")
        features = features.drop(cols_to_drop, axis=1)
        print('Finished.')
        return features

    def drop_null(self, features, col_drop_rate, row_drop_rate):
        """
        There we consider `np.NaN`, `np.inf`, ``''``(empty string), `None`  as null values.
        Args:
            features ():
            col_drop_rate ():
            row_drop_rate ():

        Returns:row and column dropped DataFrame, and row dropped index

        """
        print(f'Dropping column(s) and row(s) with ratio {col_drop_rate:.2f} and {row_drop_rate:.2f} respectively...',
              end=' ')
        # deep copy data to dropped_data
        dropped_data = features.copy(deep=True)
        # drop columns and rows
        col_null_stats = self._get_null_stats(features, axis=0)
        row_null_stats = self._get_null_stats(features, axis=1)
        col_drop_index = col_null_stats.index[col_null_stats >= col_drop_rate]
        row_drop_index = row_null_stats.index[row_null_stats >= row_drop_rate]
        dropped_data.drop(col_drop_index, axis=1, inplace=True)
        dropped_data.drop(row_drop_index, axis=0, inplace=True)
        print('Finished.')
        return dropped_data, row_drop_index

    @staticmethod
    def fill_na(features, fill_na_method):
        """
        Fill NaN cells.
        
        Return `fillna_data` dataframe
        """
        print('Fill null values...', end=' ')
        # deep copy data to fillna_data
        fillna_data = features.copy(deep=True)
        fillna_data.fillna(method=fill_na_method, inplace=True)
        print('Finished.')
        return fillna_data

    def feature_encoding(self, features, features_to_enc):
        """
        Encoding object/string variables, here we need to avoid NaN values so just using fillna_data
        """
        print('Feature encoding...', end=' ')
        # encode each feature
        for feat in features_to_enc:
            _df = features[feat].copy(deep=True)
            le = LabelEncoder()
            # ignore NaN
            _df.loc[~_df.isna()] = le.fit_transform(_df.loc[~_df.isna()])
            features[feat] = _df.astype('category')
            self.feature_encoders.append(le)
        print('Finished.')
        return features

    @staticmethod
    def convert_dtypes(features, int_cols=None, float_cols=None, cate_cols=None, bool_cols=None):
        # parse dtype
        if int_cols:
            features[int_cols] = features[int_cols].astype(np.int)
        if float_cols:
            features[float_cols] = features[float_cols].astype(np.float)
        if cate_cols:
            features[cate_cols] = features[cate_cols].astype('category')
        if bool_cols:
            features[bool_cols] = features[bool_cols].astype('bool')
        return features

    @staticmethod
    def dim_reduce_pca(features, evr_threshold):
        """
        Perform PCA to all features if `features_to_pca` is None, else only on specific features.
        """
        print('Dimensionality reduction using PCA...', end=" ")
        if evr_threshold <= 0 or evr_threshold >= 1:
            raise ValueError('evr_threshold must be in range [0, 1]')

        pca = PCA(n_components=None).fit(features)
        pca_data = pca.transform(features)
        evrs_cumsum = np.cumsum(pca.explained_variance_ratio_)
        for idx, cumsum in enumerate(evrs_cumsum):
            if cumsum >= evr_threshold:
                features = pca_data[:, :idx]
        print('Finished.')
        return features

    @staticmethod
    def make_folds(X, y=None, n_splits=5, strategy=None, group=None, shuffle=True, random_state=None):
        """ strategy = None / 'stratified' / 'group' """

        # stratified strategy
        if strategy == "stratified":
            spliter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            if y is None:
                raise Exception('Please provide y parameter.')
            else:
                idx_generator = spliter.split(X, y=y)
        # group strategy
        elif strategy == 'group':
            spliter = GroupKFold(n_splits=n_splits)
            if group is None:
                raise Exception('Please provide group parameter.')
            else:
                idx_generator = spliter.split(X, y=y, groups=group)
        # not specific strategy
        else:
            spliter = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
            idx_generator = spliter.split(X, y=y)
        return idx_generator

    @staticmethod
    def make_leave_out(X, y=None, p=5, strategy=None, group=None):
        ### strategy = None / 'group'

        # group strategy
        if strategy == 'group':
            spliter = LeaveOneGroupOut() if p == 1 else LeavePGroupsOut(p)
            if group is None:
                raise Exception('Please provide group parameter.')
            else:
                idx_generator = spliter.split(X, y=y, groups=group)
        # not specific strategy
        else:
            spliter = LeaveOneOut() if p == 1 else LeavePOut(p)
            idx_generator = spliter.split(X, y=y, groups=group)
        return idx_generator

    @staticmethod
    def make_shuffle(X, y=None, n_splits=5, test_size=0.25, train_size=0.75, strategy=None, group=None,
                     random_state=None):
        """
        strategy =
        Args:
            X ():
            y ():
            n_splits ():
            test_size (float):
            train_size (float):
            strategy (str): can be None , 'stratified' or 'group'
            group ():
            random_state (int):

        Returns:

        """

        # stratified strategy
        if strategy == "stratified":
            spliter = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, train_size=train_size,
                                             random_state=random_state)
            if y is None:
                raise Exception('Please provide y parameter.')
            else:
                idx_generator = spliter.split(X, y=y)
        # group strategy
        elif strategy == 'group':
            spliter = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, train_size=train_size,
                                        random_state=random_state)
            if group is None:
                raise Exception('Please provide group parameter.')
            else:
                idx_generator = spliter.split(X, y=y, groups=group)
        # not specific strategy
        else:
            spliter = ShuffleSplit(n_splits=n_splits, test_size=test_size, train_size=train_size,
                                   random_state=random_state)
            idx_generator = spliter.split(X, y=y)
        return idx_generator
