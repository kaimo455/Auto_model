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
    def __init__(self, data_path, data_type='csv', parse_dates=None, **kwargs):
        print('Preproocessor initializing...')
        # load data
        self.data = self.load_data(data_path, data_type, parse_dates, **kwargs)
        print('Finished.')
    
    def load_data(self, data_path, data_type, parse_dates, **kwargs):
        """
        function to load different type of data. Supporting "csv".
        - parse datetime
        - convert columns to corresponding dtype
        """
        print('Loading data from \'{}\'...'.format(data_path), end=' ')
        # load csv data
        if data_type == 'csv':
            data = pd.read_csv(data_path, **kwargs)
        if data_type == 'xlsx' or data_type == 'xls':
            data = pd.read_excel(data_path, **kwargs)
        # parse datetime
        if parse_dates:
            self.date_cols_ = parse_dates
            print('Parsing datetime...', end=' ')
            for col in parse_dates:
                data[col] = pd.to_datetime(data[col], infer_datetime_format=True)
        print('Finished.')
        return data

    def _get_null_stats(self, axis):
        """
        Calculate null values statistic
        """
        # regard np.inf and empty string as null value
        pd.options.mode.use_inf_as_na = True
        data = self.data.replace('', np.nan)
        # check the missing value statistic
        null_stats = data.isna().sum(axis=axis) / data.shape[axis]
        return null_stats
    
    def drop_cols(self, cols_to_drop, is_return=False):
        """
        Drop some specific columns
        """
        self.cols_to_drop_ = cols_to_drop
        print('Dropping specific column(s)...', end=" ")
        self.data = self.data.drop(cols_to_drop, axis=1)
        print('Finished.')
        if is_return:
            return self.data

    def drop_null(self, col_drop_rate, row_drop_rate, is_return=False):
        """
        There we consider `np.NaN`, `np.inf`, ``''``(empty string), `None`  as null values.
        
        Return `dropped_data` dataframe.
        """
        print(f'Dropping column(s) and row(s) with ratio {col_drop_rate:.2f} and {row_drop_rate:.2f} respectively...', end=' ')
        self.col_drop_rate_ = col_drop_rate
        self.row_drop_rate_ = row_drop_rate
        # deep copy data to dropped_data
        dropped_data = self.data.copy(deep=True)
        # drop columns
        col_null_stats = self._get_null_stats(axis=0)
        col_idx = col_null_stats >= col_drop_rate
        dropped_data.drop(col_null_stats.index[col_idx], axis=1, inplace=True)
        self.data = dropped_data
        # drop rows
        row_null_stats = self._get_null_stats(axis=1)
        row_idx = row_null_stats >= row_drop_rate
        dropped_data.drop(row_null_stats.index[row_idx], axis=0, inplace=True)
        self.data = dropped_data
        print('Finished.')
        if is_return:
            return self.data

    def fill_na(self, fill_na_method, is_return=False):
        """
        Fill NaN cells.
        
        Return `fillna_data` dataframe
        """
        print('Fill null values...', end=' ')
        # deep copy data to fillna_data
        fillna_data = self.data.copy(deep=True)
        fillna_data.fillna(method=fill_na_method, inplace=True)
        self.data = fillna_data
        print('Finished.')
        if is_return:
            return self.data
    
    def feature_encoding(self, features_to_enc, is_return=False):
        """
        Encoding object/string variables, here we need to avoid NaN values so just using fillna_data
        """
        print('Feature encoding...', end=' ')
        # encode each feature
        self.les_ = []
        for feat in features_to_enc:
            # ignore NaN
            _df = self.data[feat].copy(deep=True)
            le = LabelEncoder()
            _df.loc[~_df.isna()] = le.fit_transform(_df.loc[~_df.isna()])
            self.data[feat] = _df.astype('category')
            # save encoder
            self.les_.append(le)
        print('Finished.')
        if is_return:
            return self.data
        
    def convert_dtypes(self, int_cols=None, float_cols=None, cate_cols=None, bool_cols=None, is_return=False):
        # parse dtype
        if int_cols:
            self.int_cols_ = int_cols
            self.data[int_cols] = self.data[int_cols].astype(np.int)
        if float_cols:
            self.float_cols_ = float_cols
            self.data[float_cols] = self.data[float_cols].astype(np.float)
        if cate_cols:
            self.cate_cols_ = cate_cols
            self.data[cate_cols] = self.data[cate_cols].astype('category')
        if bool_cols:
            self.bool_cols_ = bool_cols
            self.data[bool_cols] = self.data[bool_cols].astype('bool')
        if is_return:
            return self.data
    
    def dim_reduction_pca(self, features_to_pca, evr_threshold, is_return=False):
        """
        Perform PCA to all features if `features_to_pca` is None, else only on specific features.
        """
        print('Dimensionality reduction using PCA...', end=" ")
        if evr_threshold <= 0 or evr_threshold >= 1:
            raise ValueError('evr_threshold must be in range [0, 1]')
        data = self.data[features_to_pca] if features_to_pca else self.data
        self.pca_ = PCA(n_components=None).fit(data)
        pca_data = self.pca_.transform(data)
        evrs_cumsum = np.cumsum(self.pca_.explained_variance_ratio_)
        for idx, cumsum in enumerate(evrs_cumsum):
            if cumsum >= evr_threshold:
                self.data = pca_data[:, :idx]
        print('Finished.')
        if is_return:
            return self.data
        
    def scaling(self, cate_cols, n_cate_to_scale=3, is_return=False):
        # numeric columns
        int_cols = list(self.data.dtypes.index[self.data.dtypes == np.int].values)
        float_cols = list(self.data.dtypes.index[self.data.dtypes == np.float].values)
        num_cols = int_cols + float_cols
        # category columns
        cat_cols = [col for col in cate_cols if len(self.data[col].cat.categories) > n_cate_to_scale]
        self.data[num_cols+cat_cols] = StandardScaler().fit_transform(self.data[num_cols+cat_cols])
        if is_return:
            return self.data
    
    @staticmethod
    def make_folds(X, y=None, n_splits=5, strategy=None, group=None, shuffle=True, random_state=None):
        ### strategy = None / 'stratified' / 'group'

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
    def make_shuffle(X, y=None, n_splits=5, test_size=0.25, train_size=0.75, strategy=None, group=None, random_state=None):
        ### strategy = None / 'stratified' / 'group'

        # stratified strategy
        if strategy == "stratified":
            spliter = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, train_size=train_size, random_state=random_state)
            if y is None:
                raise Exception('Please provide y parameter.')
            else:
                idx_generator = spliter.split(X, y=y)
        # group strategy
        elif strategy == 'group':
            spliter = GroupShuffleSplit(n_splits=n_splits, test_size=test_size, train_size=train_size, random_state=random_state)
            if group is None:
                raise Exception('Please provide group parameter.')
            else:
                idx_generator = spliter.split(X, y=y, groups=group)
        # not specific strategy
        else:
            spliter = ShuffleSplit(n_splits=n_splits, test_size=test_size, train_size=train_size, random_state=random_state)
            idx_generator = spliter.split(X, y=y)
        return idx_generator