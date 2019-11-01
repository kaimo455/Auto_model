from . import preprocessor, lightgbmOptimizer, logisticOptimizer, randomforestOptimizer, svmOptimizer

__all__ = ['preprocessor', 'lightgbmOptimizer', 'logisticOptimizer', 'randomforestOptimizer', 'svmOptimizer']

PARAMS_EXAMPLES = {
    'lightgbm': {
        'base_params': {'task': 'train', \
            'objective': 'binary', \
            'tree_learner': 'serial', \
            'num_threads': 4, \
            'device_type': 'cpu', \
            'seed': 1213, \
            'bagging_seed': 42, \
            'feature_fraction_seed': 3, \
            'first_metric_only': True, \
            'max_delta_step': 0, \
            'bin_construct_sample_cnt': 200000, \
            'histogram_pool_size': -1, \
            'is_unbalance': True, \
            'metric': 'auc,binary_logloss,binary_error', \
            'metric_freq': 1},
        'cat_params': {'boosting': ['gbdt']},
        'int_params': {'num_leaves': (2, 1024, 8), \
            'max_depth': (1, 100, 1), \
            'min_data_in_leaf': (2, 500, 4), \
            'bagging_freq': (0, 100, 1), \
            'min_data_per_group': (100, 500, 10), \
            'max_cat_threshold': (16, 256, 2), \
            'max_cat_to_onehot': (1, 100, 1), \
            'max_bin': (127, 511, 2), \
            'min_data_in_bin': (3, 128, 8)},
        'float_params': {'min_sum_hessian_in_leaf': (0, 0.1), \
            'bagging_fraction': (0.1, 1), \
            'pos_bagging_fraction': (0.1, 1), \
            'neg_bagging_fraction': (0.1, 1), \
            'feature_fraction': (0.1, 1), \
            'feature_fraction_bynode': (0.1, 1), \
            'lambda_l1': (0, 500), \
            'lambda_l2': (1000, 4000), \
            'sigmoid': (0.1, 500), \
            'cat_l2': (10, 1000), \
            'cat_smoth': (10, 1000), \
            'min_gain_to_split': (0, 100)}
    },
    'randomforest': {
        'base_params': {'bootstrap': True, \
            'oob_score': False, \
            'n_jobs': 4, \
            'random_state': 1213, \
            'warm_start': False, \
            'class_weight': 'balanced_subsample'},
        'cat_params': {'criterion': ['gini', 'entropy']},
        'int_params': {'n_estimators': (10, 500, 10), \
            'max_depth': (1, 50, 1), \
            'min_samples_split': (2, 500, 2), \
            'min_samples_leaf': (1, 250, 1), \
            'max_leaf_nodes': (2, 1024, 4)},
        'float_params': {'min_weight_fraction_leaf': (0, 0.5), \
            'min_impurity_decrease': (0, 1), \
            'max_features': (0.1, 1)}
    },
    'svm': {
        'base_params': {'shrinking': True, \
            'probability': True, \
            'cache_size': 200, \
            'class_weight': 'balanced', \
            'max_iter': -1, \
            'decision_function_shape': 'ovr', \
            'random_state': 1213},
        'tuning_params': {'C': (-3, 3), \
            'kernel': ['linear'], \
            'degree': (2, 20, 1), \
            'gamma': (0, 1), \
            'coef0': (0, 1000), \
            'tol': (0, 1)}
    }
}