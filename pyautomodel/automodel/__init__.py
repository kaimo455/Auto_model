from . import preprocessor, lightgbmOptimizer, logisticOptimizer, randomforestOptimizer, svmOptimizer

__all__ = ['preprocessor', 'lightgbmOptimizer', 'logisticOptimizer', 'randomforestOptimizer', 'svmOptimizer']

PARAMS_EXAMPLES = {
    'lightgbm': {
        'base_params': {'task': 'train',
                        'objective': 'binary',
                        'tree_learner': 'serial',
                        'num_threads': 4,
                        'device_type': 'cpu',
                        'seed': 1213,
                        'bagging_seed': 42,
                        'feature_fraction_seed': 3,
                        'first_metric_only': True,
                        'max_delta_step': 0,
                        'bin_construct_sample_cnt': 200000,
                        'histogram_pool_size': -1,
                        'is_unbalance': True,
                        'metric': 'auc,binary_logloss,binary_error',
                        'max_bin': 255,
                        'metric_freq': 1},
        'cat_params': {'boosting': ['gbdt']},
        'int_params': {'num_leaves': (31, 1024, 8),
                       'max_depth': (1, 100, 1),
                       'min_data_in_leaf': (20, 500, 4),
                       'bagging_freq': (0, 100, 1),
                       'min_data_per_group': (100, 500, 10),
                       'max_cat_threshold': (32, 256, 2),
                       'max_cat_to_onehot': (4, 100, 1),
                       'min_data_in_bin': (3, 128, 2)},
        'float_params': {'min_sum_hessian_in_leaf': (0, 0.1),
                         'bagging_fraction': (0.1, 1),
                         'pos_bagging_fraction': (0.1, 1),
                         'neg_bagging_fraction': (0.1, 1),
                         'feature_fraction': (0.1, 1),
                         'feature_fraction_bynode': (0.1, 1),
                         'lambda_l1': (0, 500),
                         'lambda_l2': (0, 4000),
                         'sigmoid': (1, 500),
                         'cat_l2': (10, 1000),
                         'cat_smooth': (10, 1000),
                         'min_gain_to_split': (0, 100)}
    },
    'xgboost': {
        'base_params': {'booster': 'gbtree',
                        'verbosity': 1,
                        'nthread': 4,
                        'disable_default_eval_metric': 0,
                        'max_delta_step': 0,
                        'tree_method': 'auto',
                        'max_leaves': 0,
                        'max_bin': 256,
                        'predictor': 'cpu_predictor',
                        'num_parallel_tree': 1,
                        'objective': 'binary:logistic',
                        'base_score': 0.5,
                        'eval_metric': 'auc',
                        'seed': 1213},
        'cat_params': {},
        'int_params': {'max_depth': (1, 100, 1)},
        'float_params': {'min_split_loss': (0, 100),
                         'min_child_weight': (0, 100),
                         'subsample': (0.1, 1),
                         'colsample_bytree': (0.1, 1),
                         'colsample_bylevel': (0.1, 1),
                         'colsample_bynode': (0.1, 1),
                         'reg_alpha' : (0, 500),
                         'reg_lambda': (0, 4000),
                         'scale_pos_weight': (0.1, 1)}
    },
    'catboost': {
        'base_params': {'objective': 'Logloss',
                        'custom_metric': ['Accuracy', 'AUC'],
                        'eval_metric': 'AUC',
                        'random_seed': 1213,
                        'bootstrap_type': 'Bayesian',
                        'bagging_temperature': 1,
                        'use_best_model': True,
                        'has_time': False,
                        'leaf_estimation_method': 'Newton',
                        'approx_on_full_history': False,
                        'scale_pos_weight': 1,
                        'boosting_type': 'Plain',
                        'allow_const_label': False,
                        'metric_period': 1},
        'cat_params': {'sampling_frequency': ['PerTree', 'PerTreeLevel'], \
                       'sampling_unit': ['Object'], \
                       'grow_policy': ['SymmetricTree'], \
                       'nan_mode': ['Min', 'Max', 'Forbidden']},
        'int_params': {'max_depth': (5, 12, 1),
                       'min_data_in_leaf': (1, 2, 1),
                       'num_leaves': (31, 32, 4),
                       'one_hot_max_size': (2, 3, 1),
                       'fold_permutation_block': (1, 2, 1)},
        'float_params': {'reg_lambda': (0, 50),
                       'random_strength': (0.1, 1),
                       'colsample_bylevel': (0.1, 1),
                       'fold_len_multiplier': (2.0, 4.0)}
    },
    'randomforest': {
        'base_params': {'bootstrap': True,
                        'oob_score': False,
                        'n_jobs': 4,
                        'random_state': 1213,
                        'warm_start': False,
                        'class_weight': 'balanced_subsample'},
        'cat_params': {'criterion': ['gini', 'entropy']},
        'int_params': {'n_estimators': (10, 500, 10),
                       'max_depth': (1, 50, 1),
                       'min_samples_split': (2, 500, 2),
                       'min_samples_leaf': (1, 250, 1),
                       'max_leaf_nodes': (2, 1024, 4)},
        'float_params': {'min_weight_fraction_leaf': (0, 0.5),
                         'min_impurity_decrease': (0, 1),
                         'max_features': (0.1, 1)}
    },
    'svm': {
        'base_params': {'shrinking': True,
                        'probability': True,
                        'cache_size': 200,
                        'class_weight': 'balanced',
                        'max_iter': -1,
                        'decision_function_shape': 'ovr',
                        'random_state': 1213},
        'tuning_params': {'C': (-3, 3),
                          'kernel': ['linear'],
                          'degree': (2, 20, 1),
                          'gamma': (0, 1),
                          'coef0': (0, 1000),
                          'tol': (0, 1)}
    }
}

"""XGBoost
## unused hyperparameters
        # sketch_eps | only in tree_method=approx
        # updater | default: grow_colmaker,prune,  usually set automatically
        # refresh_leaf | default: 1
        # process_type | default: default
        # grow_policy | default: depthwise
        # sample_type | used in dart
        # normalize_type | used in dart
        # rate_drop | used in dart
        # one_drop | used in dart
        # skip_drop | used in dart
        # reg_lambda | used in gblinear
        # reg_alpha | used in gblinear
        # updater | used in gblinear
        # feature_selector | used in gblinear
        # top_k | used in gblinear
"""
"""Catboost
## unused hyperparmeters
        # subsample | only used in bootstrap_type='Poisson|Bernoulli'
        # mvs_reg | only used in bootstrap_type='MVS'
        # leaf_estimation_backtracking | not used in leaf_estimation_method='Newton' for binary classification
        # class_weight | using scale_pos_weight instead
        # score_function | default: Correlation (NewtonL2 if the growing policy is set to Lossguide)
        # leaf_estimation_iterations | default: Depends on the training objective
        # ignored_features
        # best_model_min_trees
        # output_borders
        # monotone_constraints
"""