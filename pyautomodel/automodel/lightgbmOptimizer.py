#!usr/bin python3
import lightgbm
import json
import pickle
import numpy as np
import pandas as pd
from .preprocessor import Preprocessor
from sklearn.model_selection import train_test_split
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK


class LightgbmOptimizerBinary:
    """
    Use hyperopt to optimize lightgbm binary classifier.
    """

    def __init__(self, data, label,
                 base_params: dict, cat_params: dict, int_params: dict, float_params: dict,
                 num_opts=1000, trials_path='./trials.pkl', load_trials=False,
                 lgb_num_boost_round=3000, lgb_early_stopping_rounds=400,
                 lr_callback=None,
                 test_size=0.25, shuffle=True, random_state=1213,
                 cv=None, strategy='stratified', group=None):
        """LightGBM hyperparameters optimizer initializer.
        
        Arguments:
            data {DataFrame} -- dataset for training
            label {Series} -- labels for training
            cat_params {dict} -- categorical hyperparameters to tune
            int_params {dict} -- integer hyperparameters to tune
            float_params {dict} -- float/double hyperparameters to tune
            lr_callback {func} -- function take current_round as inputs to schedule lr
        
        Keyword Arguments:
            trials_path {str} -- hyperopt trials output file path, if load_trials is True then try to load current \
                exist trials file (default: {'./trials.pkl'})
            load_trials {bool} -- whether to load current exist trials file (default: {False})
            lgb_num_boost_round {int} -- number of training rounds for lightgbm model (default: {3000})
            lgb_early_stopping_rounds {int} -- early stopping rounds for lightgbm model (default: {400})
        """
        # store data and label we need to use this to generate folds generator
        self.data = data
        self.label = label
        if cv:
            # parameters for making folds dataset
            self._n_splits = cv
            self._strategy = strategy
            self._group = group
            # construct dataset
            self.dataset = lightgbm.Dataset(self.data, self.label)
        # parameters for making hold out dataset
        self._test_size = test_size
        self._shuffle = shuffle
        self._random_state = random_state
        # construct training and evaluating dataset
        self.train_dataset, self.eval_dataset = self._make_hold_out(data, label)
        # hyperparameters
        self._base_params, self._cat_params = base_params, cat_params
        self._int_params, self._float_params = int_params, float_params
        self._all_params = self._init_params()
        # lightgbm other hyperparameter
        self._lgb_num_boost_round, self._lgb_early_stopping_rounds = lgb_num_boost_round, lgb_early_stopping_rounds
        self._lr_callback = lr_callback
        # optimizer
        self._num_opts, self._trials_path, self._load_trials = num_opts, trials_path, load_trials
        self.trials = self._init_trials()

    def _init_params(self):
        """Initialize hyperparameters.
        """
        # categorical hyperparameters
        cat_params_hp = {param: hp.choice(param, candidates) for param, candidates in self._cat_params.items()}
        # integer hyperparameters
        int_params_hp = {param: hp.choice(param, np.arange(*start_end_step, dtype=np.int))
                         for param, start_end_step in self._int_params.items()}
        # float hyperparameters
        float_params_hp = {param: hp.uniform(param, *candidates) for param, candidates in self._float_params.items()}
        # generate all hyperparameters
        return dict(self._base_params, **cat_params_hp, **int_params_hp, **float_params_hp)

    def _init_trials(self):
        """Initialize trials database.
        """
        if self._load_trials:
            trials = pickle.load(open(self._trials_path, "rb"))
            current_iter = len(trials.losses())
            self._num_opts += current_iter
        else:
            trials = Trials()
        return trials

    def _init_folds(self):
        self._folds = Preprocessor.make_folds(self.data, self.label,
                                              n_splits=self._n_splits, strategy=self._strategy, group=self._group,
                                              shuffle=self._shuffle, random_state=self._random_state)

    def _index2value(self):
        """
        Convert hyperopt.choice optimized index back to original values
        """
        if hasattr(self, 'best_params'):
            # convert categorical from index back to value
            for key, value in self._cat_params.items():
                self.best_params[key] = value[self.best_params[key]]
            # convert integer from index back to value
            for key, value in self._int_params.items():
                self.best_params[key] = np.arange(*value, dtype=np.int)[self.best_params[key]]
        else:
            raise AttributeError('Best hyperparameters not exist.')

    def _make_hold_out(self, data, label):
        """Split training and evaluating set
        Return (train_set, eval_set)
        """
        X_train, X_eval, y_train, y_eval = train_test_split(data, label,
                                                            test_size=self._test_size,
                                                            random_state=self._random_state,
                                                            shuffle=self._shuffle)
        train_set = lightgbm.Dataset(X_train, y_train)
        eval_set = lightgbm.Dataset(X_eval, y_eval)
        return train_set, eval_set

    def _object_score(self, params):
        """Using all hyperparameters to train LightGBM. Return the objective function score.
        """
        lgb_clf = lightgbm.train(params=params,
                                 train_set=self.train_dataset,
                                 num_boost_round=self._lgb_num_boost_round,
                                 valid_sets=[self.train_dataset, self.eval_dataset],
                                 valid_names=['Train', 'Eval'],
                                 early_stopping_rounds=self._lgb_early_stopping_rounds,
                                 verbose_eval=-1,
                                 callbacks=[lightgbm.callback.reset_parameter(learning_rate=self._lr_callback)])
        # define return dict
        ret_dict = {'status': STATUS_OK}
        # get auc results
        auc_train = lgb_clf.best_score['Train']['auc']
        auc_eval = lgb_clf.best_score['Eval']['auc']
        ret_dict.update({'train_auc': auc_train, 'eval_auc': auc_eval})
        # get all other metrics results
        if 'binary_error' in params['metric'].split(','):
            error_train = lgb_clf.best_score['Train']['binary_error']
            error_eval = lgb_clf.best_score['Eval']['binary_error']
            ret_dict.update({'train_error': error_train, 'eval_error': error_eval})
        # define loss
        # we invoke difference between train auc and eval auc as penalty
        # eval_auc - (train_auc - eval_auc)
        # that is maximize inverse of the above formula
        loss = -(2 * auc_eval - auc_train)
        ret_dict.update({'loss': loss})
        return ret_dict

    def _object_score_cv(self, params):
        """Using all hyperparameters to train LightGBM in a CV fashion. Return the objective function score.
        """
        # we need to initialize folds every time training LightGBM because it's a generator
        self._init_folds()
        # get evaluation results
        eval_hist = lightgbm.cv(params=params,
                                train_set=self.dataset,
                                num_boost_round=self._lgb_num_boost_round,
                                folds=self._folds,
                                nfold=None, stratified=None, shuffle=None, seed=None,
                                metrics=None, fobj=None, feval=None, init_model=None,
                                feature_name='auto', categorical_feature='auto',
                                early_stopping_rounds=self._lgb_early_stopping_rounds,
                                fpreproc=None, verbose_eval=-1, show_stdv=True,
                                callbacks=[lightgbm.callback.reset_parameter(learning_rate=self._lr_callback)],
                                eval_train_metric=False)
        # define return dict
        ret_dict = {'status': STATUS_OK}
        # get auc results
        auc_mean = eval_hist['auc-mean']
        ret_dict.update({'auc_mean': auc_mean})
        # get all other metrics results
        if 'binary_error' in params['metric'].split(','):
            error_mean = eval_hist['binary_error-mean']
            ret_dict.update({'binary_error-mean': error_mean})
        # define loss, use [-1] to get last round result
        loss = -auc_mean[-1]
        ret_dict.update({'loss': loss})
        return ret_dict

    def optimize(self):
        """The main entrance of optimizing LightGBM model.
        """
        if hasattr(self, '_n_splits'):
            self.best_params = fmin(self._object_score_cv, self._all_params, algo=tpe.suggest,
                                    max_evals=self._num_opts, trials=self.trials)
        else:
            self.best_params = fmin(self._object_score, self._all_params, algo=tpe.suggest,
                                    max_evals=self._num_opts, trials=self.trials)
        # save trials for further fine-tune
        pickle.dump(self.trials, open(self._trials_path, "wb"))
        # there are some params in best_params is index value
        # we need to convert back to actual value
        self._index2value()

    def get_best_model(self):
        """Use best hyperparameters to train LightGBM model.
        """
        # check if the hyperparameters is tuned
        if not hasattr(self, 'best_params'):
            raise AttributeError('Best hyperparameters are not exist because not optimize yet.')
        lgb_clf = lightgbm.train(params=dict(self.best_params, **self._base_params),
                                 train_set=self.train_dataset,
                                 num_boost_round=self._lgb_num_boost_round,
                                 valid_sets=[self.train_dataset, self.eval_dataset],
                                 valid_names=['Train', 'Eval'],
                                 early_stopping_rounds=self._lgb_early_stopping_rounds,
                                 verbose_eval=1,
                                 callbacks=[lightgbm.callback.reset_parameter(learning_rate=self._lr_callback)])
        return lgb_clf

    def get_best_params(self):
        if hasattr(self, 'best_params'):
            return self.best_params
        else:
            raise AttributeError('Best hyperparameters not exist.')
