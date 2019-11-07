#!usr/bin python3

import lightgbm
import json
import pickle
import numpy as np
import pandas as pd
from preprocessor import Preprocessor
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK


class LightgbmOptimizerBinary:
    """
    Use hyperopt to optimize lightgbm binary classifier.
    """

    def __init__(self, X_train, y_train, X_eval, y_eval, X_test, y_test,
                 base_params: dict, cat_params: dict, int_params: dict, float_params: dict,
                 num_opts=1000, trials_path='./trials.pkl', load_trials=False,
                 lgb_num_boost_round=3000, lgb_early_stopping_rounds=400, 
                 cv=None, strategy='stratified', group=None, shuffle=True, random_state=1213):
        """LightGBM hyperparameters optimizer initializer.
        
        Arguments:
            X_train {DataFrame} -- training dataset
            y_train {Series} -- training labels
            X_eval {DataFrame} -- evaluation dataset
            y_eval {Series} -- evaluation labels
            X_test {DataFrame} -- testing dataset
            y_test {Series} -- testing labels
            cat_params {dict} -- categorical hyperparameters to tune
            int_params {dict} -- integer hyperparameters to tune
            float_params {dict} -- float/double hyperparameters to tune
        
        Keyword Arguments:
            trials_path {str} -- hyperopt trials output file path, if load_trials is True then try to load current \
                exist trials file (default: {'./trials.pkl'})
            load_trials {bool} -- whether to load current exist trials file (default: {False})
            lgb_num_boost_round {int} -- number of training rounds for lightgbm model (default: {3000})
            lgb_early_stopping_rounds {int} -- early stopping rounds for lightgbm model (default: {400})
        """
        # data
        self.X_train, self.X_eval, self.X_test = X_train, X_eval, X_test
        self.y_train, self.y_eval, self.y_test = y_train, y_eval, y_test
        self.train_dataset = lightgbm.Dataset(data=self.X_train, label=self.y_train)
        self.eval_dataset = lightgbm.Dataset(data=self.X_eval, label=self.y_eval)
        self.test_dataset = lightgbm.Dataset(data=self.X_test, label=self.y_test)
        # hyperparameters
        self.base_params, self.cat_params = base_params, cat_params
        self.int_params, self.float_params = int_params, float_params
        self.all_params = self._init_params()
        self.best_params = None
        # lightgbm other hyperparameter
        self.lgb_num_boost_round, self.lgb_early_stopping_rounds = lgb_num_boost_round, lgb_early_stopping_rounds
        # optimizer
        self.num_opts, self.trials_path, self.load_trials = num_opts, trials_path, load_trials
        self.trials = self._init_trials()
        # whether use CV to trian and optimize model
        if cv:
            self._n_splits = cv
            self._strategy = strategy
            self._group = group
            self._shuffle = shuffle
            self._random_state = random_state

    def _init_params(self):
        """
        Initialize hyperparameters.
        """
        # categorical hyperparameters
        self.cat_params_hp = {param: hp.choice(param, candidates)
                              for param, candidates in self.cat_params.items()}
        # integer hyperparameters
        self.int_params_hp = {param: hp.choice(param, np.arange(*start_end_step, dtype=np.int))
                              for param, start_end_step in self.int_params.items()}
        # float hyperparameters
        self.float_params_hp = {param: hp.uniform(param, *candidates)
                                for param, candidates in self.float_params.items()}
        # generate all hyperparameters
        return dict(self.base_params, **self.cat_params_hp,
                    **self.int_params_hp, **self.float_params_hp)

    def _init_trials(self):
        """
        Initialize trials database.
        """
        if self.load_trials:
            trials = pickle.load(open(self.trials_path, "rb"))
            current_iter = len(trials.losses())
            self.num_opts += current_iter
        else:
            trials = Trials()
        return trials

    def _init_folds(self):
        self._folds = Preprocessor.make_folds(self.X_train, self.y_train, 
                                              n_splits=self._n_splits, strategy=self._strategy, group=self._group,
                                              shuffle=self._shuffle, random_state=self._random_state)

    def _index2value(self):
        """
        Convert hyperopt.choice optimized index back to original values
        """
        if not hasattr(self, 'best_params'):
            raise ValueError('Best hyperparameters not exist.')
        if not self.best_params:
            raise AttributeError('Model has not been trained.')
        # convert categorical from index back to value
        for key, value in self.cat_params.items():
            self.best_params[key] = value[self.best_params[key]]
        # convert integer from index back to value
        for key, value in self.int_params.items():
            self.best_params[key] = np.arange(*value, dtype=np.int)[self.best_params[key]]

    def _object_score(self, params):
        """
        Using all hyperparameters to train LightGBM. Return the objective function score.
        """
        lgb_clf = lightgbm.train(params=params,
                                 train_set=self.train_dataset,
                                 num_boost_round=self.lgb_num_boost_round,
                                 valid_sets=[self.train_dataset, self.eval_dataset],
                                 valid_names=['Train', 'Eval'],
                                 early_stopping_rounds=self.lgb_early_stopping_rounds,
                                 verbose_eval=-1,
                                 callbacks=[lightgbm.callback.reset_parameter(
                                     learning_rate=lambda current_round: 0.6 * (0.99 ** current_round))])
        # define return dict
        ret_dict = {'status': STATUS_OK}
        # get auc results
        auc_train = best_score['Train']['auc']
        auc_eval = best_score['Eval']['auc']
        ret_dict.update({'train_auc': auc_train, 'eval_auc': auc_eval})
        # get all other metrics results
        if 'binary_error' in params['metric'].split(','):
            error_train = best_score['Train']['binary_error']
            error_eval = best_score['Eval']['binary_error']
            ret_dict.update({'train_error': error_train, 'eval_error': error_eval})
        # define loss
        # we invoke difference between train auc and eval auc as penalty
        # eval_auc - (train_auc - eval_auc)
        # that is maximize inverse of the above formula
        loss = -(2 * auc_eval - auc_train)
        ret_dict.update({'loss': loss})
        return ret_dict

    def _object_score_cv(self, params):
        """
        Using all hyperparameters to train LightGBM in a CV fashion. Return the objective function score.
        """
        # we need to initialize folds every time
        self._init_folds()
        eval_hist = lightgbm.cv(params=params,
                                train_set=self.train_dataset,
                                num_boost_round=self.lgb_num_boost_round,
                                folds=self._folds,
                                nfold=None, stratified=None, shuffle=None, seed=None,
                                metrics=None,
                                fobj=None,
                                feval=None,
                                init_model=None,
                                feature_name='auto',
                                categorical_feature='auto',
                                early_stopping_rounds=self.lgb_early_stopping_rounds,
                                fpreproc=None,
                                verbose_eval=-1,
                                show_stdv=True,
                                callbacks=[lightgbm.callback.reset_parameter(
                                     learning_rate=lambda current_round: 0.6 * (0.99 ** current_round))],
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
        """
        The main entrance of optimizing LightGBM model.
        """
        if hasattr(self, '_n_splits'):
            best_params = fmin(self._object_score_cv, self.all_params, algo=tpe.suggest,
                               max_evals=self.num_opts, trials=self.trials)
        else:
            best_params = fmin(self._object_score, self.all_params, algo=tpe.suggest,
                               max_evals=self.num_opts, trials=self.trials)
        # save trials for further fine-tune
        pickle.dump(self.trials, open(self.trials_path, "wb"))
        # store best hyperparameters
        self.best_params = best_params
        # there are some params in best_params is index value
        # we need to convert back to actual value
        self._index2value()

    def get_best_model(self):
        """
        Use best hyperparameters to train lihgtgbm model.
        """
        lgb_clf = lightgbm.train(params=dict(self.best_params, **self.base_params),
                                 train_set=self.train_dataset,
                                 num_boost_round=self.lgb_num_boost_round,
                                 valid_sets=[self.train_dataset, self.eval_dataset],
                                 valid_names=['Train', 'Eval'],
                                 early_stopping_rounds=self.lgb_early_stopping_rounds,
                                 verbose_eval=1,
                                 callbacks=[lightgbm.callback.reset_parameter(
                                     learning_rate=lambda current_round: 0.6 * (0.99 ** current_round))])
        return lgb_clf

    def get_best_params(self):
        if not self.best_params:
            raise AttributeError('Best hyperparameters not exist.')
        return self.best_params