#!usr/bin python3

import lightgbm
import json
import pickle
import numpy as np
import pandas as pd
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK


class LightgbmOptimizerBinary:
    """
    Use hyperopt to optimize lightgbm binary classifier.
    """

    def __init__(self, X_train, y_train, X_eval, y_eval, X_test, y_test,
                 base_params: dict, cat_params: dict, int_params: dict, float_params: dict,
                 num_opts=1000, trials_path='./trials.pkl', load_trials=False,
                 lgb_num_boost_round=3000, lgb_early_stopping_rounds=400):
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

    def _index2value(self):
        """
        Convert hyperopt.choice optimized index back to original values
        """
        if not hasattr(self, 'best_params'):
            raise ValueError('Best hyperparameters not exist.')
        if not self.best_params:
            raise AttributeError('Model has not been .')
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
        # we invoke difference between train auc and eval auc as penalty
        # eval_auc - (train_auc - eval_auc)
        # that is maximize inverse of the above formula
        return {'loss': -(2 * lgb_clf.best_score['Eval']['auc'] - lgb_clf.best_score['Train']['auc']),
                'train_auc': lgb_clf.best_score['Train']['auc'],
                'eval_auc': lgb_clf.best_score['Eval']['auc'],
                'train_error': lgb_clf.best_score['Train']['binary_error'],
                'eval_error': lgb_clf.best_score['Eval']['binary_error'],
                'status': STATUS_OK}

    def get_best_params(self):
        if not self.best_params:
            raise AttributeError('Best hyperparameters not exist.')
        return self.best_params

    def optimize_lgb(self):
        """
        The main entrance of optimizing LightGBM.
        """
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
