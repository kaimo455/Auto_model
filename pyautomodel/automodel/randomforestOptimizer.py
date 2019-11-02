import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

class RandomForestOptimizer:
    
    def __init__(self, X_train, y_train, X_eval, y_eval, X_test, y_test, \
                 base_params: dict, cat_params: dict, int_params: dict, float_params: dict, \
                 num_opts=1000, trials_path='./trials.pkl', load_trials=False):
        # data
        self.X_train, self.X_eval, self.X_test = X_train, X_eval, X_test
        self.y_train, self.y_eval, self.y_test = y_train, y_eval, y_test
        # hyperparameters
        self.base_params, self.cat_params = base_params, cat_params
        self.int_params, self.float_params = int_params, float_params
        self.all_params = self._init_params()
        # optimizer
        self.num_opts, self.trials_path, self.load_trials = num_opts, trials_path, load_trials
        self.trials = self._init_trials()

    def _init_params(self):
        """
        Initialize hyperparameters.
        """
        # categorical hyperparameters
        self.cat_params_hp = {param: hp.choice(param, candidates) \
                      for param, candidates in self.cat_params.items()}
        # integer hyperparameters
        self.int_params_hp = {param: hp.uniform(param, np.arange(*start_end_step, dtype=np.int)) \
                              for param, start_end_step in self.int_params.items()}
        # float hyperparameters
        self.float_params_hp = {param: hp.uniform(param, *candidates) \
                                for param, candidates in self.float_params.items()}
        # generate all hyperparameters
        return dict(self.base_params, **self.cat_params_hp, \
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

    def _object_score(self, params):
        """
        Using all hyperparameters to train RandomFroestClassifier. Return the objective function score.
        """
        # train model
        self.rf_clf = RandomForestClassifier(**params).fit(self.X_train, self.y_train)
        # auc scores
        self.auc_train = roc_auc_score(self.y_train, self.rf_clf.predict_proba(self.X_train)[:, 1])
        self.auc_eval = roc_auc_score(self.y_eval, self.rf_clf.predict_proba(self.X_eval)[:, 1])
        # accuracy scores
        self.acc_train = accuracy_score(self.y_train, self.rf_clf.predict(self.X_train))
        self.acc_eval = accuracy_score(self.y_eval, self.rf_clf.predict(self.X_eval))
        # we invoke difference between train auc and eval auc as penalty
        # eval_auc - (train_auc - eval_auc)
        # that is maximize inverse of the above formula
        return {'loss': -(2*self.auc_eval - self.auc_train), \
                'train_auc': self.auc_train, \
                'eval_auc': self.auc_eval, \
                'train_error': self.acc_train, \
                'eval_error': self.acc_eval, \
                'status': STATUS_OK}

    def optimize_rf(self):
        """
        The main entrence of optimizing RandomForest classifier.
        """
        best_params = fmin(self._object_score, self.all_params, algo=tpe.suggest, \
                           max_evals = self.num_opts, trials=self.trials)
        # save trials for further fine-tune
        pickle.dump(self.trials, open(self.trials_path, "wb"))
        # convert categorical from index back to value
        for key, value in self.cat_params.items():
            best_params[key] = value[best_params[key]]
        # store best hyperparameters
        self.best_params = best_params
        return self.best_params

    def best_model(self):
        """
        Use best hyperparameters to train RandomForest model.
        """
        rf_clf = RandomForestClassifier(**self.best_params)
        rf_clf = rf_clf.fit(pd.concat([self.X_train, self.X_eval]), pd.concat([self.y_train, self.y_eval]))
        return rf_clf