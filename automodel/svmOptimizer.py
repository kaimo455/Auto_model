import pickle
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

class SvmOptimizer:

    def __init__(self, X_train, y_train, X_eval, y_eval, X_test, y_test, \
                 base_params: dict, tuning_params: dict, \
                 num_opts=1000, trials_path='./trials.pkl', load_trials=False):
        # data
        self.X_train, self.X_eval, self.X_test = X_train, X_eval, X_test
        self.y_train, self.y_eval, self.y_test = y_train, y_eval, y_test
        # hyperparameters
        self.base_params, self.tuning_params = base_params, tuning_params
        self.all_params = self._init_params()
        # optimizer
        self.num_opts, self.trials_path, self.load_trials = num_opts, trials_path, load_trials
        self.trials = self._init_trials()

    def _init_params(self):
        # all applied setting
        _params = {**self.base_params, \
                   'C': hp.loguniform('C', *self.tuning_params['C']), \
                   'tol': hp.uniform('tol', *self.tuning_params['tol'])}
        # list to store all available choices
        kernel_choices = []
        # linear setting
        if 'linear' in self.tuning_params['kernel']:
            kernel_linear = {**_params, 'kernel': 'linear'}
            kernel_choices.append(kernel_linear)
        # poly setting
        if 'poly' in self.tuning_params['kernel']:
            kernel_poly = {
                **_params, 'kernel': 'poly', 
                'degree': hp.choice('degree', np.arange(*self.tuning_params['degree'], dtype=np.int)), \
                'gamma': hp.uniform('gamma_poly', *self.tuning_params['gamma']), \
                'coef0': hp.uniform('coef0_poly',  *self.tuning_params['coef0'])}
            kernel_choices.append(kernel_poly)
        # rbf setting
        if 'rbf' in self.tuning_params['kernel']:
            kernel_rbf = {
                **_params, 'kernel': 'rbf', \
                'gamma': hp.uniform('gamma_rbf', *self.tuning_params['gamma'])}
            kernel_choices.append(kernel_rbf)
        # sigmoid setting
        if 'sigmoid' in self.tuning_params['kernel']:
            kernel_sigmoid = {
                **_params, 'kernel': 'sigmoid', \
                'gamma': hp.uniform('gamma_sigmoid', *self.tuning_params['gamma']), \
                'coef0': hp.uniform('coef0_sigmoid',  *self.tuning_params['coef0'])}
            kernel_choices.append(kernel_sigmoid)
        # precomputed setting
        if 'precomputed' in self.tuning_params['kernel']:
            kernel_prec = {**_params, 'kernel': 'precomputed'}
            kernel_choices.append(kernel_prec)
        # all hyperparameters
        return hp.choice('params', kernel_choices)
        
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
        # train model
        self.svm_clf = SVC(**params).fit(self.X_train, self.y_train)
        # auc scores
        self.auc_train = roc_auc_score(self.y_train, self.svm_clf.predict_proba(self.X_train)[:, 1])
        self.auc_eval = roc_auc_score(self.y_eval, self.svm_clf.predict_proba(self.X_eval)[:, 1])
        # accuracy scores
        self.acc_train = accuracy_score(self.y_train, self.svm_clf.predict(self.X_train))
        self.acc_eval = accuracy_score(self.y_eval, self.svm_clf.predict(self.X_eval))
        # we invoke difference between train auc and eval auc as penalty
        # eval_auc - (train_auc - eval_auc)
        # that is maximize inverse of the above formula
        return {'loss': -(2*self.auc_eval - self.auc_train), \
                'train_auc': self.auc_train, \
                'eval_auc': self.auc_eval, \
                'train_error': self.acc_train, \
                'eval_error': self.acc_eval, \
                'status': STATUS_OK}
        
    def optimize_svm(self):
        """
        The main entrence of optimizing RandomForest classifier.
        """
        best_params = fmin(self._object_score, self.all_params, algo=tpe.suggest, \
                           max_evals = self.num_opts, trials=self.trials)
        # save trials for further fine-tune
        pickle.dump(self.trials, open(self.trials_path, "wb"))
        # store best hyperparameters
        self.best_params = best_params
        return self.best_params
        
    def best_model(self):
        """
        Use best hyperparameters to train RandomForest model.
        """
        svm_clf = SVC(**self.best_params).fit(self.X_train, self.y_train)
        svm_clf = svm_clf.fit(pd.concat([self.X_train, self.X_eval]), pd.concat([self.y_train, self.y_eval]))
        return svm_clf