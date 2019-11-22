import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, accuracy_score
from .preprocessor import Preprocessor

class LogisticOptimizer:
    """
    Use scikit-learn: LogisticRegressionCV to perform cross-validation and tune hyperparameter for \
    logistic regression.
    Example:
    ```
    lr_clf = LogisticOptimizer(data, label)
    lr_clf.optimize_logistic()
    lr_clf.save_params()
    lr_clf.save_scores()
    ```
    """
    
    def __init__(self, data, label, 
                 Cs=10, cv=3, penalty='elasticnet', scoring='roc_auc', 
                 solver='saga', max_iter=5000, class_weight='balanced', 
                 n_jobs=4, l1_ratios=np.arange(0, 1, 0.25, dtype=np.float), 
                 output_path='logistic_logs.json'):
        # data
        self.dataset, self.label = data, label
        # logistic regression CV hyperparameters
        self._Cs = Cs
        self._cv = cv
        self._penalty = penalty
        self._scoring = scoring
        self._solver = solver
        self._max_iter = max_iter
        self._class_weight = class_weight
        self._n_jobs = n_jobs
        self._l1_ratios = l1_ratios
        self._output_path = output_path
        
    def optimize(self):
        self.logistic_reg = LogisticRegressionCV(Cs=self._Cs, cv=self._cv, penalty=self._penalty, scoring=self._scoring, \
                                            solver=self._solver, max_iter=self._max_iter, class_weight=self._class_weight, \
                                            n_jobs=self._n_jobs, l1_ratios=self._l1_ratios)
        self.logistic_reg = self.logistic_reg.fit(self.dataset, self.label)
        self.best_params = {'penalty': self._penalty,
                            'tol': self.logistic_reg.tol,
                            'C': self.logistic_reg.C_[0],
                            'class_weight': self.logistic_reg.class_weight,
                            'solver': self._solver,
                            'max_iter': self._max_iter,
                            'n_jobs': self._n_jobs,
                            'l1_ratio': self.logistic_reg.l1_ratio_[0]}

    def get_best_model(self):
        """
        Use best hyperparameters to train lihgtgbm model.
        """
        # check if the hyperparameters is tuned
        if not hasattr(self, 'best_params'):
            raise AttributeError('Best hyperparameters are not exist because not optimize yet.')
        lr_clf = LogisticRegression(**self.best_params).fit(self.dataset, self.label)
        return lr_clf
    
    def get_best_params(self):
        if hasattr(self, 'best_params'):
            return self.best_params
        else:
            raise AttributeError('Best hyperparameters not exist.')