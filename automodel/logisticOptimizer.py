import json
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, accuracy_score

class LogisticOptimizer:
    """
    Use scikit-learn: LogisticRegressionCV to perform cross-validation and tune hyperparameter for \
    logistic regression.
    Example:
    ```
    lr_clf = LogisticOptimizer(X_train, y_train, X_eval, y_eval, X_test, y_test)
    lr_clf.optimize_logistic()
    lr_clf.save_params()
    lr_clf.save_scores()
    ```
    """
    
    def __init__(self, X_train, y_train, X_eval, y_eval, X_test, y_test, \
                 Cs=10, cv=3, penalty='elasticnet', scoring='roc_auc', \
                 solver='saga', max_iter=5000, class_weight='balanced', \
                 n_jobs=4, l1_ratios=np.arange(0, 1, 0.25, dtype=np.float), \
                 output_path='logistic_logs.json'):
        # data
        self.X_train, self.y_train = pd.concat([X_train, X_eval]), pd.concat([y_train, y_eval])
        self.X_test, self.y_test = X_test, y_test
        # logistic regression CV hyperparameters
        self.Cs = Cs
        self.cv = cv
        self.penalty = penalty
        self.scoring = scoring
        self.solver = solver
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.l1_ratios = l1_ratios
        self.output_path = output_path
        
    def optimize_logistic(self):
        self.logistic_reg = LogisticRegressionCV(Cs=self.Cs, cv=self.cv, penalty=self.penalty, scoring=self.scoring, \
                                            solver=self.solver, max_iter=self.max_iter, class_weight=self.class_weight, \
                                            n_jobs=self.n_jobs, l1_ratios=self.l1_ratios)
        self.logistic_reg = self.logistic_reg.fit(self.X_train, self.y_train)

    def get_scores(self):
        return {'auc_train': roc_auc_score(self.y_train, self.logistic_reg.predict_proba(self.X_train)[:, 1]), \
                'auc_test': roc_auc_score(self.y_test, self.logistic_reg.predict_proba(self.X_test)[:, 1]), \
                'acc_train': accuracy_score(self.y_train, self.logistic_reg.predict(self.X_train)), \
                'acc_test': accuracy_score(self.y_test, self.logistic_reg.predict(self.X_test))}

    def save_params(self):
        json.dump({'params': {'C': self.logistic_reg.C_[0], \
                              'class_weight': self.logistic_reg.class_weight, \
                              'cv': self.logistic_reg.cv, \
                              'l1_ratio': self.logistic_reg.l1_ratio_[0], \
                              'tol': self.logistic_reg.tol}}, open(self.output_path, 'w'))

    def save_scores(self):
        self.scores = self.get_scores()
        json.dump({'scores': self.scores}, open(self.output_path, 'a'))