import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from contextlib import contextmanager
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

def get_logger(level=INFO, output='./log.log'):
    """get logger
    
    Arguments:
        output {str} -- log records output file path
    
    Returns:
        logger -- logger
    """
    # get logger
    logger = getLogger()
    logger.setLevel(level)
    # set two handler, one for stream output
    st_handler = StreamHandler()
    st_handler.setFormatter(Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(st_handler)
    # one for file output if output is set
    if output:
        f_handler = FileHandler(filename=output, mode='a+')
        f_handler.setFormatter(Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(f_handler)
    return logger

@contextmanager
def timer(logger, op_name: str):
    """get executing time
    
    Arguments:
        logger {logger} -- logger for output information
        op_name {str} -- operation name
    """
    start_time = time.time()
    yield
    end_time = time.time()
    logger.info(f'[{op_name}] done in {end_time - start_time:.0f} s')

def reduce_mem_usage(df, logger=None, verbose=True):
    """reduce memory usage
    
    Arguments:
        df {DataFrame} -- the dataframe need memory reduction
    
    Keyword Arguments:
        logger {logger} -- python logger instance (default: {None})
        verbose {bool} -- whether output information about memory reduction (default: {True})
    
    Returns:
        DataFrame -- memory reduced dataframe
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose and logger:
        logger.info(f'Mem. usage decreased to {end_mem:5.2f} Mb ({(start_mem - end_mem)/start_mem:.1f}% reduction)')
    return df

def draw_roc_auc(y_train, y_test, y_train_prob, y_test_prob):
    # calculate fpr and tpr
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_prob)
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)
    auc_train = roc_auc_score(y_train, y_train_prob)
    auc_test = roc_auc_score(y_test, y_test_prob)
    # plot
    plt.subplots(figsize=(8, 6))
    plt.plot(fpr_train, tpr_train, label=f'train auc: {auc_train:.4f}')
    plt.plot(fpr_test, tpr_test, label=f'test auc: {auc_test:.4f}')
    plt.plot([0,1], [0,1], color='red', linestyle='--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.legend()
    plt.show()

def draw_pr_auc(y_train, y_test, y_train_prob, y_test_prob):
    # calculate precision and recall
    p_train, r_train, _ = precision_recall_curve(y_train, y_train_prob)
    p_test, r_test, _ = precision_recall_curve(y_test, y_test_prob)
    ap_train = average_precision_score(y_train, y_train_prob)
    ap_test = average_precision_score(y_test, y_test_prob)
    # plot
    plt.subplots(figsize=(8, 6))
    plt.plot(r_train, p_train, label=f'train average precision: {ap_train:.4f}')
    plt.plot(r_test, p_test, label=f'test average precision: {ap_test:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()

def find_optimal_roc_threshold(y_train, y_train_prob):
    # sensitivity + specificity - 1 = TPR + (1 - FPR) - 1
    # = TPR - FPR
    # Maximize(TPR - FPR)

    # calculate fpr and tpr
    fpr, tpr, thresholds = roc_curve(y_train, y_train_prob)
    idx = np.arange(len(thresholds))
    roc = pd.DataFrame({'tf': pd.Series(tpr - fpr, index=idx),
                        'thresholds': pd.Series(thresholds, index=idx),
                        'fpr': fpr,
                        'tpr': tpr})
    return roc.sort_values(by='tf', axis=0, ascending=False).iloc[0]['thresholds']

def find_optimal_pr_threshold(y_train, y_train_prob=None):
    # Maximize(Recall + Precision - 1)
    # calculate precision and recall
    precision, recall, thresholds = precision_recall_curve(y_train, y_train_prob)
    idx = np.arange(len(thresholds))
    pr = pd.DataFrame({'tf': pd.Series(recall[:-1] + precision[:-1] - 1, index=idx).abs(),
                        'thresholds': pd.Series(thresholds, index=idx)})
    return pr.sort_values(by='tf', axis=0, ascending=False).iloc[0]['thresholds']