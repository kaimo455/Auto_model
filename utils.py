import time
from contextlib import contextmanager
from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter

import numpy as np

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
            _diff = df[col].max() - df[col].min()
            if str(col_type)[:3] == 'int':
                if _diff <= (np.iinfo(np.int8).max - np.iinfo(np.int8).min):
                    df[col] = df[col].astype(np.int8)
                elif _diff <= (np.iinfo(np.int16).max - np.iinfo(np.int16).min):
                    df[col] = df[col].astype(np.int16)
                elif _diff <= (np.iinfo(np.int32).max - np.iinfo(np.int32).min):
                    df[col] = df[col].astype(np.int32)
                elif _diff <= (np.iinfo(np.int64).max - np.iinfo(np.int64).min):
                    df[col] = df[col].astype(np.int64)
            else:
                if _diff <= (np.iinfo(np.float16).max - np.iinfo(np.float16).min):
                    df[col] = df[col].astype(np.float16)
                elif _diff <= (np.iinfo(np.float32).max - np.iinfo(np.float32).min):
                    df[col] = df[col].astype(np.float32)
                else: df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose and logger:
        logger.info(f'Mem. usage decreased to {end_mem:5.2f} Mb ({(start_mem - end_mem)/start_mem:.1f}% reduction)')
    return df

