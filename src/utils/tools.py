"""
this finction enables reading and saving data in csv and parquet
"""

from bz2 import compress
from distutils import extension
import errno,joblib,json,logging,warnings,os
from tkinter import NO
from venv import logger
import pandas as pd
logger =logging.getLogger('__main__.'+'__name__')

def read_data(file_path,usecols=None,use_logs=True):
    """
    read data with detection of extension type
    """
    #check file exist
    if use_logs:
        logger.info(f"reading data:{file_path}")
    if not os.path.exists(file_path):
        logger.critical("Dataset was not found, exiting.\n")
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),file_path)
    extension =file_path.split('.')[-1].lower()

    if extension=='csv':
        if usecols is None:
            df=pd.read_csv(file_path,usecols=usecols,index_col=0)
        else:
           df=pd.read_csv(file_path,usecols=usecols) 
    elif extension== 'parquet':
        df=pd.read_parquet(file_path,usecols=usecols)
    else:
        raise ValueError(f"extension not supported. use 'csv' or 'parquet' ")
    if use_logs:
        logger.info(f"reading data complete the shape of the data is {df.shape}")

    return df, extension

def save_data():
    pass

def save_compressed_data(df,file_path):
    if os.path.exists(file_path):
        logger.debug(f"file path already exist")
        logger.debug(f"overwrite {file_path}") 
        warnings.warn(f"{file_path} already exist and was overwritten")
    joblib.dump(df,file_path,compress=('gzip',3))

def save_json(obj,file_path):
    if os.path.exists(file_path):
        logger.debug(f"file path already exist")
        logger.debug(f"overwrite {file_path}") 
        warnings.warn(f"{file_path} already exist and was overwritten")
    with open(file_path,'w') as file:
        json.dump(obj,file)