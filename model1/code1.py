from logging import config
import os,sys,time,joblib,inspect,json
from venv import logger
import pandas as pd
import numpy as np
start_time=time.time()

# Configuration
sys.path.append("..")
from __config__ import config
locals().update(config)
# Set directory Name
file_dir=os.path.abspath(os.path.join(__file__,"../.."))
print(file_dir)
# file_dir=os.path.abspath(os.path.join(os.getcwd(),"../.."))  # command in Jupyter
sys.path.append(file_dir)

common_data_path=os.path.abspath(os.path.join(file_dir,'data/common_data'))
log_path=os.path.abspath(os.path.join(file_dir,'logs/common_data'))

# print the paths
print(common_data_path)
print(log_path)

# Import functions
from src.utils.logs import *

#setup logger
logger=setup_logger(__name__, log_path+'/'+__file__[:-3])
logger.start("****Start Code1 to run ********\n")
logger.gconfig(logger_config(config))

def make_segment_directories(base_directory,sub_directory):
    """
    Function creates directory if it does not already exist
    """
    try:
        os.makedirs(os.path.join(base_directory,sub_directory))
        print(f"{sub_directory} created")
    except:
        print(f"{sub_directory} already exist")
        pass
make_segment_directories(file_dir,'data')

logger.end(f'***** Time taken : {logger_runtime(start_time,time.time())}*****\n')
