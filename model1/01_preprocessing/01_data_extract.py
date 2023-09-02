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
file_dir=os.path.abspath(os.path.join(__file__,"../../.."))
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

df=pd.read_csv(common_data_path+'/lgd.csv')
logger.info(logger_df_size(df,'lgd.csv'))
logger.info("Description of the data lgd.csv")
logger.info(f"Data Description:{df.describe()}")
logger.end(f'***** Time taken : {logger_runtime(start_time,time.time())}*****\n')
