"""
Function for setting-up logs execution script
Author: Omveer Singh
Email: omveer3.singh@gmail.com
"""

import sys, logging, time
from pprint import pformat

START_LEVEL_NUM=25
END_LEVEL_NUM=26
GCONFIG_LEVEL_NUM=27
LPARAMS_LEVEL_NUM=28
logging.addLevelName(START_LEVEL_NUM,"START")
logging.addLevelName(END_LEVEL_NUM,"END")
logging.addLevelName(GCONFIG_LEVEL_NUM,"GLOBAL CONFIG")
logging.addLevelName(LPARAMS_LEVEL_NUM,"PARAMETERS")

def start(self, message,*args,**kws):
    if self.isEnabledFor(START_LEVEL_NUM):
        #yes logger take its '*args' as 'args'
        self._log(START_LEVEL_NUM,message,args,**kws)
def end(self, message,*args,**kws):
    if self.isEnabledFor(END_LEVEL_NUM):
        #yes logger take its '*args' as 'args'
        self._log(END_LEVEL_NUM,message,args,**kws)
def gconfig(self,message,*args,**kws):
    if self.isEnabledFor(GCONFIG_LEVEL_NUM):
        #yes logger take its '*args' as 'args'
        self._log(GCONFIG_LEVEL_NUM,message,args,**kws)
def lparams(self,message,*args,**kws):
    if self.isEnabledFor(LPARAMS_LEVEL_NUM):
        #yes logger take its '*args' as 'args'
        self._log(LPARAMS_LEVEL_NUM,message,args,**kws)

logging.Logger.start=start
logging.Logger.end=end
logging.Logger.gconfig=gconfig
logging.Logger.lparams=lparams

def setup_logger(name,filename):
    """
    Setup logger to log output to console and file
    :param name: Name of logger
    :type name: str
    
    :param filename: where to save logger
    :type filename: str
    
    :return: logger
    """
    logger = logging.getLogger(name)
    
    # Set logger level
    logger.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s','%Y-%m-%d %H:%M')
    
    #setup file 
    file_handler=logging.FileHandler(f'{filename}.log','w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Set up console logging 
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def logger_df_size(df,df_name, newline=False):
    """
    text for logging dataframe size
    
    :param df: data frame to log size for
    :type df: pandas.DataFrame
    
    :param df_name: name of the dataframe to log
    :type df_name: str
    
    :return: string to log
    """
    return(f'{df_name} has shape - Rows:{df.shape[0]:,} | Columns:{df.shape[1]:,}'+('\n' if newline else ''))

def logger_config(config):
    """
    Text for logging config dictionaries
    
    :param config: Dictionary
    :type config: dictionaries
    
    """
    return '\n'+pformat(config)+'\n'

def logger_runtime(start_time,end_time):
    """
    :param start_time: Script running start time
    :type start_time: float
    
    :param end_time: Script running end time
    :type end_time: float
    """
    return time.strftime("%Hh:%Mm:%Ss",time.gmtime(end_time - start_time))