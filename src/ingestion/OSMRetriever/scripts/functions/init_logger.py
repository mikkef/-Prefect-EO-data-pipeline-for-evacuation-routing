# Configure logger
import os
from loguru import logger

logfolder=os.path.join(os.getcwd(),'logs')
logfile = os.path.join(logfolder,'osm_retriever_{time:YYYY-MM-DD}.log')

if not os.path.exists(logfolder):
    os.mkdir(logfolder)

logger.add(logfile, format='{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}',
           level='INFO', rotation='5 MB', backtrace= True, diagnose=True, colorize=True)

