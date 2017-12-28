"""
A class used to log status
"""

import logging
import os
import time

__author__ = 'Cong Bao'

class Logger(object):

    DEBUG = 'DEBUG'
    INFO = 'INFO'
    WARN = 'WARN'
    ERROR = 'ERROR'
    CRITICAL = 'CRITICAL'

    def __init__(self, name, log_dir='./log/'):
        self.name = name
        self.log_dir = log_dir
        self.__logger = logging.getLogger(self.name)
        self.__logger.setLevel(logging.INFO)
        self.__date = time.strftime('%Y-%m-%d', time.localtime())
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
        fh = logging.FileHandler(self.log_dir + self.name + '-' + self.__date + '.log')
        fh.setLevel(logging.INFO)
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        fmt = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
        fh.setFormatter(fmt)
        sh.setFormatter(fmt)
        self.__logger.addHandler(fh)
        self.__logger.addHandler(sh)

    def log(self, msg, level='INFO'):
        if level == Logger.DEBUG:
            self.__logger.debug(msg)
        elif level == Logger.INFO:
            self.__logger.info(msg)
        elif level == Logger.WARN:
            self.__logger.warning(msg)
        elif level == Logger.ERROR:
            self.__logger.error(msg)
        elif level == Logger.CRITICAL:
            self.__logger.critical(msg)
        else:
            pass
