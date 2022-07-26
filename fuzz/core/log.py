import os
import sys
import time
import traceback
from IPython.core.ultratb import ColorTB

class Logger(object):
    def __init__(self, file_name = ''):
        self.__console__ = sys.stdout
        sys.stdout = self
        self.file_name = file_name
        self.not_change_line = False

    def write(self, message):
        self.to_console(message)
        self.to_file(message)

    def to_console(self, message):
        self.__console__.write(message)

    def to_file(self, message):
        write_file = True
        if '\r' in message:
            if self.not_change_line == False: self.not_change_line = True
            self.line_message = message
            write_file = False
        elif self.not_change_line:
            message = self.line_message + message
            self.not_change_line = False
        if write_file:
            with open(self.file_name, 'a') as logfile:
                logfile.write(message)

    def flush(self):
        pass
    
    def reset(self):
        sys.stdout = self.__console__


if __name__ == '__main__':
    # 自定义目录存放日志文件
    log_path = '../Logs/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    # 日志文件名按照程序运行时间设置
    log_file_name = log_path + 'log-' + time.strftime("%Y.%m.%d-%Hh %Mm %Ss", time.localtime()) + '.log'
    
    
    print('before')
    
    logger = Logger(log_file_name)
    
    def overwrite_console(p_str):
        logger.to_console("{}\r".format(p_str))
    
    try:
        logger.to_file("log only!\n")
        logger.to_console("console only!\n")
        print("both log and console")
        print(log_file_name)
        
        s = 'abcdefghijkl'
        
        for x in range(0,5):
            overwrite_console(s[x]*(5-x))
    
        print(2/0)
    except:
        color = ColorTB()
        exc = sys.exc_info()
        logger.to_file(traceback.format_exc())
        logger.reset()
        for _str in color.structured_traceback(*exc):
            print(_str)
    finally:
        logger.reset()
