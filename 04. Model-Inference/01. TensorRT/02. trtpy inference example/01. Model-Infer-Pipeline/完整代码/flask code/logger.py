# 一个可以打印保存并打印日志的logger
import logging

initialized_logger = {}

def get_root_logger(logger_name='sr', log_level=logging.INFO, log_file=None):
    logger = logging.getLogger(logger_name)
    # if the logger has been initialized, just return it
    if logger_name in initialized_logger:
        return logger

    format_str = '%(asctime)s %(levelname)s: %(message)s'
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(format_str))
    logger.addHandler(stream_handler)
    logger.propagate = False
    logger.setLevel(log_level)
    # add file handler
    file_handler = logging.FileHandler(log_file, 'w')
    file_handler.setFormatter(logging.Formatter(format_str))
    file_handler.setLevel(log_level)
    logger.addHandler(file_handler)
    initialized_logger[logger_name] = True
    return logger
    
if __name__ == "__main__":
    log_file = 'basicsr.log'    
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info("hello!")