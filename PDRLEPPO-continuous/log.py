import logging
import os
from datetime import datetime

# 创建logs目录（如果不存在）
if not os.path.exists('logs_logs'):
    os.makedirs('logs_logs')

# 配置日志
def setup_logger(log_dir):
    # 创建logger对象
    logger = logging.getLogger('my_app')
    logger.setLevel(logging.DEBUG)

    # 生成日志文件名（使用当前日期）
    log_filename = log_dir
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger 