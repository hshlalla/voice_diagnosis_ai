import json, os
import logging.config
filepath = os.environ['DEMENTIA_CONFIG_FILE']
json_data = json.loads(open(filepath).read())

runtimeName = json_data.get('runtimeName', None)
port = json_data.get('port', 9090)
recordFileRoot = json_data.get('recordFileRoot', None)
threadcount = json_data.get('threadcount', None)
log_config_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logging.conf')
logging.config.fileConfig(log_config_file_path)

def getLogger(loggerName):
    logger = logging.getLogger('dementia.{}'.format(loggerName))
    
    return logger

if __name__ == '__main__':
    pass