import configparser
import re


def load_config(config_file='config.txt'):
    config = configparser.ConfigParser()
    config.read(config_file)
    
    config_dict = {}
    for section in config.sections():
        for key, value in config[section].items():
            if ',' in value and not (value.startswith('(') or value.startswith('[')):
                config_dict[key.upper()] = [item.strip() for item in value.split(',')]
            elif value.startswith('(') and value.endswith(')'):
                config_dict[key.upper()] = re.compile(value)
            elif value.isdigit():
                config_dict[key.upper()] = int(value)
            else:
                config_dict[key.upper()] = value
    
    return config_dict
