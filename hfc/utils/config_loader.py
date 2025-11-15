import yaml
from easydict import EasyDict

def load_config(config_path):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return EasyDict(config_dict)
