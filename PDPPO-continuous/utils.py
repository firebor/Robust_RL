import numpy as np
import torch


class Dict(dict):
    def __init__(self, config, section_name, location=False):
        super(Dict, self).__init__()
        self.initialize(config, section_name, location)

    def initialize(self, config, section_name, location):
        for key, value in config.items(section_name):
            if location:
                self[key] = value
            else:
                self[key] = eval(value)

    def __getattr__(self, val):
        return self[val] 