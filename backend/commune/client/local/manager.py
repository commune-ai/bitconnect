from minio import Minio
import pickle
import os
import io
from commune.ray.actor import ActorBase

from .utils import load_json, write_json

class LocalManager(ActorBase):
    default_cfg_path = f"{os.environ['PWD']}/commune/config/client/block/local.yaml"
    def __init__(self, cfg):
        self.cache = {}
        self.pwd = cfg['pwd']

    def write(self, path, data, type='json'):
        if type == 'json': 
            write_json(path=path, data=data)
        else:
            raise(f"{type} not currently supported")

    def load(self, path,type='json'):
        if type == 'json': 
            return load_json(path=path)
        else:
            raise(f"{type} not currently supported")
