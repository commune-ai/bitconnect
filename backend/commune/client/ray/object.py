import ray
import os
from commune.ray.actor import ActorBase
"""

Background Actor for Message Brokers Between Quees

"""
class RayObjectClient:
    def __init__(self, cfg = {}):
        self.cfg = cfg
        self.server = RayObjectServer.deploy(actor=True)

    def put(self,*args, **kwargs):
        return ray.get(self.server.put.remote(*args, **kwargs))
    def get(self, *args, **kwargs):
        return ray.get(ray.get(self.server.get.remote(*args, **kwargs)))
        


class RayObjectServer(ActorBase):
    default_cfg_path= f"{os.environ['PWD']}/commune/client/ray/object.yaml"
    def __init__(self, cfg):
        self.cfg = cfg
        self.cache = {}
    def put(self,key,value):
        self.cache[key] = ray.put(value)
    def get(self, key, block=False):
        if block:
            while True:
                if key in self.cache:
                    return self.cache[key]  
        return self.cache[key]

    def ls(self, key):
        return [k.startswith(key) for k in self.cache.keys()]

    def delete(self, key):
        del self.cache[key]

    
    def list_objects(self, key=''):
        object_list = []
        for k,v in self.cache.items():
            if bool(key) and key in k:
                object_list += [v]
        return object_list
    def has_object(self, object_key):
        return bool(object_key in self.objects)
