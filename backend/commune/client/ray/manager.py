from .object import RayObjectClient
from .queue import RayQueueClient
from commune.ray import ActorBase
import os
class RayManager(ActorBase):
    default_cfg_path=f"{os.environ['PWD']}/commune/config/client/block/ray.yaml"
    def __init__(self, cfg):
        self.queue = RayQueueClient(cfg=cfg['queue'])
        self.object = RayObjectClient(cfg=cfg['object'])

    def write(self,  data , topic, mode='queue', **kwargs):
        if mode == 'queue':
            self.queue.put(topic=topic, item=data)
        elif mode == 'object':
            key = kwargs.get('key', topic)
            value = kwargs.get('value', data)
            self.object.put(key=key, value=value)


    def read(self, topic, mode='queue', block=False, **kwargs):
        if mode == 'queue':
            return_obj = self.queue.get(topic=topic, block=block)
        elif mode == 'object':
            key = kwargs.get('key', topic)
            return_obj =  self.object.get(key=key, block=block)
        
        return return_obj

    def load(*args, **kwargs):
        return self.read(*args, **kwargs)
    def save(*args, **kwargs):
        return self.write(*args, **kwargs)