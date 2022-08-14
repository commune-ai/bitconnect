import ray
from ray.util.queue import Queue

"""

Background Actor for Message Brokers Between Quees

"""
from commune.ray.utils import kill_actor, create_actor

class RayObjectServer(object):
    def __init__(self, cfg):
        self.cache = {}
    def put(self,key,value):
        self.cache[key] = ray.put(value)
    def get(self, key):
        return self.cache[key]

    def list_objects(self):
        return list(self.cache.values())

    def has_object(self, object_key):
        return bool(object_key in self.objects)

    @classmethod
    def deploy(cls, cfg=None, actor=True):
        if cfg is None:
            cfg = cls.load_config()

        if actor: 
            return create_actor(cls=cls
                    actor_kwargs={'cfg': cfg},
                    actor_name=cfg['actor']['name'],
                    resources=cfg['actor']['resources'],
                    max_concurrency=cfg['actor']['max_concurrency'],
                    detached=True,
                    return_actor_handle=True,
                    refresh=False)

        else:
            return cls(cfg=cfg)

    




