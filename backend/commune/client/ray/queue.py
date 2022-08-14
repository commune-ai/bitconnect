import ray
from ray.util.queue import Queue
import os, sys
sys.path.append(os.environ['PWD'])
from commune.utils.misc import (dict_put,
                     dict_get,
                     dict_has,
                      dict_delete)
from commune.ray.actor import ActorBase
os.environ['NUMEXPR_MAX_THREADS'] = str(os.cpu_count())
"""

Background Actor for Message Brokers Between Quees
 
"""

import fcntl, hashlib

class SystemMutex:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        lock_id = hashlib.md5(self.name.encode('utf8')).hexdigest()
        self.fp = open(f'/tmp/.lock-{lock_id}.lck', 'wb')
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_EX)

    def __exit__(self, _type, value, tb):
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_UN)
        self.fp.close()

class RayQueueClient:
    def __init__(self, cfg={}):
        self.cfg = cfg
        self.server = RayQueueServer.deploy(actor=True)
    def delete_topic(self,*args, **kwargs):
        return ray.get(self.server.delete_topic.remote(*args, **kwargs))
    def create_topic(self, *args, **kwargs):
        return ray.get(self.server.create_topic.remote(*args, **kwargs))
    def put(self, *args, **kwargs):
        return ray.get(self.server.put.remote(*args, **kwargs))
    def get(self, *args, **kwargs):
        return ray.get(self.server.get.remote(*args, **kwargs))
    def getattr(self, *args, **kwargs):
        return ray.get(self.server.getattr.remote(*args, **kwargs))
    def exists(self, *args, **kwargs):
        return ray.get(self.server.exists.remote(*args, **kwargs))

    def delete_all(self, *args, **kwargs):
        ray.get(self.server.delete_all.remote( *args, **kwargs))
    def size(self,  *args, **kwargs):
        # The size of the queue
        return ray.get(self.server.size.remote( *args, **kwargs))

    def empty(self, *args, **kwargs):
        return ray.get(self.server.empty.remote( *args, **kwargs))

    def full(self,  *args, **kwargs):
        # Whether the queue is full.
        return ray.get(self.server.full.remote( *args, **kwargs))

class RayQueueServer(ActorBase):
    default_cfg_path= f"{os.environ['PWD']}/commune/client/ray/queue.yaml"
    def __init__(self, cfg):
        self.cfg = cfg
        self.queue = {}

    def delete_topic(self,topic,
                     force=False,
                     grace_period_s=5,
                     verbose=False):
        if dict_has(self.queue, topic):
            self.queue[topic].shutdown(force=force,grace_period_s=grace_period_s)
            dict_delete(self.queue, topic)
            if verbose:
                print(f"{topic} shutdown (force:{force}, grace_period(s): {grace_period_s})")

        else:
            if verbose:
                print(f"{topic} does not exist" )
        # delete queue topic in dict


    def create_topic(self, topic,
                     maxsize=10,
                     actor_options=None,
                     refresh=False,
                     verbose=False):


        with SystemMutex('create-topic-queue'):
            if  topic in self.queue:
                if refresh:
                    # kill the queue
                    self.delete_topic(topic=topic,force=True)
                    queue = Queue(maxsize=maxsize, actor_options=actor_options)
                else: 
                    queue = self.queue[topic]

            else:
                queue = Queue(maxsize=maxsize,
                                            actor_options=actor_options)
            
            
            self.queue[topic] = queue


    def put(self, topic, item, block=True, timeout=None, queue_kwargs=dict(maxsize=100,actor_options=None,refresh=False,verbose=False)):
        """
        Adds an item to the queue.

            If block is True and the queue is full, blocks until the queue is no longer full or until timeout.
            There is no guarantee of order if multiple producers put to the same full queue.
            
            Raises
                Full – if the queue is full and blocking is False.
                
                Full – if the queue is full, blocking is True, and it timed out.
                
                ValueError – if timeout is negative.
        """

        if not self.exists(topic):
            self.create_topic(topic=topic, **queue_kwargs)

        self.queue[topic].put(item, block=block, timeout=timeout)




    def get(self, topic, block=True, timeout=None, queue_kwargs=dict(maxsize=10,actor_options=None,refresh=False,verbose=False)):
        if not self.exists(topic):
            self.create_topic(topic=topic, **queue_kwargs)

        if self.size(topic) == 0 and block == False:
            return None

        return self.queue[topic].get(block=block, timeout=timeout)


    def exists(self, topic):
        return bool(topic in self.queue)

    def delete_all(self):
        for topic in self.queue:
            
            self.delete_topic(topic, force=True)


    def size(self, topic):
        # The size of the queue
        if self.exists(topic): 
            return self.queue[topic].size()
        else:
            return 0

    def empty(self, topic):
        # Whether the queue is empty.
        if self.exists(topic):
            return self.queue[topic].empty()

    def full(self, topic):
        # Whether the queue is full.
        return self.get_topic(topic).full()

