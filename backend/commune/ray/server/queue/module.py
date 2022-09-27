import ray
import os,sys
sys.path.append(os.getenv('PWD'))
from ray.util.queue import Queue
from commune.utils import dict_put,dict_get,dict_has,dict_delete
from copy import deepcopy
from commune import BaseModule
"""

Background Actor for Message Brokers Between Quees

"""

class QueueServer(BaseModule):

    default_config_path = 'ray.server.queue'
    def __init__(self,config=None, **kwargs):
        BaseModule.__init__(self, config=config, **kwargs)
        self.queue = {}
        
    def delete_topic(self,topic,
                     force=False,
                     grace_period_s=5,
                     verbose=False):

        queue = self.queue.get(topic)
        if isinstance(queue, Queue) :
            queue.shutdown(force=force, grace_period_s=grace_period_s)
            if verbose:
                print(f"{topic} shutdown (force:{force}, grace_period(s): {grace_period_s})")
        else:
            if verbose:
                print(f"{topic} does not exist" )
        # delete queue topic in dict
        self.queue.pop(topic)


    def get_queue(self, topic, **kwargs):
        return self.queue.get(topic)
    
    def topic_exists(self, topic, **kwargs):
        return isinstance(self.queue.get(topic), Queue)



    def create_topic(self, topic:str,
                     maxsize:int=10,
                     actor_options:dict=None,
                     refresh=False,
                     verbose=False, **kwargs):


        if self.topic_exists(topic):
            if refresh:
                # kill the queue
                self.delete_topic(topic=topic,force=True)
                queue = Queue(maxsize=maxsize, actor_options=actor_options)

                if verbose:
                    print(f"{topic} Created (maxsize: {maxsize})")

            else:
                if verbose:
                    print(f"{topic} Already Exists (maxsize: {maxsize})")
        else:
            queue = Queue(maxsize=maxsize, actor_options=actor_options)
            if verbose:
                print(f"{topic} Created (maxsize: {maxsize})")

        self.queue[topic] = queue

        return queue 

    def list_topics(self, **kwargs):
        return list(self.queue.keys())

    ls = list_topics
    topics = property(list_topics)

    def put(self, topic, item, block=True, timeout=None, queue_kwargs=dict(maxsize=10,actor_options=None,refresh=False,verbose=False), **kwargs):
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
            queue_kwargs['topic'] = topic
            self.create_topic(**queue_kwargs)

        self.queue[topic].put(item, block=block, timeout=timeout)


    def get(self, topic, block=True, timeout=None, **kwargs):
        return self.queue[topic].get(block=block, timeout=timeout)


    def exists(self, topic, **kwargs):
        return bool(topic in self.topics)

    def delete_all(self):
        for topic in self.topics():
            self.delete_topic(topic, force=True)


    def get_topic(self, topic, *args, **kwargs):
        if dict_has(self.queue, topic):
            return dict_get(self.queue, topic)
        else:
            self.create_topic(topic=topic, **kwargs)
    def size(self, topic):
        # The size of the queue
        return self.get_topic(topic).size()

    def empty(self, topic):
        # Whether the queue is empty.

        return self.get_topic(topic).empty()

    def full(self, topic):
        # Whether the queue is full.
        return self.get_topic(topic).full()




class QueueClient(QueueServer):

    default_config_path = 'ray.server.queue'
    def __init__(self,config=None, **kwargs):
        BaseModule.__init__(self, config=config, **kwargs)
        self.queue = {}
        
    def delete_topic(self,topic,
                     force=False,
                     grace_period_s=5,
                     verbose=False):

        queue = self.queue.get(topic)
        if isinstance(queue, Queue) :
            queue.shutdown(force=force, grace_period_s=grace_period_s)
            if verbose:
                print(f"{topic} shutdown (force:{force}, grace_period(s): {grace_period_s})")
        else:
            if verbose:
                print(f"{topic} does not exist" )
        # delete queue topic in dict
        self.queue.pop(topic)


    def get_queue(self, topic):
        return self.queue.get(topic)
    
    def topic_exists(self, topic):
        return isinstance(self.queue.get(topic), Queue)



    def create_topic(self, topic,
                     maxsize=10,
                     actor_options=None,
                     refresh=False,
                     verbose=False, **kwargs):


        if self.topic_exists(topic):
            if refresh:
                # kill the queue
                self.delete_topic(topic=topic,force=True)
                queue = Queue(maxsize=maxsize, actor_options=actor_options)

                if verbose:
                    print(f"{topic} Created (maxsize: {maxsize})")

            else:
                if verbose:
                    print(f"{topic} Already Exists (maxsize: {maxsize})")
        else:
            queue = Queue(maxsize=maxsize, actor_options=actor_options)
            if verbose:
                print(f"{topic} Created (maxsize: {maxsize})")

        self.queue[topic] = queue

        return queue 

    def list_topics(self):
        return list(self.queue.keys())

    ls = list_topics
    topics = list_topics

    def put(self, topic, item, block=True, timeout=None, queue_kwargs=dict(maxsize=10,actor_options=None,refresh=False,verbose=False)):
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
            queue_kwargs['topic'] = topic
            self.create_topic(**queue_kwargs)

        self.queue[topic].put(item, block=block, timeout=timeout)


    def get(self, topic, block=True, timeout=None):
        return self.queue[topic].get(block=block, timeout=timeout)


    def exists(self, topic):
        return bool(topic in self.topics())

    def delete_all(self):
        for topic in self.topics:
            self.delete_topic(topic, force=True)
        


    def get_topic(self, topic, *args, **kwargs):
        if dict_has(self.queue, topic):
            return dict_get(self.queue, topic)
        else:
            self.create_topic(topic=topic, **kwargs)
    def size(self, topic):
        # The size of the queue
        return self.get_topic(topic).size()

    def empty(self, topic):
        # Whether the queue is empty.

        return self.get_topic(topic).empty()

    def full(self, topic):
        # Whether the queue is full.
        return self.get_topic(topic).full()


    def __del__(self):
        self.delete_all()

from functools import partial

class RayActorClient:
    def __init__(self, module):
        self.module =module
        for fn_key in module._ray_method_signatures.keys():

            def fn(self, fn_key,module, *args, **kwargs):
                ray_get = kwargs.pop('ray_get', False)
                object_id =(getattr(module, fn_key).remote(*args, **kwargs))
                if ray_get == True:
                    return ray.get(object_id)

                else:
                    return object_id

            setattr(self, fn_key, partial(fn, self, fn_key, module))
        
        


if __name__ == '__main__':
    import streamlit as st
    actor_name =  'queue_server'
    module = QueueServer.deploy(actor={'refresh':False, 'name': actor_name},ray={'address': 'auto', 'namespace': 'default'})
    
    st.write(module)
    client_module = QueueServer.import_module_class('ray.client.module.ClientModule')
    st.write(client_module(server=actor_name).put('bro', 'fam'))
    st.write(client_module(server=actor_name).get('bro', ray_get=True))
    # client = RayActorClient(module)
    # st.write(client.put('bro', {'whadup'}, ray_get=True))
    # st.write(client.get('bro', ray_get=True))
    # st.write(ray.get(module.ls.remote()))
    # st.write(ray.get(module.put.remote('bro', {'whatup'})))
    # st.write(ray.get(module.describe.remote())['QueueServer'].describe())

    # with module.get_ray_context({'address': 'auto', 'namespace': 'default'}) as r:
    #     st.write(r.__dict__)
