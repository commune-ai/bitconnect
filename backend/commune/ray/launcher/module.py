
from copy import deepcopy
import sys
import datetime
import os
import asyncio
import multiprocessing
import torch
sys.path.append(os.environ['PWD'])
from commune.config import ConfigLoader
import ray
from ray.util.queue import Queue
import torch
from commune.utils import  (RunningMean,
                        chunk,
                        get_object,
                        even_number_split,
                        torch_batchdictlist2dict,
                        round_sig,
                        timer,
                        tensor_dict_shape,
                        nan_check,
                        dict_put, dict_has, dict_get, dict_hash
                        )

from commune import BaseModule


def cache():
    def wrapper_fn(self,*args, **kwargs):
        self.save_
    

class Launcher(BaseModule):


    default_config_path = f"ray.launcher"

    def __inif__(self, config=None, **kwargs):
        BaseModule.__init__(self, config=config, **kwargs)
        self.actor_map = {}

        self.config['queue'] = {
            'in': 'launcher.in',
            'out': 'launcher.out'
        }

    @property
    def actor_map(self):
        self.config['actor_map'] = self.config.get('actor_map', {} )
        return self.config['actor_map']
        # self.spawn_actors()

    default_max_actor_count = 10
    @property
    def max_actor_count(self):
        return self.config.get('max_actor_count', self.default_max_actor_count)

    def get_job_results(job_kwargs):
        dict_hash(job_kwargs)

    def send_job(self, job_kwargs, block=False):
        print('BRO')
        self.client['ray'].queue.put(topic=self.config['queue']['in'], item=job_kwargs, block=block )
        

    def run_job(self, module, fn, kwargs={}, args=[], override={}, cron=None):

       
        actor, actor_name = self.launch_actor(module=module,override=override)
        job_id = getattr(actor, fn).remote(*args,**kwargs)
        job_kwargs = {'actor_name':actor_name,
                        'fn': fn,
                        'kwargs': kwargs}

        self.register_job(actor_name=actor_name, job_id=job_id)
        # if cron:
        #     self.register_cron(name=cron['name'], interval=cron['interval'], 
        #                         job = job_kwargs )

        self.client.ray.queue.put(topic=self.config['queue']['out'],item=job_id)
        return job_id

    @property
    def resource_limit(self):
        return {'gpu': torch.cuda.device_count(), 
                'cpu': multiprocessing.cpu_count()}


    # def load_balance(self, proposed_actor = None):

    #     while self.actor_count >= self.max_actor_count:
    #         for actor_name in self.actor_names:
    #             self.remove_actor(actor_name)
            
    def launch_actor(self, module, override={} , **kwargs):
        # self.load_balance(proposed_actor=actor_name)

        actor = self.get_module(config=module, actor=True, override=override) 
        
        actor_name = self.register_actor(actor=actor)
        return actor, actor_name

    def actor_exists(self, actor):
        actor_exists(actor)
    
    def register_actor(self, actor):
        actor_name = ray.get(actor.getattr.remote('actor_name'))
        self.actor_map[actor_name] = actor
        return  actor_name

    def get_actor_replicas(self, actor_name):
        return list(filter(lambda f: actor_name in self.actor_names[:len(actor_name)], self.actor_names))       


    def process(self, **kwargs):

        run_job_kwargs = self.client['ray'].queue.get(topic=self.config['queue']['in'], block=True )        
        # print(run_job_kwargs,'BRO')
        self.run_job(**run_job_kwargs)
        out_item = ray.get(self.get_jobs('finished'))


    @property
    def actor_names(self):
        return list(self.actor_map.keys())

    @property
    def actors(self):
        return list(self.actor_map.values())

    @property
    def actor_count(self):
        return len(self.actors)



    def remove_actor(self,actor):
        '''
        params:
            actor: name of actor or handle
        '''
        self.get_config()

        assert actor in self.actor_map, 'Please specify an actor in the actor map'
        self.kill_actor(actor)
        del self.actor_map[actor]
        self.put_config()

  
    rm = remove = remove_actor

    def remove_all_actors(self):
        for actor in self.actor_names:
            self.remove_actor(actor)

    rm_all = remove_all = remove_all_actors



if __name__=="__main__":
    import streamlit as st
    module = Launcher.deploy()

    module.bro()

    st.write(module.cache)

    

