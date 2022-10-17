
from copy import deepcopy
import sys
import datetime
import os
import asyncio
import multiprocessing
import torch
import pandas as pd
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
    

class APIModule(BaseModule):


    default_config_path = "bittensor.cortex.api"

    def __inif__(self, config=None, **kwargs):
        BaseModule.__init__(self, config=config, **kwargs)
        self.actor_map = {}



    def actor_info_map(self):
        info_map = {}
        for actor_name in self.actor_names:
            info_map[actor_name] = self.get_actor(actor_name, wrap=True).actor_info()
        
        return info_map

    def actor_resource_map(self):
        info_map = {}
        for actor_name in self.actor_names:
            info_map[actor_name] = self.get_actor(actor_name, wrap=True).resource_usage()
        
        return info_map

    @property
    def actor_map(self):
        self.get_config()
        actor_names = self.config.get('actor_names', [])
        actor_map = {}
        for actor_name in actor_names:
            if self.actor_exists(actor_name):
                actor_map[actor_name] = ray.get_actor(actor_name)
            
        self.put_config()
        return actor_map
        # self.spawn_actors()

    default_max_actor_count = 10
    @property
    def max_actor_count(self):
        return self.config.get('max_actor_count', self.default_max_actor_count)

    def get_job_results(job_kwargs):
        dict_hash(job_kwargs)

    def send_job(self, job_kwargs, block=False):
        self.client['ray'].queue.put(topic=self.config['queue']['in'], item=job_kwargs, block=block )
        

    def submit_job(self, module, fn, kwargs={}, args=[], ):


        assert self.actor_exists(module), f'{module} does not exists'
        actor = self.get_actor(module)
        job_id = getattr(actor, fn).remote(*args,**kwargs)

        self.queue.put(topic='api.in',item=job_id)
        
        return job_id

    @property
    def resource_limit(self):
        return {'gpu': torch.cuda.device_count(), 
                'cpu': multiprocessing.cpu_count(),
                'memory': 1}

    # def load_balance(self, proposed_actor = None):

    #     while self.actor_count >= self.max_actor_count:
    #         for actor_name in self.actor_names:
    #             self.remove_actor(actor_name)
            

    def resolve_actor_class(self,module):
        assert module in self.simple2module, f'options are {list(self.simple2module.keys())}'
        module_path = self.simple2module[module]
        actor_class= self.get_object(module_path)
        return actor_class

    def add_actor(self, module, 
                    name= None,
                    tag=None,
                    refresh=False,
                    resources={'num_gpus':0, 'num_cpus': 1},
                    wrap = False,
                     **kwargs):

        actor_class = self.resolve_actor_class(module)

        if kwargs.get('refresh_cache')==True:
            self.rm_config()

        self.get_config()

        if not isinstance(name, str):
            name = module
            if isinstance(tag, str ):
                name = module +'-'+ tag 

        kwargs['actor'] = kwargs.get('actor',  {})
        kwargs['actor'].update(dict(refresh=refresh, name=name, resources=resources))


        actor_class = self.resolve_actor_class(module)

        actor = actor_class.deploy(**kwargs)

        # # st.write(actor.__dict__)

        self.actor_map[name] = actor.id
        self.config['actor_names'] = self.actor_names
        self.config['actor_names'].append(name)
        self.put_config()

        if wrap:
            actor = self.wrap_actor(actor)

        return actor

        
        # return actor

    get_actor = add_actor
    def get_actor_replicas(self, actor_name):
        return self.list_actors(actor_name)

    get_replicas = get_actor_replicas


    @property
    def actor_names(self):
        self.get_config()
        return list(self.actor_map.keys())

    @property
    def actors(self):
        self.get_config()
        return list(self.actor_map.values())

    @property
    def actor_count(self):
        return len(self.actors)

    def running_actors(self, mode='name'):
        if mode in ['names', 'name']:
            return self.actor_names()
        elif mode in ['actors', 'actor']:
            return self.actors

    def total_resource_usage(self):
        total_resource_usage = self.resource_usage()
        for actor_resource_usage in self.actor_resource_map().values():
            for k,v in actor_resource_usage.items():
                total_resource_usage[k] += v
        
        return total_resource_usage

    def list_actors(self, key=None):
        actor_names = self.actor_names
        if key == None:
            return actor_names
        else:
            return [a for a in actor_names if a.startswith(key)]   


    def remove_actor(self,actor):
        '''
        params:
            actor: name of actor or handle
        '''
        self.get_config()

        assert actor in self.actor_map, 'Please specify an actor in the actor map'
        self.kill_actor(actor)
        self.config['actor_names'] = self.actor_names
        self.put_config()

    rm_actor = rm = remove = remove_actor

    def actor_df(self):

        df = []
        for actor_name, actor_info in self.actor_info_map().items():
            df.append({
                'name': actor_info['name'],
                'id': actor_info['id'],
                'cpus': actor_info['resources'].get('cpus', 0),
                'gpus': actor_info['resources'].get('gpus', 0),
                'memory': actor_info['resources'].get('memory', 0),
            })

        return pd.DataFrame(df)
    @staticmethod
    def st_test():

        import streamlit as st
        module = APIModule.deploy(actor={'refresh': True}, wrap=True)

        st.write(module.getattr('module_tree'))
        st.write(module.list_actors())
        # actor = module.add_actor('algovera.base', refresh=True, wrap=True)
        st.write(module.actor_info_map())
        st.write(module.actor_resource_map())
        st.write(module.actor_df())
        # st.write(actor.get_name(), actor.get_resources())
        # st.write(actor.get_id())
        # st.write(module.getattr('actor_map'))

        # module.add_actor('algovera.base-2')
        # st.write(module.getattr('actor_map'))


    def remove_all_actors(self):
        for actor in self.actor_names:
            self.remove_actor(actor)

    rm_all = remove_all = remove_all_actors




if __name__=="__main__":


    APIModule.st_test()