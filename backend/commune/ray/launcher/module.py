
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


    default_config_path = "ray.launcher"

    def __inif__(self, config=None, **kwargs):
        BaseModule.__init__(self, config=config, **kwargs)
        self.actor_map = {}

        self.config['queue'] = {
            'in': 'launcher.in',
            'out': 'launcher.out'
        }

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
        

    def run_job(self, module, fn, kwargs={}, args=[], override={}, cron=None):

        actor = self.add_actor(module=module,override=override)
        job_id = getattr(actor, fn).remote(*args,**kwargs)
        job_kwargs = {'actor_name':actor.actor_name,
                        'fn': fn,
                        'kwargs': kwargs}

        self.register_job(actor_name=actor.actor_name, job_id=job_id)
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
            


    def resolve_module(self, module):

        return module

    def resolve_actor_class(self,module):
        assert module in self.simple2module, f'options are {list(self.simple2module.keys())}'
        module_path = self.simple2module[module]
        actor_class= self.get_object(module_path)
        return actor_class

    def add_actor(self, module, 
                    tag=None,
                    refresh=False,
                    resources={'num_gpus':0, 'num_cpus': 1},
                    wrap = False,
                     **kwargs):


        if kwargs.get('refresh_cache')==True:
            self.rm_config()

        self.get_config()


        if tag == None:
            if len(module.split('-')) == 1:
                module = module
            elif len(module.split('-')) == 2:
                module, tag = module.split('-')


        actor_class = self.resolve_actor_class(module)
        actor_name = actor_class.get_module_path()
        if tag != None:
            actor_name = '-'.join([actor_name, tag])
        kwargs['actor'] = kwargs.get('actor',  {})
        kwargs['actor'] = dict(refresh=refresh, name=actor_name, resources=resources)




        actor = actor_class.deploy(**kwargs)

        # # st.write(actor.__dict__)


        self.actor_map[actor_name] = actor.id
        self.config['actor_names'] = self.actor_names
        self.config['actor_names'].append(actor_name)

        self.put_config()

        if wrap:
            actor = self.wrap_actor(actor)

        return actor

        
        # return actor

    get_actor = add_actor
    def get_actor_replicas(self, actor_name):
        return list(filter(lambda f: actor_name == self.actor_names[:len(actor_name)], self.actor_names))       

    get_replicas = get_actor_replicas

    def process(self, **kwargs):
        self.get_config()
        run_job_kwargs = self.client['ray'].queue.get(topic=self.config['queue']['in'], block=True )        
        # print(run_job_kwargs,'BRO')
        self.run_job(**run_job_kwargs)
        out_item = ray.get(self.get_jobs('finished'))


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

    @staticmethod
    def st_test():

        import streamlit as st
        module = Launcher.wrap_actor(Launcher.deploy(actor={'refresh': False}))
        # # st.write(module.module_tree)
        actor = Launcher.wrap_actor(module.get_actor('algovera.base-1', refresh=True))
        st.write(actor.actor_info())
        # st.write(actor.get_name(), actor.get_resources())
        # st.write(actor.get_id())
        # st.write(module.getattr('actor_map'))

        # module.add_actor('algovera.base-2')
        # st.write(module.getattr('actor_map'))
        



  
    rm = remove = remove_actor



    def remove_all_actors(self):
        for actor in self.actor_names:
            self.remove_actor(actor)

    rm_all = remove_all = remove_all_actors




if __name__=="__main__":


    Launcher.st_test()