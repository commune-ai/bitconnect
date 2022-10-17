
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
        actor_names = deepcopy(list(self.config['running_actor_map'].keys()))
        actor_map = {}
        for actor_name in actor_names:
            if self.actor_exists(actor_name):
                actor_map[actor_name] = ray.get_actor(actor_name)
            else:
                del self.config['running_actor_map'][actor_name]
            
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
                'cpu': multiprocessing.cpu_count(),
                'memory_percent': 0.5}

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

    def add_actor_replicas(self, module, replicas=1,*args, **kwargs):
        module_name_list = [f"{module}-{i}" for i in range(replicas)]
        for module_name in module_name_list:
            self.add_actor(module=module_name, *args, **kwargs)
    def add_actor(self, module, 

                    actor=dict(refresh=False, resources={'num_gpus':0, 'num_cpus': 1}),
                    refresh_cache=False,
                     **kwargs):


        actor['name'] = deepcopy(module)


        for k in ['refresh', 'resources', 'max_concurrency']:
            if k in kwargs:
                actor[k] = kwargs.pop(k)

        if len(module.split('-')) == 1:
            pass
        elif len(module.split('-')) == 2:
            module,tag = module.split('-')
        elif len(module.split('-')) == 3:
            module,tag,replica = module.split('-')
        else:
            raise Exception('please use one "-" in the format of  {module} or {module}-{tag} or "{module}-{tag}-{replica}" or ')

        if refresh_cache:
            self.rm_config()


        self.get_config()
        kwargs['actor'] = actor
        actor_class = self.resolve_actor_class(module)
        module_actor = actor_class.deploy(**kwargs)

        # # st.write(actor.__dict__)

        self.config['running_actor_map'] = self.config.get('running_actor_map', {})
        self.config['running_actor_map'][actor['name']] = kwargs['actor']

        self.put_config()

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

    def remove_actor(self,actor_name,timeout=10):
        '''
        params:
            actor: name of actor or handle
        '''
        self.get_config()

        assert actor_name in self.actor_map, f'Please specify an actor in the actor map OPTIONS: {self.actor_names}'
        self.kill_actor(actor_name)
        assert not self.actor_exists(actor_name)
        del self.config['running_actor_map'][actor_name]

        self.put_config()

    @staticmethod
    def st_test():

        import streamlit as st
        module = Launcher.deploy(actor=False, wrap=True)
        # # st.write(module.module_tree)
        # actor = module.get_actor('algovera.base-1')
        # module.add_actor(module=f'bittensor.receptor.pool-{0}', wallet=None, refresh=False)
        # # st.write(module.remove_actor('algovera.base-3'))
        st.write(module.get_actor('bittensor.receptor.pool-0'))
        # # module.remove_all_actors()
        # st.write(ray.get(ray.get_actor('ray.server.queue').put.remote('bro', 'bro')))
        # st.write(module.get_actor('bittensor.receptor.receptor'))
        # st.write(module.getattr('available_modules'))
        # st.write(actor.get_name(), actor.get_resources())
        # st.write(actor.get_id())
        # st.write(module.getattr('actor_map'))

        # module.add_actor('algovera.base-2')
        # st.write(module.getattr('actor_map'))

    @property
    def available_modules(self):
        return list(self.simple2module.keys())

    module_list = available_modules
    

  
    rm = remove = remove_actor



    def remove_all_actors(self):
        for actor in self.actor_names:
            self.remove_actor(actor)

    rm_all = remove_all = remove_all_actors




if __name__=="__main__":


    Launcher.st_test()