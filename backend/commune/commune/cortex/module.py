from __future__ import annotations
from commune.cortex.utils import *
import datetime
from commune.config.loader import ConfigLoader
import streamlit as st
import os
import subprocess, shlex
import ray
import torch
import gradio as gr
import socket
import json
from importlib import import_module
from munch import Munch
import types
import inspect
from commune.ray.utils import kill_actor
from copy import deepcopy
import argparse
import psutil
import gradio
import asyncio
from ray.experimental.state.api import list_actors, list_objects, list_tasks
import streamlit as st
import nest_asyncio
from typing import *
from glob import glob



class Module:


    @classmethod
    def launch(cls, module:str, fn:str=None ,kwargs:dict={}, args=[], actor=False, **additional_kwargs):
        module_class = cls.import_object(module)
        module_kwargs = {**kwargs}
        module_args = [*args]
        module_init_fn = fn
        if module_init_fn == None:
            if actor:
                # ensure actor is a dictionary
                if actor == True:
                    actor = {}
                assert isinstance(actor, dict), f'{type(actor)} should be dictionary fam'
                default_actor_name = module_class.__name__
                actor['name'] = actor.get('name', default_actor_name )
                module_object = cls.create_actor(cls=module_class, cls_kwargs=module_kwargs, cls_args=module_args, **actor)

            else:
                module_object =  module_class(*module_args,**module_kwargs)

        else:
            module_init_fn = getattr(module_class,module_init_fn)
            module_object =  module_init_fn(*module_args, **module_kwargs)

        return module_object
    launch_module = launch

    @classmethod
    def load_module(cls, path):
        prefix = f'{cls.root_dir}.'
        if cls.root_dir+'.' in path[:len(cls.root_dir+'.')]:
            path = path.replace(f'{cls.root_dir}.', '')
        module_path = cls.simple2path(path)
        module_class =  cls.import_object(module_path)
        return module_class

    ############# TIME LAND ############
    @property
    def current_timestamp(self):
        return self.get_current_timestamp()

    def current_datetime(self):
        datetime.datetime.fromtimestamp(self.current_timestamp)
    
    def start_datetime(self):
        datetime.datetime.fromtimestamp(self.start_timestamp)
    
    def get_age(self) :
        return  self.get_current_timestamp() - self.start_timestamp

    age = property(get_age) 
    @staticmethod
    def get_current_timestamp():
        return  datetime.datetime.utcnow().timestamp()

    
    @classmethod
    def import_module(cls, import_path:str) -> 'Object':
        return import_module(import_path)

    @classmethod
    def import_object(cls, key:str)-> 'Object':
        module = '.'.join(key.split('.')[:-1])
        object_name = key.split('.')[-1]
        print(module, object_name,  'BROOO1',  import_module(module)) 
        obj =  getattr(import_module(module), object_name)
        print(obj,  'BROOO2') 
        return obj
    get_object = import_object
    

    def class_name(self):
        return self.__class__.__name__

    @staticmethod
    def is_hidden_function(fn):
        if isinstance(fn, str):
            return fn.startswith('__') and fn.endswith('__')
        else:
            raise NotImplemented(f'{fn}')

    @staticmethod
    def get_functions(object):
        functions = get_functions(object)
        return functions

    @staticmethod
    def get_annotations(fn:callable) -> dict:
        return fn.__annotations__
    
    def is_parent(child, parent):
        return bool(parent in Module.get_parents(child))

    @classmethod
    def get_parents(cls, obj=None):
        if obj == None:
            obj = cls

        return list(obj.__mro__[1:-1])

    @classmethod
    def is_module(cls, obj=None):
        if obj == None:
            obj = cls
        return Module in cls.get_parents(obj)

    @classmethod
    def functions(cls, obj=None, return_type='str', **kwargs):
        if obj == None:
            obj = cls
        functions =  get_functions(obj=obj, **kwargs)
        if return_type in ['str', 'string']:
            return functions
        
        elif return_type in ['func', 'fn','functions']:
            return [getattr(obj, f) for f in functions]
        else:
            raise NotImplementedError

    @classmethod
    def hasfunc(cls, key):
        fn_list = cls.functions()
        return bool(len(list(filter(lambda f: f==key, fn_list)))>0)

    @classmethod
    def filterfunc(cls, key):
        fn_list = cls.functions()
        ## TODO: regex
        return list(filter(lambda f: key in f, fn_list))


    @classmethod
    def run_python(cls):
        cls.run_command(f'python {path}')


    @classmethod
    def parents(cls):
        return get_parents(cls)

    @classmethod
    def describe_module_schema(cls, obj=None, **kwargs):
        if obj == None:
            obj = cls
        return get_module_function_schema(obj, **kwargs)

    @property
    def module_path(self):
        return self.get_module_path()

    @staticmethod
    def run_command(command:str):

        process = subprocess.run(shlex.split(command), 
                            stdout=subprocess.PIPE, 
                            universal_newlines=True)
        return process

    @property
    def tmp_dir(self):
        return f'/tmp/{self.root_dir}/{self.class_name}'


    @classmethod
    def get_module_path(cls, obj=None,  simple=True):
        if obj == None:
            obj = cls
        module_path =  inspect.getmodule(obj).__file__
        # convert into simple
        if simple:
            module_path = cls.path2simple(path=module_path)

        return module_path


    ###########################
    #   RESOURCE LAND
    ###########################
    @staticmethod
    def check_pid(pid):        
        return check_pid(pid)

    @staticmethod
    def kill_pid(pid):        
        return kill_pid(pid)
    @property
    def pid(self):
        return os.getpid()

    def memory_usage(self, mode='gb'):
        '''
        get memory usage of current process in bytes
        '''
        import os, psutil
        process = psutil.Process(self.pid)
        usage_bytes = process.memory_info().rss
        usage_percent = process.memory_percent()
        mode_factor = 1

        if mode  in ['gb']:
            mode_factor = 1e9
        elif mode in ['mb']:
            mode_factor = 1e6
        elif mode in ['b']:
            mode_factor = 1
        elif mode in ['percent','%']:
            return usage_percent
        elif mode in ['ratio', 'fraction', 'frac']: 
            return usage_percent / 100
        else:
            raise Exception(f'{mode} not supported, try gb,mb, or b where b is bytes')

        return usage_bytes / mode_factor

    @staticmethod
    def memory_available(mode ='percent'):

        memory_info = Module.memory_info()
        available_memory_bytes = memory_info['available']
        available_memory_ratio = (memory_info['available'] / memory_info['total'])
    
        mode_factor = 1
        if mode  in ['gb']:
            mode_factor = 1e9
        elif mode in ['mb']:
            mode_factor = 1e6
        elif mode in ['b']:
            mode_factor = 1
        elif mode in ['percent','%']:
            return  available_memory_ratio*100
        elif mode in ['fraction','ratio']:
            return available_memory_ratio
        else:
            raise Exception(f'{mode} not supported, try gb,mb, or b where b is bytes')

        return usage_bytes / mode_factor


    @staticmethod
    def memory_used(mode ='percent'):

        memory_info = Module.memory_info()
        available_memory_bytes = memory_info['used']
        available_memory_ratio = (memory_info['used'] / memory_info['total'])
    
        mode_factor = 1
        if mode  in ['gb']:
            mode_factor = 1e9
        elif mode in ['mb']:
            mode_factor = 1e6
        elif mode in ['b']:
            mode_factor = 1
        elif mode in ['percent','%']:
            return  available_memory_ratio*100
        elif mode in ['fraction','ratio']:
            return available_memory_ratio
        else:
            raise Exception(f'{mode} not supported, try gb,mb, or b where b is bytes')

        return usage_bytes / mode_factor

    @staticmethod
    def memory_info():
        virtual_memory = psutil.virtual_memory()
        return {k:getattr(virtual_memory,k) for k in ['available', 'percent', 'used', 'shared', 'free', 'total', 'cached']}

    @staticmethod
    def get_memory_info(pid:int = None):
        if pid == None:
            pid = os.getpid()
        # return the memory usage in percentage like top
        process = psutil.Process(pid)
        memory_info = process.memory_full_info()._asdict()
        memory_info['percent'] = process.memory_percent()
        memory_info['ratio'] = memory_info['percent'] / 100
        return memory_info


    def resource_usage(self):
        resource_dict =  self.config.get('actor', {}).get('resources', None)
        resource_dict = {k.replace('num_', ''):v for k,v in resource_dict.items()}
        resource_dict['memory'] = self.memory_usage(mode='ratio')
        return  resource_dict


    ##############
    #   RAY LAND
    ##############
    @classmethod
    def get_default_actor_name(cls):
        return cls.get_module_path(simple=True)


    @classmethod
    def ray_stop(cls):
        cls.run_command('ray stop')

    @classmethod
    def ray_start(cls):
        cls.run_command('ray start --head')

    @classmethod
    def ray_restart(cls):
        cls.ray_stop()
        cls.ray_start()

    @classmethod
    def ray_status(cls):
        cls.run_command('ray status')

    @staticmethod
    def ray_initialized():
        return ray.is_initialized()

    @property
    def actor_id(self):
        return self.get_id()

    default_ray_env = {'address':'auto', 
                     'namespace': 'default',
                      'ignore_reinit_error': False,
                      'dashboard_host': '0.0.0.0'}
    @classmethod
    def ray_init(cls,init_kwargs={}):

        # init_kwargs['_system_config']={
        #     "object_spilling_config": json.dumps(
        #         {"type": "filesystem", "params": {"directory_path": "/tmp/spill"}},
        #     )
        # }
        init_kwargs =  {**cls.default_ray_env, **init_kwargs}
        if cls.ray_initialized():
            # shutdown if namespace is different
            if cls.ray_namespace() == cls.default_ray_env['namespace']:
                return cls.ray_runtime_context()
            else:
                ray.shutdown()
  
        ray_context = ray.init(**init_kwargs)
        return ray_context

    init_ray = ray_init
    @staticmethod
    def create_actor(cls,
                 name, 
                 cls_kwargs,
                 cls_args =[],
                 detached=True, 
                 resources={'num_cpus': 1.0, 'num_gpus': 0},
                 cpus = 0,
                 gpus = 0,
                 max_concurrency=50,
                 refresh=False,
                 return_actor_handle=False,
                 verbose = True,
                 redundant=False,
                 tag_seperator = '-',
                 tag = None,
                 wrap = False,
                 **kwargs):

        if cpus > 0:
            resources['num_cpus'] = cpus
        if gpus > 0:
            resources['num_gpus'] = gpus

        if not torch.cuda.is_available() and 'num_gpus' in resources:
            del resources['num_gpus']

        # configure the option_kwargs

        if tag != None:
            tag = str(tag)
            name = tag_seperator.join([name, tag])

        options_kwargs = {'name': name,
                          'max_concurrency': max_concurrency,
                           **resources}
        if detached:
            options_kwargs['lifetime'] = 'detached'
        # setup class init config
        # refresh the actor by killing it and starting it (assuming they have the same name)
        
        if refresh:
            if Module.actor_exists(name):
                kill_actor(actor=name,verbose=verbose)
                # assert not Module.actor_exists(name)

        if redundant:
            # if the actor already exists and you want to create another copy but with an automatic tag
            actor_index = 0
            while Module.actor_exists(name):
                name =  f'{name}-{actor_index}' 
                actor_index += 1

        if not Module.actor_exists(name):
            actor_class = ray.remote(cls)
            actor_handle = actor_class.options(**options_kwargs).remote(*cls_args, **cls_kwargs)

        actor = Module.get_actor(name)

        if wrap:
            actor = Module.wrap_actor(actor)

        return actor

    @staticmethod
    def get_actor_id( actor):
        assert isinstance(actor, ray.actor.ActorHandle)
        return actor.__dict__['_ray_actor_id'].hex()


    @classmethod
    def wrap_actor(cls, actor):
        # wrapper_module_path = 'commune.ray.client.module.ClientModule'
        # return Module.get_module(module=wrapper_module_path, server=actor)
        return actor

    @classmethod
    def deploy_module(cls, module:str, **kwargs):
        module_class = cls.import_object(module)
        return module_class.deploy(**kwargs)
    get_module = deploy_module

    @staticmethod
    def kill_actor(actor, verbose=True):

        if isinstance(actor, str):
            if Module.actor_exists(actor):
                actor = ray.get_actor(actor)
            else:
                if verbose:
                    print(f'{actor} does not exist for it to be removed')
                return None
        
        return ray.kill(actor)
        
    @staticmethod
    def kill_actors(actors):
        return_list = []
        for actor in actors:
            return_list.append(Module.kill_actor(actor))
        
        return return_list
            
    @staticmethod
    def actor_exists(actor):
        if isinstance(actor, str):
            try:
                ray.get_actor(actor)
                actor_exists = True
            except ValueError as e:
                actor_exists = False
            
            return actor_exists
        else:
            raise NotImplementedError

    @staticmethod
    def get_actor(actor_name, wrap=False):
        actor =  ray.get_actor(actor_name)
        # actor = Module.add_actor_metadata(actor)
        if wrap:
            actor = Module.wrap_actor(actor=actor)
        return actor

    @property
    def ray_context(self):
        return self.init_ray()

    @staticmethod
    def ray_runtime_context():
        return ray.get_runtime_context()

    @classmethod
    def ray_namespace(cls):
        return ray.get_runtime_context().namespace

    @staticmethod
    def get_ray_context():
        return ray.runtime_context.get_runtime_context()
    @property
    def context(self):
        if Module.actor_exists(self.actor_name):
            return self.init_ray()

    @property
    def actor_name(self):
        actor_config =  self.config.get('actor', {})
        if actor_config == None:
            actor_config = {}
        return actor_config.get('name')
    
    @property
    def default_actor_name(self):
        return self.get_module_path(simple=True)
    @property
    def actor_running(self):
        return self.is_actor_running

    def is_actor_running(self):
        return isinstance(self.actor_name, str)

    @property
    def actor_config(self):
        return self.config.get('actor',None)

    @property
    def actor_handle(self):
        if not hasattr(self, '_actor_handle'):
            self._actor_handle = self.get_actor(self.actor_name)
        return self._actor_handle


    @staticmethod
    def list_objects( *args, **kwargs):
        return ray.experimental.state.api.list_objects(*args, **kwargs)

    @staticmethod
    def list_actors(state='ALIVE', detail=True, *args, **kwargs):
        kwargs['filters'] = kwargs.get('filters', [("state", "=", state)])
        kwargs['detail'] = detail

        actor_info_list =  list_actors(*args, **kwargs)
        final_info_list = []
        for i, actor_info in enumerate(actor_info_list):
            resource_map = {'memory':  Module.get_memory_info(pid=actor_info['pid'])}
            resource_list = actor_info_list[i].pop('resource_mapping', [])

            for resource in resource_list:
                resource_map[resource['name'].lower()] = resource['resource_ids']
            actor_info_list[i]['resources'] = resource_map

            try:
                ray.get_actor(actor_info['name'])
                final_info_list.append(actor_info_list[i])
            except ValueError as e:
                pass

        return final_info_list

    @staticmethod
    def actor_map(*args, **kwargs):
        actor_list = Module.list_actors(*args, **kwargs)
        actor_map  = {}
        for actor in actor_list:
            actor_name = actor.pop('name')
            actor_map[actor_name] = actor
        return actor_map

    @staticmethod   
    def list_actor_names():
        return list(Module.actor_map().keys())

    @staticmethod
    def list_tasks(running=False, name=None, *args, **kwargs):
        filters = []
        if running == True:
            filters.append([("scheduling_state", "=", "RUNNING")])
        if isinstance(name, str):
            filters.append([("name", "=", name)])
        
        if len(filters)>0:
            kwargs['filters'] = filters

        return ray.experimental.state.api.list_tasks(*args, **kwargs)


    @staticmethod
    def list_nodes( *args, **kwargs):
        return list_nodes(*args, **kwargs)


    ############ TESTING LAND ##############

    ############################################

    @classmethod
    def test(cls):
        import streamlit as st
        for attr in dir(cls):
            if attr[:len('test_')] == 'test_':
                getattr(cls, attr)()
                st.write('PASSED',attr)
