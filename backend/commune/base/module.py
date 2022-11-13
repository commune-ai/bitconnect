from commune.utils import get_object, dict_any, dict_put, dict_get, dict_has, dict_pop, deep2flat, Timer, dict_override, get_functions, get_function_schema, kill_pid
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
import asyncio
from ray.experimental.state.api import list_actors, list_objects, list_tasks

def cache(self, **kwargs):

    if 'keys' in kwargs:
        key_dict = {k: kwargs.get('keys') for k in ['save', 'load']}
    else:
        key_dict = dict(save=dict_any(x=input_kwargs, keys=['save', 'write'], default=[]),
                        load= dict_any(x=input_kwargs, keys=['load', 'read'], default=[]))

    def wrap_fn(fn):
        def new_fn(self, *args, **kwargs):
            [setattr(self, k, self.get_json(k)) for k in key_dict['load']]
            fn(self, *args, **kwargs)
            [self.put_json(k, getattr(self, k)) for k in key_dict['save']]
        return new_fn
    
    return wrap_fn
def enable_cache(**input_kwargs):
    load_kwargs = dict_any(x=input_kwargs, keys=['load', 'read'], default={})
    if isinstance(load_kwargs, bool):
        load_kwargs = dict(enable=load_kwargs)

    save_kwargs = dict_any(x=input_kwargs, keys=['save', 'write'], default={})
    if isinstance(save_kwargs, bool):
        save_kwargs = dict(enable=save_kwargs)

    refresh = dict_any(x=input_kwargs, keys=['refresh', 'refresh_cache'], default=False)
    assert isinstance(refresh, bool), f'{type(refresh)}'

    def wrapper_fn(fn):
        def new_fn(self, *args, **kwargs):

            if refresh: 
                self.cache = {}
            else:
                self.load_cache(**load_kwargs)

            output = fn(self, *args, **kwargs)
            self.save_cache(**save_kwargs)
            return output
        
        return new_fn
    return wrapper_fn



import streamlit as st
import nest_asyncio
class Module:
    client = None
    client_module_class_path = 'client.manager.module.ClientModule'
    # assumes Module is .../{src}/base/module.py
    root_dir = __file__[len(os.getenv('PWD'))+1:].split('/')[0]
    root_path = os.path.join(os.getenv('PWD'), root_dir)
    root = root_path
    default_ray_env = {'address': 'auto', 'namespace': 'default'}
    ray_context = None
    config_loader = ConfigLoader(load_config=False)

    def __init__(self, config=None, override={}, client=None , loop=None, init_ray=True, **kwargs):
        

        # nest_asyncio.apply()

        self.config = self.resolve_config(config)
        self.override_config(override=override)
        self.start_timestamp =self.current_timestamp
        # for passing down the client to  submodules to avoid replicating a client for every submodule

        if self.class_name != 'client.manager':
            self.client = self.get_clients(client=client) 
        self.get_submodules(get_submodules_bool = kwargs.get('get_submodules', True))

        # self.
        self.cache = {}
        # set asyncio loop
        # self.loop = self.new_event_loop()
        # self.set_event_loop(loop=self.get_event_loop())
        # self.loop = self.new_event_loop()
    @property
    def registered_clients(self):
        return self.clients.registered_clients if self.clients else []
    @property
    def client_config(self):
        for k in ['client', 'clients']:
            client_config =  self.config.get(k, None)
            if client_config != None:
                return client_config
        return client_config

    running_actors_dir = '/tmp/commune/running_actors'
    def register_actor(self):
        if self.actor_running:
            self.client.local.put_json(path=f'{running_actors_dir}/{self.actor_name}/config.json', data=data, **kwargs)

    def deregister_actor(self):
        if self.actor_running:

            self.client.local.rm(path=f'{running_actors_dir}/{self.actor_name}/config.json', recursive=True)

    def get_clients(self, client=None):

        if client == False:
            return None
        elif client == True:
            pass
        elif client == None:
            if self.client_config == None and client == None:
                return None
            client_module_class = self.get_object(self.client_module_class_path)
            client_config = client_module_class.default_config()
            # does the config have clients
            if isinstance(self.client_config, list):
                client_config['include'] = self.client_config
            elif isinstance(self.client_config, dict):
                client_config  = self.client_config
            elif self.client_config == None:
                return 
        elif isinstance(client, client_module_class):
            return client
        elif isinstance(client, dict):
            client_config = client
        elif isinstance(client, list):
            # this is a list of clients
            assert all([isinstance(c, str)for c in client]), f'all of the clients must be string if you are passing a list'
            client_config['include'] = client
        else:
            raise NotImplementedError
        return client_module_class(config=client_config)
            
    # def get_config(self, config=None):
    #     if getattr(self, 'config') != None:
    #         assert isinstance(self,dict)
    #     if config == None:

    #         assert self.default_config_path != None
    #         config = self.config_loader.load(path=self.default_config_path)
    #     return config
    

    def get_submodules(self, submodule_configs=None, get_submodules_bool=True):
        
        if get_submodules_bool == False:
            return None
        '''
        input: dictionary of modular configs
        '''
        if submodule_configs == None:
            submodule_configs = self.config.get('submodule',self.config.get('submodules',{}))
    
        assert isinstance(submodule_configs, dict)
        for submodule_name, submodule in submodule_configs.items():
            submodule_kwargs, submodule_args = {},[]
            if isinstance(submodule, str):
                submodule_kwargs = {'module':submodule }
            elif isinstance(submodule, list):
                submodule_args = submodule
            elif isinstance(submodule, dict):
                submodule_kwargs = submodule
                
            submodule = self.get_module(*submodule_args,**submodule_kwargs)
            dict_put(self.__dict__, submodule_name, submodule)

    ############ LOCAL CACHE LAND ##############

    ############################################

    cache = {}

    @enable_cache()
    def put_cache(self, k, v, **kwargs):
        dict_put(self.cache, k, v)
    

    @enable_cache()
    def get_cache(self, k, default=None, **kwargs):
        return dict_get(self.cache, k,default)

    @enable_cache(save= {'disable':True})
    def in_cache(self, k):
        return dict_has(self,cache, k)
    has_cache = in_cache
    @enable_cache()
    def pop_cache(self, k):
        return dict_pop(self.cache, k)


    del_cache = delete_cache = pop_cache 

    has_cache = cache_has = cache_exists = exists_cache =in_cache

    last_saved_timestamp=0
    @property
    def state_staleness(self):
        return self.current_timestamp - self.last_saved_timestamp

    def resolve_args_kwargs(x):
        if isinstsance(x, dict):
            return [], x
        elif type(x) in [list,tuple,set]:
            return x , {}
        else:
            raise NotImplementedError(type(x))

    @staticmethod
    def enable_cache(**input_kwargs):
        return enable_cache(**input_kwargs)

    @classmethod
    def cache(cls,keys=None,**kwargs):
        return cache(keys=keys, **kwargs)
    enable_cache = cache_enable = cache_wrap = enable_cache

    @property
    def cache_path(self):
        return os.path.join(self.tmp_dir, 'cache.json')

    def resolve_path(self, path, extension = '.json'):
        path = path.replace('.', '/')
        path = os.path.join(self.tmp_dir,path)
        path_dir = os.path.dirname(path)
        os.makedirs(path_dir,exist_ok=True)
        if path[-len(extension):] != extension:
            path = path + extension
        return path

    def get_json(self,path, default=None, **kwargs):
        path = self.resolve_path(path=path)
        try:
            data = self.client.local.get_json(path=path, **kwargs)
        except FileNotFoundError as e:
            if isinstance(default, dict):
                data = self.put_json(path, default)
            else:
                raise e

        return data

    def put_json(self, path, data, **kwargs):
        path = self.resolve_path(path=path)
        self.client.local.put_json(path=path, data=data, **kwargs)
        return data

    def ls_json(self, path=None):
        path = self.resolve_path(path=path)
        if not self.client.local.exists(path):
            return []
        return self.client.local.ls(path)
        
    def exists_json(self, path=None):
        path = self.resolve_path(path=path)
        return self.client.local.exists(path)

    def rm_json(self, path=None, recursive=True, **kwargs):
        path = self.resolve_path(path)
        if not self.client.local.exists(path):
            return 
    
        return self.client.local.rm(path,recursive=recursive, **kwargs)

    def glob_json(self, pattern ='**',  tmp_dir=None):
        if tmp_dir == None:
            tmp_dir = self.tmp_dir
        paths =  self.client.local.glob(tmp_dir+'/'+pattern)
        return list(filter(lambda f:self.client.local.isfile(f), paths))
    
    def refresh_json(self):
        self.rm_json()

    def load_cache(self, **kwargs):
        enable_bool =  kwargs.get('enable', True)
        assert isinstance(enable_bool, bool), f'{disable_bool}'
        if not enable_bool:
            return None
        path = kwargs.get('path',  self.cache_path)


        
        self.client.local.makedirs(os.path.dirname(path), True)
        data = self.client.local.get_json(path=path, handle_error=True)
        
        if data == None:
            data  = {}
        self.cache = data


    def save_cache(self, **kwargs):
        enable_bool =  kwargs.get('enable', True)
        assert isinstance(enable_bool, bool), f'{disable_bool}'
        if not enable_bool:
            return None

        path = kwargs.get('path',  self.cache_path)

        staleness_period=kwargs.get('statelness_period', 100)
  
        self.client.local.makedirs(os.path.dirname(path), True)
        data =  self.cache
        self.client.local.put_json(path=path, data=data)

    save_state = save_cache
    load_state = load_cache
    
    @property
    def refresh_cache_bool(self):
        refresh_bool = self.config.get('refresh_cache', False)
        if refresh_bool == False:
            refresh_bool = self.config.get('cache', False)
        
        return refresh_bool

    def init_cache(self):
        if self.refresh_cache_bool:
            self.cache = {}
            self.save_cache()
        self.load_cache()

    def reset_cache(self):
        self.cache = {}
        self.save_cache()


    def put_config(self, path=None):
        if path ==  None:
            path = 'config'
        return self.put_json(path, self.config)

    def rm_config(self, path=None):
        if path ==  None:
            path = 'config'
        return self.rm_json(path)

    refresh_config = rm_config
    def get_config(self,  path=None, handle_error =True):
        if path ==  None:
            path = 'config'
        config = self.get_json(path, handle_error=handle_error)
        if isinstance(config, dict):
            self.config = config

    def put_state_dict(self, path=None, exclude=None, include=None):
        if path == None:
            path = 'state_dict'

        state_dict = self.__dict__
        return self.put_json(path, state_dict)

    @property
    def module2path(self):
        module2path = {}
        for k in self.simple_module_list:
            module2path[k] =  '/'.join([self.root_path, k.replace('.', '/')])

        return module2path
    @property
    def module_fs(self):
        module_fs = {}
        for k in self.simple2module.keys():
            
            module_path = '/'.join([os.getenv('PWD'), 'commune',k.replace('.', '/')])
            file_list = self.client.local.ls(module_path)
            dict_put(module_fs,k, file_list)

        return module_fs

    def get_state_dict(self, path=None):
        if path == None:
            path = 'state_dict'

        state_dict =  self.get_json(path)
        self.__dict__ =  state_dict

    @property
    def simple_module_list(self):
        return self.list_modules()
    module_list = simple_module_list

    def list_modules(self):
        return list(self.simple2module.keys())

    @property
    def simple2module(self):
        return {'.'.join(k.split('.')[:-2]):k for k in self.full_module_list}

    @property
    def module2simple(self):
        return {v:k for k,v in self.simple2module.items()}


    @staticmethod
    def simple2path( simple):
        path = deepcopy(simple.replace('.', '/'))
        if simple[:len(Module.root_dir)] != Module.root_dir:
            path = os.path.join(Module.root, simple, 'module.yaml')
        module_name = Module.load_config(simple).get('module')
        full_path = '.'.join([Module.root_dir, simple,'module', module_name])
        return full_path

    def get_module_class(self,module:str):
        if module[:len(self.root_dir)] != self.root_dir:
            module = '.'.join([self.root_dir, module])

        if module in self.simple2module:
            module_path = self.simple2module[module]

        elif module in self.module2simple:
            module_path = module
        else:
            raise Exception(f'({module}) not in options {list(self.simple2module.keys())} (short) and {list(self.simple2module.values())} (long)')
        
        
        if self.root_dir != module_path[:len(self.root_dir)]:
            module_path = '.'.join([self.root_dir, module_path])
        module_class= self.import_object(module_path)
        return module_class

    @property
    def full_module_list(self):
        modules = []
        failed_modules = []
        for root, dirs, files in os.walk(self.root_path):
            if all([f in files for f in ['module.py', 'module.yaml']]):
                try:
                    cfg = self.config_loader.load(root)   
                    if cfg == None:
                        cfg = {}           
                except Exception as e:
                    cfg = {}


                module_path = root.lstrip(os.environ['PWD']).replace('/', '.')
                module_path = '.'.join(module_path.split('.')[1:])
                module_path = self.root_dir + '.'+ module_path

                if isinstance(cfg.get('module'), str):
                    module_name = cfg.get('module').split('.')[-1]
                    modules.append(f"{module_path}.module.{module_name}")
                elif module_path == None: 
                    raise NotImplemented(root)

        return modules



    def submit_fn(self, fn:str, queues:dict={}, block:bool=True,  *args, **kwargs):

        if queues.get('in'):
            input_item = self.queue.get(topic=queues.get('in'), block=block)
            if isinstance(input_item, dict):
                kwargs = input_item
            elif isinstance(input_item, list):
                args = input_item
            else:
                args = [input_item]
        
        out_item =  getattr(self, fn)(*args, **kwargs)

        if isinstance(queues.get('out'), str):
            self.queue.put(topic=queues.get('out'), item=out_item)
    
    def stop_loop(self, key='default'):
        return self.loop_running_loop.pop(key, None)

    def running_loops(self):
        return list(self.loop_running_map.keys())

    loop_running_map = {}
    def start_loop(self, in_queue=None, out_queue=None, key='default', refresh=False):
        
        in_queue = in_queue if isintance(in_queue,str) else 'in'
        out_queue = out_queue if isintance(out_queue,str) else 'out'
        
        if key in self.loop_running_map:
            if refresh:
                while key in self.loop_running_map:
                    self.stop_loop(key=key)
            else:
                return 
        else:
            self.loop_running_map[key] = True

        while key in self.loop_running_map:
            input_dict = self.queue.get(topic=in_queue, block=True)
            fn = input_dict['fn']
            fn_kwargs = input_dict.get('kwargs', {})
            fn_args = input_dict.get('args', [])
            output_dict  = self.submit_fn(fn=fn, *fn_args, **fn_kwargs)
            self.queue.put(topic=out_queue,item=output_dict)


    module_tree = module_list

    @classmethod
    def launch(cls, module:str, fn:str=None ,kwargs:dict={}, args=[], actor=False, wrap=True, **additional_kwargs):
        try:
            module_class =  cls.load_module(module)
        except Exception as e:
            module_class = cls.import_object(module)

        module_init_fn = fn
        module_kwargs = {**kwargs}
        module_args = [*args]
        
        if module_init_fn == None:
            if actor:
                # ensure actor is a dictionary
                if actor == True:
                    actor = {}
                assert isinstance(actor, dict), f'{type(actor)} should be dictionary fam'
                parents = cls.get_parents(module_class)
                if cls.is_module(module_class):
                    default_actor_name = module_class.get_default_actor_name()
                else:
                    default_actor_name = module_class.__name__
                

                actor['name'] = actor.get('name', default_actor_name )
                module_object = cls.create_actor(cls=module_class, cls_kwargs=module_kwargs, cls_args=module_args, **actor)
                module_object.actor_name = actor['name']
                if wrap == True: 
                    module_object = cls.wrap_actor(module_object)
            else:
                module_object =  module_class(*module_args,**module_kwargs)


        else:
            module_init_fn = getattr(module_class,module_init_fn)
            module_object =  module_init_fn(*module_args, **module_kwargs)


        return module_object
    launch_module = launch
    #############

    # RAY ACTOR TINGS, TEHE
    #############

    @classmethod
    def load_module(cls, path):
        module_path = cls.simple2path(path)
        module_class =  cls.import_object(module_path)
        return module_class


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
    def get_config_path(cls, simple=False):
        module_path = cls.get_module_path(simple=simple)
        if os.getenv('PWD') != module_path[len(os.getenv('PWD'))]:

            config_path = os.path.join(os.getenv('PWD'), cls.get_module_path(simple=simple))

        if simple == False:
            config_path = config_path.replace('.py', '.yaml')
        return config_path

    def resolve_config(self, config, override={}, recursive=True ,return_munch=False, **kwargs):
        


        if config == None:
            config_path =  os.path.join(os.getenv('PWD'), self.get_config_path())
            print(config_path, 'CONFIG')
            assert type(config_path) in [str, dict, Munch], f'CONFIG type {type(config)} no supported'
            config = self.load_config(config=config_path, 
                                override=override, 
                                return_munch=return_munch)
        assert isinstance(config, dict), type(config)
        return config
        # st.write(config)
        



        if config == None:
            config = 'base'
            config = cls.resolve_config(config=config, 
                        override=override, 
                        recursive=recursive,
                        return_munch=return_munch)
            # assert isinstance(config, dict),  f'bruh the config should be {type(config)}'
            # self.save_config(config=config)

        return config

    @classmethod
    def load_config(cls, config=None, override={}, recursive=True, return_munch=False):
        """
        config: 
            Option 1: dictionary config (passes dictionary) 
            Option 2: absolute string path pointing to config
        """
        if config == None:
            config = cls.get_config_path()
        return Module.config_loader.load(path=config, 
                                     override=override,
                                     recursive=recursive,
                                     return_munch=return_munch)

    @classmethod
    def save_config(cls, path=None, config=None):
        """
        config: 
            Option 1: dictionary config (passes dictionary) 
            Option 2: absolute string path pointing to config
        """
        if path == None:
            path = cls.get_config_path()
        return cls.config_loader.save(path=path, cfg=config)
    @classmethod
    def default_cfg(cls, *args,**kwargs):
        return cls.config_loader.load(path=cls.get_config_path(),*args, **kwargs)

    default_config = default_cfg
    config_template = default_cfg
    _config = default_cfg

    @classmethod
    def deploy_module(cls, module:str, **kwargs):
        module_class = cls.get_object(module)
        return module_class.deploy(**kwargs)
    get_module = deploy_module

    @staticmethod
    def check_config(config):
        assert isinstance(config, dict)
        assert 'module' in config
    @staticmethod
    def get_object(path:str, prefix = 'commune'):
        return get_object(path=path, prefix=prefix)

    import_module_class = get_object
    
    @staticmethod
    def import_module(key):
        return import_module(key)

    @staticmethod
    def import_object(key):
        module_path = '.'.join(key.split('.')[:-1])
        module = import_module(module_path)
        object_name = key.split('.')[-1]
        obj = getattr(module, object_name)
        return obj

    @staticmethod
    def ray_initialized():
        return ray.is_initialized()

    @property
    def actor_id(self):
        return self.get_id()

    def get_id(self):
        return dict_get(self.config, 'actor.id')
    def get_name(self):
        return dict_get(self.config, 'actor.name')

    def actor_info(self):
        actor_info_dict = dict_get(self.config, 'actor')
        actor_info_dict['resources'] = self.resource_usage()
        return actor_info_dict



    @staticmethod
    def add_actor_metadata(actor):
        # actor_id = Module.get_actor_id(actor)
        # actor.config_set.remote('actor.id', actor_id)

        # actor_name = ray.get(actor.getattr.remote('actor_name'))
        # setattr(actor, 'actor_id', actor_id)
        # setattr(actor, 'actor_name', actor_name)
        # setattr(actor, 'id', actor_id)
        # setattr(actor, 'name', actor_name)
        return actor


    @classmethod 
    def deploy(cls, actor=False , skip_ray=False, wrap=False,  **kwargs):
        """
        deploys process as an actor or as a class given the config (config)
        """

        config = kwargs.pop('config', None)
        config = cls.load_config(config=config)

        ray_config = config.get('ray', {})
        if not cls.ray_initialized():
            ray_context =  cls.init_ray(init_kwargs=ray_config)
        
        if actor:
            actor_config =  config.get('actor', {})

            assert isinstance(actor_config, dict), f'actor_config should be dict but is {type(actor_config)}'
            if isinstance(actor, dict):
                actor_config.update(actor)
            elif isinstance(actor, bool):
                pass
            else:
                raise Exception('Only pass in dict (actor args), or bool (uses config["actor"] as kwargs)')  

            try:

                actor_config['name'] =  actor_config.get('name', cls.get_default_actor_name())                
                config['actor'] = actor_config
                kwargs['config'] = config
                actor = cls.create_actor(cls=cls,  cls_kwargs=kwargs, **actor_config)
                
                actor_id = cls.get_actor_id(actor)  
                actor =  cls.add_actor_metadata(actor)
            except ray.exceptions.RayActorError:
                actor_config['refresh'] = True
                config['actor'] = actor_config
                kwargs['config'] = config
                actor = cls.create_actor(cls=cls, cls_kwargs=kwargs, **actor_config)
                actor_id = cls.get_actor_id(actor)  
                actor =  cls.add_actor_metadata(actor)

            if wrap:
                actor = cls.wrap_actor(actor)

            return actor 
        else:
            kwargs['config'] = config
            kwargs['config']['actor'] = None
            return cls(**kwargs)

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
                 **kwargs):


        if cpus > 0:
            resources['num_cpus'] = cpus
        if gpus > 0:
            resources['num_gpus'] = gpus

        if not torch.cuda.is_available() and 'num_gpus' in resources:
            del resources['num_gpus']

        # configure the option_kwargs

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


        return Module.get_actor(name)





    def get(self, key):
        return self.getattr(key)

    def getattr(self, key):
        return getattr(self, key)

    def hasattr(self, key):
        return hasattr(self, key)

    def setattr(self, key, value):
        return self.__setattr__(key,value)

    def deleteattr(self, key):
        del self.__dict__[key]
        return key
    
    rmattr = rm = delete = deleteattr

    def down(self):
        self.kill_actor(self.config['actor']['name'])

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
            return actor in Module.list_actor_names()
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
    @property
    def module(self):
        return self.config['module']

    @property
    def name(self):
        return self.config.get('name', self.module)
    
    def class_name(self):
        return self.__class__.__name__

    def mapattr(self, from_to_attr_dict={}):
        '''
        from_to_attr_dict: dict(from_key:str->to_key:str)
        '''
        for from_key, to_key in from_to_attr_dict.items():
            self.copyattr(from_key=from_key, to_key=to_key)

    def copyattr(self, from_key, to_key):
        '''
        copy from and to a desintatio
        '''
        attr_obj = getattr(self, from_key)  if hasattr(self, from_key) else None
        setattr(self, to, attr_obj)

    def dict_keys(self):
        return self.__dict__.keys()

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
    def get_function_schema( fn, *args, **kwargs):
        return get_function_schema(fn=fn, *args, **kwargs)

    @classmethod
    def get_function_schemas(cls, obj=None, *args,**kwargs):
        if obj == None:
            obj = cls
        
        fn_map = {}
        for fn_key in obj.get_functions(obj):
            # st.write(fn)
            fn = getattr(obj, fn_key)
            if not callable(fn) or isinstance(fn, type) or isinstance(fn, types.BuiltinFunctionType):
                continue
            fn_map[fn_key] = cls.get_function_schema(fn=fn, *args, **kwargs)
        return fn_map
    

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
    def describe(cls, obj=None, streamlit=False, sidebar=True,**kwargs):
        if obj == None:
            obj = cls

        assert is_class(obj)

        fn_list = cls.functions(return_type='fn', obj=obj, **kwargs)
        
        fn_dict =  {f.__name__:f for f in fn_list}
        if streamlit:
            import streamlit as st
            for k,v in fn_dict.items():
                with (st.sidebar if sidebar else st).expander(k):
                    st.write(k,v)
        else:
            return fn_dict
        
        

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
    def get_module_filepath(cls, obj=None, include_pwd=True):
        if obj == None:
            obj = cls
        filepath = inspect.getfile(obj)
        if not include_pwd:
            filepath = cls.get_module_filepath().replace(os.getenv('PWD')+'/', '')
        return filepath


    @classmethod
    def run_streamlit(cls, port=8501):
        filepath = cls.get_module_filepath(include_pwd=False)
        cls.run_command(f'streamlit run {filepath} --server.port={port} -- -fn=streamlit')


    @classmethod
    def gradio(cls):
        functions, names = [], []

        fn_map = {
            'fn1': lambda x :  int(x),
            'fn2': lambda x :  int(x) 
        }

        for fn_name, fn in fn_map.items():
            inputs = [gr.Textbox(label='input', lines=3, placeholder=f"Enter here...")]
            outputs = [gr.Number(label='output', precision=None)]
            names.append(fn_name)
            functions.append(gr.Interface(fn=fn, inputs=inputs, outputs=outputs))
        
        return gr.TabbedInterface(functions, names)


    @classmethod
    def run_gradio(cls, port=8501, host='0.0.0.0'):
        filepath = cls.get_module_filepath(include_pwd=False)
        interface = cls.gradio()
 
        interface.launch(server_port=port,
                        server_name=host,
                        inline= False,
                        share= None,
                        debug=False,
                        enable_queue= None,
                        max_threads=10,
                        auth= None,
                        auth_message= None,
                        prevent_thread_lock= False,
                        show_error= True,
                        show_tips= False,
                        height= 500,
                        width= 900,
                        encrypt= False,
                        favicon_path= None,
                        ssl_keyfile= None,
                        ssl_certfile= None,
                        ssl_keyfile_password= None,
                        quiet= False)
        





    @classmethod
    def run_python(cls):
        cls.run_command(f'python {filepath}')

    @classmethod
    def argparse(cls):
        parser = argparse.ArgumentParser(description='Gradio API and Functions')
        parser.add_argument('-fn', '--function', dest='function', help='run a function from the module', type=str, default="streamlit")
        parser.add_argument('-kwargs', '--kwargs', dest='kwargs', help='arguments to the function', type=str, default="{}")  
        parser.add_argument('-args', '--args', dest='args', help='arguments to the function', type=str, default="[]")  

        return parser.parse_args()




    @classmethod
    def parents(cls):
        return get_parents(cls)




    @staticmethod
    def timeit(fn, trials=1, time_type = 'seconds', timer_kwargs={} ,*args,**kwargs):
        
        elapsed_times = []
        results = []
        
        for i in range(trials):
            with Timer(**timer_kwargs) as t:
                result = fn(*args, **kwargs)
                results.append(result)
                elapsed_times.append(t.elapsed_time)
        return dict(mean=np.mean(elapsed_times), std=np.std(elapsed_times), trials=trials, results=[])

    time = timeit
    # timer
    timer = Timer

    @classmethod
    def describe_module_schema(cls, obj=None, **kwargs):
        if obj == None:
            obj = cls
        return get_module_function_schema(obj, **kwargs)

    def config_set(self,k,v, **kwargs):
        return dict_put(self.config, k,v)

    def config_get(self,k, ):
        return dict_get(self.config, k,v)


    def override_config(self,override:dict={}):
        self.dict_override(input_dict=self.config, override=override)
    
    @staticmethod
    def dict_override(*args, **kwargs):
        return dict_override(*args,**kwargs)
    
    @staticmethod
    def import_object(path):
        module = '.'.join(path.split('.')[:-1])
        object_name = path.split('.')[-1]
        return getattr(import_module(module), object_name)


    @property
    def module_path(self):
        return self.get_module_path()


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
    def run_command(command:str):

        process = subprocess.run(shlex.split(command), 
                            stdout=subprocess.PIPE, 
                            universal_newlines=True)
        
        return process


    def resolve(self, key=None, value=None):
        if value == None:
            return getattr(self, key)
        else:
            return getattr(self, key)

    @classmethod
    def create_pool(cls, replicas=3, actor_kwargs_list=[], **kwargs):
        if actor_list == None:
            actor_kwargs_list = [kwargs]*replicas

        actors = []
        for actor_kwargs in actor_kwargs_list:
            actors.append(cls.deploy(**a_kwargs))

        return ActorPool(actors=actors)

    @staticmethod
    def check_pid(pid):        
        return check_pid(pid)

    @staticmethod
    def kill_pid(pid):        
        return kill_pid(pid)

    @property
    def tmp_dir(self):
        return f'/tmp/{self.root_dir}/{self.name}'

    @staticmethod
    def get_actor_id( actor):
        assert isinstance(actor, ray.actor.ActorHandle)
        return actor.__dict__['_ray_actor_id'].hex()

    def resource_usage(self):
        resource_dict =  self.config.get('actor', {}).get('resources', None)
        resource_dict = {k.replace('num_', ''):v for k,v in resource_dict.items()}
        resource_dict['memory'] = self.memory_usage(mode='ratio')
        return  resource_dict

    @classmethod
    def wrap_actor(cls, actor):
        wrapper_module_path = 'ray.client.module.ClientModule'
        return Module.get_module(module=wrapper_module_path, server=actor)

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


    @staticmethod
    def list_objects( *args, **kwargs):
        return ray.experimental.state.api.list_objects(*args, **kwargs)

    @staticmethod
    def list_actors(state='ALIVE', detail=True, *args, **kwargs):
        kwargs['filters'] = kwargs.get('filters', [("state", "=", state)])
        kwargs['detail'] = detail

        actor_info_list =  list_actors(*args, **kwargs)
        for i, actor_info in enumerate(actor_info_list):
            resource_map = {'memory':  Module.get_memory_info(pid=actor_info['pid'])}
            resource_list = actor_info_list[i].pop('resource_mapping', [])
            for resource in resource_list:
                resource_map[resource['name'].lower()] = resource['resource_ids']

            actor_info_list[i]['resources'] = resource_map

        return actor_info_list

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




    @classmethod
    def get_module_path(cls, obj=None,  simple=True):
        if obj == None:
            obj = cls
        module_path =  inspect.getmodule(obj).__file__
        # raise(obj)
        if simple:
            module_path = os.path.dirname(module_path.replace(Module.root, '')).replace('/', '.')[1:]

        return module_path

    @staticmethod
    def list_nodes( *args, **kwargs):
        return list_nodes(*args, **kwargs)

    ##############
    #   ASYNCIO
    ##############
    @staticmethod
    def new_event_loop(set_loop=True):
        loop = asyncio.new_event_loop()
        if set_loop:
            asyncio.set_event_loop(loop)
        return loop
    new_loop = new_event_loop 
    # @property
    # def loop(self):
    #     return getattr(self, '_loop',asyncio.get_event_loop())
    # @loop.setter
    # def loop(self, loop):
        # if loop == None:
        #     loop = asyncio.get_event_loop()
        # self._loop = loop
        # return loop
    def set_event_loop(self, loop=None, new=False):
        if loop == None:
            loop = self.new_event_loop()
        return loop
    set_loop = set_event_loop
    def get_event_loop(self):
        return asyncio.get_event_loop()     
    def async_run(self, job, loop=None): 
        if loop == None:
            loop = self.loop
        return loop.run_until_complete(job)


    # async def async_default(self):
    #     pass
    @staticmethod
    def port_connected( port : int,host:str='0.0.0.0'):
        """
            Check if the given param port is already running
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)       
        result = s.connect_ex((host, int(port)))
        return result == 0


    #### RAY HELPERS

    @staticmethod
    def ray_get(self, *jobs):
        return ray.get(jobs)

    @staticmethod
    def ray_wait( *jobs):
        finished_jobs, running_jobs = ray.wait(jobs)
        return finished_jobs, running_jobs


        
    @staticmethod
    def ray_put(*items):
        return [ray.put(i) for i in items]

    @classmethod
    def streamlit(cls):
        st.write(f'HELLO from {cls.__name__}')


    @classmethod
    def run(cls): 
        input_args = cls.argparse()
        assert hasattr(cls, input_args.function)
        kwargs = json.loads(input_args.kwargs)
        assert isinstance(kwargs, dict)

        args = json.loads(input_args.args)
        assert isinstance(args, list)

        getattr(cls, input_args.function)(*args, **kwargs)
    


if __name__ == '__main__':
    Module.run()
    st.write(Module.get_module_filepath().replace(os.getenv('PWD')+'/', ''))
