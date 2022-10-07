import ray
from commune.config import ConfigLoader
from commune.ray.utils import create_actor, actor_exists, kill_actor, custom_getattr, RayEnv
from commune.utils import dict_put, get_object, dict_get, get_module_file, get_function_defaults, get_function_schema, is_class, Timer, get_functions, check_pid, kill_pid, dict_override, dict_merge
import subprocess 
import shlex
import os
import numpy as np
import datetime
import inspect
from types import ModuleType
from importlib import import_module
from .actor_pool import ActorPool
from munch import Munch
class ActorModule: 
    default_ray_env = {'address': 'auto', 'namespace': 'default'}
    ray_context = None
    config_loader = ConfigLoader(load_config=False)
    default_config_path = None
    def __init__(self, config=None, override={}, **kwargs):
        print(config, 'LOADED FAM') 
        

        self.config = self.resolve_config(config=config)
        self.override_config(override=override)
        self.start_timestamp =self.current_timestamp
        self.cache = {}
        
    @property
    def current_timestamp(self):
        return self.get_current_timestamp()

    def current_datetime(self):
        datetime.datetime.fromtimestamp(self.current_timestamp)
    

    def start_datetime(self):
        datetime.datetime.fromtimestamp(self.start_timestamp)
    


    @staticmethod
    def get_current_timestamp():
        return  datetime.datetime.utcnow().timestamp()
        
    def resolve_config(self, config, override={}, local_var_dict={}, recursive=True, return_munch=False, **kwargs):
        if config == None:
            config = getattr(self,'config',  self.default_config_path)
        elif (type(config) in  [list, dict]): 
            if len(config) == 0:
                assert isinstance(self.default_config_path, str)
                config = self.default_config_path
        elif isinstance(config, str):
            config = config
        else:
            raise NotImplementedError(config)

        config = self.load_config(config=config, 
                             override=override, 
                            local_var_dict=local_var_dict,
                            recursive=recursive)
        
        if return_munch:
            config = Munch(config)

        return config

    @staticmethod
    def load_config(config=None, override={}, local_var_dict={}, recursive=True):
        """
        config: 
            Option 1: dictionary config (passes dictionary) 
            Option 2: absolute string path pointing to config
        """
        return ActorModule.config_loader.load(path=config, 
                                    local_var_dict=local_var_dict, 
                                     override=override,
                                     recursive=True)


    @classmethod
    def default_cfg(cls, override={}, local_var_dict={}):

        return cls.config_loader.load(path=cls.default_config_path, 
                                    local_var_dict=local_var_dict, 
                                     override=override)

    default_config = default_cfg
    config_template = default_cfg
    _config = default_cfg

    @classmethod
    def get_module(cls, module:str, **kwargs):

        module_class = cls.get_object(module)
        return module_class.deploy(**kwargs)

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
        return dict_get(self.config, 'actor')


    @staticmethod
    def add_actor_metadata(actor):
        actor_id = ActorModule.get_actor_id(actor)
        actor.config_set.remote('actor.id', actor_id)

        actor_name = ray.get(actor.getattr.remote('actor_name'))
        setattr(actor, 'actor_id', actor_id)
        setattr(actor, 'actor_name', actor_name)
        setattr(actor, 'id', actor_id)
        setattr(actor, 'name', actor_name)
        return actor

    @classmethod 
    def deploy_actor(cls,name=None, refresh=False,**kwargs):
        
        actor_kwargs = dict(
            refresh=refresh,
            name=name,
            **kwargs
        )
        return cls.deploy( actor=actor,**kwargs)

    @classmethod 
    def deploy(cls, actor=False , skip_ray=False, wrap=False,  **kwargs):
        """
        deploys process as an actor or as a class given the config (config)
        """
        config = kwargs.pop('config', None)
        config = ActorModule.resolve_config(self=cls,config=config, **kwargs)
        if cls.ray_initialized():
            skip_ray = True

        if skip_ray == False:
            ray_config = config.get('ray', {})
            ray_context =  cls.get_ray_context(init_kwargs=ray_config)
        import streamlit as st
        # st.write(ray_context, ray_config)
        if actor:
            actor_config =  config.get('actor', {})
            if isinstance(actor, dict):
                actor_config.update(actor)
            elif isinstance(actor, bool):
                pass
            else:
                raise Exception('Only pass in dict (actor args), or bool (uses config["actor"] as kwargs)')  
            config['actor'] = actor_config
            kwargs['config'] = config
            # import streamlit as st
            # st.write(actor_config, kwargs)
            actor = cls.deploy_actor(**actor_config, **kwargs)

            actor_id = cls.get_actor_id(actor)  

            actor =  cls.add_actor_metadata(actor)
            if wrap:
                actor = cls.wrap_actor(actor)

            return actor 
        else:
            
            kwargs['config'] = config
            return cls(**kwargs)

    default_ray_env = {'address': 'auto', 'namespace': 'default'}
    @classmethod
    def get_ray_context(cls,init_kwargs=None, reinit=True):
        
        if cls.ray_initialized():
            return


        if init_kwargs == None:
            init_kwargs = cls.default_ray_env

            
        if isinstance(init_kwargs, dict):

            
            for k in ['address', 'namespace']:
                default_value= cls.default_ray_env.get(k)
                init_kwargs[k] = init_kwargs.get(k,default_value)
                assert isinstance(init_kwargs[k], str), f'{k} is not in args'
            
            if ActorModule.ray_initialized() and reinit == True:
                ray.shutdown()
            init_kwargs['include_dashboard'] = True
            init_kwargs['dashboard_host'] = '172.28.0.2'
            return ray.init(ignore_reinit_error=True, **init_kwargs)
        else:
            raise NotImplementedError(f'{init_kwargs} is not supported')
    


    @classmethod
    def deploy_actor(cls,
                        name='actor',
                        detached=True,
                        resources={'num_cpus': 0.5, 'num_gpus': 0.0},
                        max_concurrency=100,
                        refresh=False,
                        verbose = True, 
                        redundant=False, 
                        return_actor_handle=True,
                        **kwargs):
 

        return create_actor(cls=cls,
                        name=name,
                        cls_kwargs=kwargs,
                        detached=detached,
                        resources=resources,
                        max_concurrency=max_concurrency,
                        refresh=refresh,
                        return_actor_handle=return_actor_handle,
                        verbose=verbose,
                        redundant=redundant)

    def get(self, key):
        return self.getattr(key)

    def getattr(self, key):
        return getattr(self, key)

    def hasattr(self, key):
        return hasattr(self, key)

    def setattr(self, key, value):
        return self.__setattr__(key,value)

    def down(self):
        self.kill_actor(self.config['actor']['name'])

    @staticmethod
    def kill_actor(actor):
        kill_actor(actor)
        return f'{actor} killed'
    
    @staticmethod
    def actor_exists(actor):
        return actor_exists(actor)

    @staticmethod
    def get_actor(actor_name):
        actor =  ray.get_actor(actor_name)
        actor = ActorModule.add_actor_metadata(actor)
        return actor

    @property
    def ray_context(self):
        return ray.runtime_context.get_runtime_context()
    @property
    def context(self):
        if self.actor_exists(self.actor_name):
            return ray.runtime_context.get_runtime_context()

    @property
    def actor_name(self):
        return self.config.get('actor', {}).get('name')
    
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
    def get_module_filepath(cls):
        return inspect.getfile(cls)

    @classmethod
    def get_config_path(cls):
        path =  ActorModule.get_module_filepath().replace('.py', '.yaml')
        assert os.path.exists(path), f'{path} does not exist'
        assert os.path.isfile(path), f'{path} is not a dictionary'
        return path

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
    def get_module_path(cls):
        return cls.default_config_path.replace('.module', '')


    @staticmethod
    def load_object(module:str, __dict__:dict, **kwargs):
        kwargs = kwargs.get('__dict__', kwargs.get('kwargs', {}))
        return ActorModule.import_object(module)(**kwargs)



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
    def __file__(self):
        return self.get_module_filepath()

    @property
    def tmp_dir(self):
        return f'/tmp/commune/{self.name}'

    _exposable_ = None  # Not necessary, just for pylint
    class __metaclass__(type):
        def __new__(cls, name, bases, state):
            methods = state['_exposed_'] = dict()

            # inherit bases exposed methods
            for base in bases:
                methods.update(getattr(base, '_exposed_', {}))

            for name, member in state.items():
                meta = getattr(member, '__meta__', None)
                if meta is not None:
                    print("Found", name, meta)
                    methods[name] = member
            return type.__new__(cls, name, bases, state)

    @staticmethod
    def get_actor_id( actor):
        assert isinstance(actor, ray.actor.ActorHandle)
        return actor.__dict__['_ray_actor_id'].hex()

    def get_resources(self):
        return self.config.get('actor', {}).get('resources', None)

    @classmethod
    def wrap_actor(cls, actor):
        wrapper_module_path = 'ray.client.module.ClientModule'
        return cls.get_module(module=wrapper_module_path, server=actor)