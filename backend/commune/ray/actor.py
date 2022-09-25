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


    @staticmethod
    def get_current_timestamp():
        return  datetime.datetime.utcnow().timestamp()
        
    def resolve_config(self, config, override={}, local_var_dict={}, recursive=True):
        if config == None:
            config = getattr(self,'config',  None)
        elif (type(config) in  [list, dict]): 
            if len(config) == 0:
                assert isinstance(self.default_config_path, str)
                config = self.default_config_path
        else:
            raise NotImplementedError(config)

        if config == None:
            assert isinstance(self.default_config_path, str)
            config = self.default_config_path

        if override == None:
            override = {}

        config = self.load_config(config=config, 
                             override=override, 
                            local_var_dict=local_var_dict,
                            recursive=True)


        

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

    @staticmethod
    def get_module(config, actor=False, override={}):
        """
        config: path to config or actual config
        client: client dictionary to avoid child processes from creating new clients
        """
        module_class = None
        # if this is a class return the class
        if is_class(config):
            module_class = config
            return module_class


        if isinstance(config, str):
            # check if object is a path to module, return None if it does not exist
            module_class = ActorModule.get_object(key=config)


        if isinstance(module_class, type):
            
            config = module_class.default_cfg()
       
        else:

            config = ActorModule.load_config(config)
            ActorModule.check_config(config)
            module_class = ActorModule.get_object(config['module'])

        return module_class.deploy(config=config, override=override, actor=actor)

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

    @classmethod 
    def deploy(cls, config=None, actor=False , override={}, local_var_dict={}, **kwargs):
        """
        deploys process as an actor or as a class given the config (config)
        """


        config = ActorModule.resolve_config(cls, config=config, local_var_dict=local_var_dict, override=override)
        
        ray_bool = kwargs.get('ray')

        if ray_bool == False:
            assert actor==False, f'actor should be disabled'
        else:
            ray_context =  cls.get_ray_context(init_kwargs=config.get('ray', {}))


        if actor:
            config['actor'] = config.get('actor', {})
            if isinstance(actor, dict):
                config['actor'].update(actor)
            elif isinstance(actor, bool):
                pass
            else:
                raise Exception('Only pass in dict (actor args), or bool (uses config["actor"] as kwargs)')  
            
            return cls.deploy_actor(cls_kwargs=dict(config=config), **config['actor'])
        else:
            return cls(config=config)

    @staticmethod
    def get_ray_context(init_kwargs, reinit=True):
        default_ray_env = {'address': 'auto', 'namespace': 'default'}

        if init_kwargs == None:
            init_kwargs = default_ray_env

            
        
        if isinstance(init_kwargs, dict):

            
            for k in ['address', 'namespace']:
                default_value= default_ray_env.get(k)
                init_kwargs[k] = init_kwargs.get(k,default_value)
                assert isinstance(init_kwargs[k], str), f'{k} is not in args'
            
            if ActorModule.ray_initialized() and reinit == True:
                ray.shutdown()

            init_kwargs['include_dashboard'] = True
            init_kwargs['dashboard_host'] = '0.0.0.0'
            print(init_kwargs)
            return ray.init(**init_kwargs)
        else:
            raise NotImplementedError(f'{init_kwargs} is not supported')
    
    @classmethod
    def deploy_actor(cls,
                        config=None,
                        cls_kwargs={},
                        name='actor',
                        detached=True,
                        resources={'num_cpus': 1, 'num_gpus': 0.1},
                        max_concurrency=1,
                        refresh=False,
                        verbose = True, 
                        redundant=False):
        if isinstance(config, dict):
            cls_kwargs = {'config': config}
        return create_actor(cls=cls,
                        name=name,
                        cls_kwargs=cls_kwargs,
                        detached=detached,
                        resources=resources,
                        max_concurrency=max_concurrency,
                        refresh=refresh,
                        return_actor_handle=True,
                        verbose=verbose,
                        redundant=redundant)

    def get(self, key):
        return self.getattr(key)

    def getattr(self, key):
        return getattr(self, key)

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
        return ray.get_actor(actor_name)


    @property
    def ray_context(self):
        return ray.runtime_context.get_runtime_context()
    @property
    def context(self):
        if self.actor_exists(self.actor_name):
            return ray.runtime_context.get_runtime_context()

    @property
    def actor_name(self):
        return self.config['actor']['name']
    

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
        path =  cls.get_module_filepath().replace('.py', '.yaml')
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

    @classmethod
    def module_path(cls, include_root=True):
        path =  os.path.dirname(cls.__file__).replace(os.getenv('PWD')+'/', '')
        if include_root == False:
            path = '/'.join(path.split('/')[1:])
        return path

    @staticmethod
    def load_object(module:str, __dict__:dict, **kwargs):
        kwargs = kwargs.get('__dict__', kwargs.get('kwargs', {}))
        return ActorModule.import_object(module)(**kwargs)


    @property
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

    "Base class to expose instance methods"
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
                    print "Found", name, meta
                    methods[name] = member
            return type.__new__(cls, name, bases, state)