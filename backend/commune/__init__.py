from commune.utils import get_object, dict_any, dict_put, dict_get, dict_has, dict_pop, deep2flat
from commune.config.loader import ConfigLoader
from commune.ray.actor import ActorModule
import streamlit as st
import os


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



class BaseModule(ActorModule):
    client = None
    default_config_path = None
 
    def __init__(self, config=None, override={}, **kwargs):

        

        ActorModule.__init__(self,config=config, override=override)

        if kwargs.get('get_clients', True) != False:
            self.client = self.get_clients() 
        if kwargs.get('get_submodules', True) != False:       
            self.get_submodules()


        


    def get_clients(self, clients=None):
        if clients == None:
            clients = self.config.get('client', self.config.get('clients'))
        
        
        if isinstance(clients, type(None)):
            return

        client_module_class = self.get_object('client.module.ClientModule')
        # if isinstance(self, client_module_class):
        #     return
        
        config = client_module_class.default_config()
        config['clients'] = clients




        if isinstance(config, dict) :
            return client_module_class(config=config)
        elif isinstance(config, client_module_class):
            return config 
        else:
            raise NotImplementedError
            
    def get_config(self, config=None):
        if getattr(self, 'config') != None:
            assert isinstance(self,dict)
        if config == None:

            assert self.default_config_path != None
            config = self.config_loader.load(path=self.default_config_path)
        return config
    

    def get_submodules(self, submodule_configs=None):
        '''
        input: dictionary of modular configs
        '''
        if submodule_configs == None:
            submodule_configs = self.config.get('submodule',self.config.get('submodules',{}))
    
        assert isinstance(submodule_configs, dict)
        for submodule_name, submodule_config in submodule_configs.items():
            submodule_class = self.get_object(submodule_config['module'])
            submodule_instance = submodule_class(config=submodule_config)
            setattr(self, submodule_name, submodule_instance)



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

    enable_cache = cache_enable = cache_wrap = enable_cache

    @property
    def cache_path(self):
        return os.path.join(self.tmp_dir, 'cache.json')

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
