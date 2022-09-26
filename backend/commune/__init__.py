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
    client_module_class_path = 'client.manager.module.ClientModule'
 
    def __init__(self, config=None, override={}, client=None ,**kwargs):

        

        ActorModule.__init__(self,config=config, override=override)
        
        # for passing down the client to  submodules to avoid replicating a client for every submodule
        self.client = self.get_clients(client=client, 
                                        get_clients_bool = kwargs.get('get_clients', True)) 
           
        self.get_submodules(get_submodules_bool = kwargs.get('get_submodules', True))


    @property
    def client_config(self):
        return self.config.get('client', self.config.get('clients'))


    def get_clients(self, client=None, get_clients_bool = True):
        if get_clients_bool == False:
            return None

        if client == None:
            client = self.client_config
        if client == None:
            return None

        client_module_class = self.get_object(self.client_module_class_path)
        
        if isinstance(self.client, client_module_class):
            return self.client
        
        config = client_module_class.default_config()
        config['clients'] = client

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
    

    def get_submodules(self, submodule_configs=None, get_submodules_bool=True):
        
        if get_submodules_bool == False:
            return None
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


    def resolve_path(self, path, root_dir=None, extension = '.json'):
        if root_dir == None:
            root_dir = self.tmp_dir
        # resolving base name
        if path == None:
            path = root_dir

        if self.client.local.isdir(os.path.join(root_dir,path)):
            return os.path.join(root_dir,path)
        elif self.client.local.isdir(path):
            return path

        # 
        path_basename, path_ext = os.path.splitext(os.path.basename(path))
        if path_ext != '.json':
            path_ext = extension
        path_basename = path_basename + path_ext
        path_dir = os.path.dirname(path)

        # ensure the path has the module cache root
        if self.tmp_dir!=path_dir[:len(self.tmp_dir)]:
            path_dir = os.path.join(root_dir, path_dir)
        if self.client.local.isdir(path_dir):
            self.client.local.makedirs(path_dir, True)
        path = os.path.join(path_dir, path_basename )
        if os.path.basename(path) == extension:
            path = os.path.dirname(path)
        return path


    def get_json(self,path, root_dir=None, **kwargs):

        path = self.resolve_path(path=path, root_dir=root_dir)
        import streamlit as st
        data = self.client.local.get_json(path=path, **kwargs)
        return data

    def put_json(self, path, data, root_dir=None, **kwargs):
        path = self.resolve_path(path=path, root_dir=root_dir)
        self.client.local.put_json(path=path, data=data, **kwargs)
        return path


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


    def ls_json(self, path=None):
        ls_path = self.tmp_dir 
        if path != None:
            ls_path = os.path.join(ls_path, path)
        return self.client.local.ls(ls_path)

    def rm_json(self, path=None, recursive=True, **kwargs):

        path = self.resolve_path(path)
        return self.client.local.rm(path,recursive=recursive, **kwargs)

    def glob_json(self, pattern ='**'):
        paths =  self.client.local.glob(self.tmp_dir+'/'+pattern)
        return list(filter(lambda f:self.client.local.isfile(f), paths))
    
    def refresh_json(self):
        self.rm_json()

    def put_config(self, path=None):
        if path ==  None:
            path = 'config'
        return self.put_json(path, self.config)

    def get_config(self,  path=None):
        if path ==  None:
            path = 'config'
        self.config = self.get_json(path)



    def put_state_dict(self, path=None, exclude=None, include=None):
        if path == None:
            path = 'state_dict'

        state_dict = self.__dict__
        return self.put_json(path, state_dict)


    def get_state_dict(self, path=None):
        if path == None:
            path = 'state_dict'

        state_dict =  self.get_json(path)
        self.__dict__ =  state_dict

