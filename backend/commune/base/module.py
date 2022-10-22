from commune.utils import get_object, dict_any, dict_put, dict_get, dict_has, dict_pop, deep2flat
from commune.config.loader import ConfigLoader
from commune.ray.actor import ActorModule
import streamlit as st
import os
from .utils import enable_cache, cache
from munch import Munch
import inspect


class BaseModule(ActorModule):
    client = None
    default_config_path = 'base'
    client_module_class_path = 'client.manager.module.ClientModule'
    # assumes BaseModule is .../{src}/base/module.py
    root_path = '/'.join(__file__.split('/')[:-2])
    root = root_path
    
    def __init__(self, config=None, override={}, client=None ,**kwargs):

        ActorModule.__init__(self,config=config, override=override, **kwargs)

        # for passing down the client to  submodules to avoid replicating a client for every submodule
        self.client = self.get_clients(client=client) 
           

        # st.write(self.__class__,self.registered_clients,'debug')
        

        self.get_submodules(get_submodules_bool = kwargs.get('get_submodules', True))

        # st.write(self.client, client, kwargs.get('get_clients', True))
        # self.register_actor()

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
        client_module_class = self.get_object(self.client_module_class_path)
        client_config = client_module_class.default_config()
        # st.write(client_config, client)
        if client == False:
            return None
        elif client == True:
            pass
        elif client == None:
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
            
    def get_config(self, config=None):
        if getattr(self, 'config') != None:
            assert isinstance(self,dict)
        if config == None:

            assert self.default_config_path != None
            config = self.config_loader.load(path=self.default_config_path)
        return config
    
    @classmethod
    def get_module(cls, module:str, **kwargs):

        module_class = cls.get_object(module)
        return module_class.deploy(**kwargs)


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


    def resolve_path(self, path, tmp_dir=None, extension = '.json'):
        if tmp_dir == None:
            tmp_dir = self.tmp_dir
        # resolving base name
        if path == None:
            path = tmp_dir

        if self.client.local.isdir(os.path.join(tmp_dir,path)):
            return os.path.join(tmp_dir,path)
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
            path_dir = os.path.join(tmp_dir, path_dir)
        if self.client.local.isdir(path_dir):
            self.client.local.makedirs(path_dir, True)
        path = os.path.join(path_dir, path_basename )
        if os.path.basename(path) == extension:
            path = os.path.dirname(path)
        return path


    def get_json(self,path, tmp_dir=None, **kwargs):

        path = self.resolve_path(path=path, tmp_dir=tmp_dir)
        import streamlit as st
        data = self.client.local.get_json(path=path, **kwargs)
        return data

    def put_json(self, path, data, tmp_dir=None, **kwargs):
        path = self.resolve_path(path=path, tmp_dir=tmp_dir)
        self.client.local.put_json(path=path, data=data, **kwargs)
        return path
    def ls_json(self, path=None, tmp_dir=None):
        if tmp_dir == None:
            tmp_dir = self.tmp_dir
        path = os.path.join(tmp_dir, path)
        if not self.client.local.exists(path):
            return []
        return self.client.local.ls(path)
        
    def exists_json(self, path=None, tmp_dir=None):
        path = self.resolve_path(path=path, tmp_dir=tmp_dir)
        return self.client.local.exists(path)

    def rm_json(self, path=None,tmp_dir=None, recursive=True, **kwargs):
        if tmp_dir == None:
            tmp_dir = self.tmp_dir
        path = os.path.join(tmp_dir, path)
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


    def resolve_module_class(self,module:str):

        if module in self.simple2module:
            module_path = self.simple2module[module]
        elif module in self.module2simple:
            module_path = module
        else:
            raise Exception(f'options are {list(self.simple2module.keys())} (short) and {list(self.simple2module.values())} (long)')
        
        actor_class= self.get_object(module_path)
        return actor_class


    @property
    def full_module_list(self):
        modules = []
        failed_modules = []
        for root, dirs, files in self.client.local.walk('/app/commune'):
            if all([f in files for f in ['module.py', 'module.yaml']]):
                try:
                    cfg = self.config_loader.load(root)   
                    if cfg == None:
                        cfg = {}           
                except Exception as e:
                    cfg = {}


                module_path = root.lstrip(os.environ['PWD']).replace('/', '.')
                module_path = '.'.join(module_path.split('.')[1:])
                if isinstance(cfg.get('module'), str):
                    module_name = cfg.get('module').split('.')[-1]
                    modules.append(f"{module_path}.module.{module_name}")
                elif module_path == None: 
                    failed_modules.append(root)

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


    def resolve_module_class(self,module):
        assert module in self.simple2module, f'options are {list(self.simple2module.keys())}'
        module_path = self.simple2module[module]
        actor_class= self.get_object(module_path)
        return actor_class

    def launch(self, module:str, 
                    refresh:bool=False,
                    resources:dict = {'num_gpus':0, 'num_cpus': 1},
                    max_concurrency:int=100,
                    name:str=None,
                     **kwargs):
        actor = kwargs.pop('actor', {})
        actor['name'] = actor.get('name', name if isinstance(name, str) else module)
        actor['max_concurrency'] = actor.get('max_concurrency', max_concurrency)
        actor['refresh'] = actor.get('refresh', refresh)
        actor['resources'] = actor.get('resources', resources)
        kwargs['actor'] = actor
        actor_class = self.resolve_module_class(module)
        return actor_class.deploy(**kwargs)

    get_actor = add_actor = launch_actor = launch

    module_tree = module_list


