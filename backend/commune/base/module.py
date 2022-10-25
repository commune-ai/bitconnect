from commune.utils import get_object, dict_any, dict_put, dict_get, dict_has, dict_pop, deep2flat
from commune.config.loader import ConfigLoader
from commune.ray.actor import Module
import streamlit as st
import os
from .utils import enable_cache, cache
from munch import Munch
import inspect
from copy import deepcopy

class Module:
    client = None
    client_module_class_path = 'client.manager.module.ClientModule'
    # assumes Module is .../{src}/base/module.py
    root_path = '/'.join(__file__.split('/')[:-2])
    root = root_path
    
    def __init__(self, config=None, override={}, client=None ,**kwargs):

        Module.__init__(self,config=config, override=override, **kwargs)

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

    def get_module_class(self,module:str):
        if module in self.simple2module:
            module_path = self.simple2module[module]
        elif module in self.module2simple:
            module_path = module
        else:
            raise Exception(f'({module}) not in options {list(self.simple2module.keys())} (short) and {list(self.simple2module.values())} (long)')
        module_class= self.import_object('commune.'+module_path)
        return module_class

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


    def launch_actor(self, module:str,**kwargs):
        

        default_actor = {
                'name' : False,
                'resources':  {'num_gpus':0, 'num_cpus': 1},
                'max_concurrency' :  100,
                'refresh' : False, 
            }

        if inspect.isclass(module):
            module
        actor = kwargs.pop('actor', default_actor)
        if actor == False:            
            kwargs['actor'] = False
        elif actor == True:
            kwargs['actor'] = default_actor
        elif isinstance(actor, dict):
            # if the actor
            kwargs['actor'] = {**default_actor, **actor}
        else:
            raise NotImplemented(actor)

        return self.get_module_class(module).deploy(**kwargs)

    get_actor = get_module =add_actor = launch_actor = launch
    module_tree = module_list

    def load_module(self, module:str, fn:str=None ,kwargs:dict={}, actor=False, **additional_kwargs):
       
        
        try:
            module_class = self.import_object(module)
        except:
            module_class =  self.get_module_class(module)

        module_init_fn = fn
        module_kwargs = {**kwargs, **additional_kwargs}

        if module_init_fn == None:

            module_object =  module_class(**module_kwargs)
        else:
            module_init_fn = getattr(module_class,module_init_fn)
            module_object =  module_init_fn(**module_kwargs)
        

        if actor

        return module_object

    #############

    # RAY ACTOR TINGS, TEHE
    #############

    default_ray_env = {'address': 'auto', 'namespace': 'default'}
    ray_context = None
    config_loader = ConfigLoader(load_config=False)
    root_path = '/'.join(__file__.split('/')[:-2])
    root = root_path
    def __init__(self, config=None, override={}, **kwargs):

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
    
    def get_age(self) :
        return  self.get_current_timestamp() - self.start_timestamp

    @staticmethod
    def get_current_timestamp():
        return  datetime.datetime.utcnow().timestamp()

    @classmethod
    def get_config_path(cls, simple=False):
        config_path = cls.get_module_path(simple=simple)
        if simple == False:
            config_path = config_path.replace('.py', '.yaml')
        return config_path

    def resolve_config(self, config, override={}, local_var_dict={}, recursive=True, return_munch=False, **kwargs):
        
        import streamlit as st ; 
        
        if config == None:
            config =  self.get_config_path(simple=True)
            import streamlit as st
            # st.write(self.get_config_path(simple=True),config, self.config , 'bro')
        elif isinstance(config, str):
            config = config
        elif isinstance(config, dict):
            pass
        else:
            raise NotImplementedError(config)

        config = self.load_config(config=config, 
                             override=override, 
                            local_var_dict=local_var_dict,
                            recursive=recursive,
                            return_munch=return_munch)

        if config == None:
            config = 'base'
            config = self.load_config(config=config, 
                        override=override, 
                        local_var_dict=local_var_dict,
                        recursive=recursive,
                        return_munch=return_munch)
            self.save_config(config=config)

        return config

    @staticmethod
    def load_config(config=None, override={}, local_var_dict={}, recursive=True, return_munch=False):
        """
        config: 
            Option 1: dictionary config (passes dictionary) 
            Option 2: absolute string path pointing to config
        """
        return Module.config_loader.load(path=config, 
                                    local_var_dict=local_var_dict, 
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
        return cls.config_loader.save(path=path, 
                                      cfg=config)
    @classmethod
    def default_cfg(cls, override={}, local_var_dict={}):

        return cls.config_loader.load(path=cls.get_config_path(), 
                                    local_var_dict=local_var_dict, 
                                     override=override)

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
        actor_id = Module.get_actor_id(actor)
        actor.config_set.remote('actor.id', actor_id)

        actor_name = ray.get(actor.getattr.remote('actor_name'))
        setattr(actor, 'actor_id', actor_id)
        setattr(actor, 'actor_name', actor_name)
        setattr(actor, 'id', actor_id)
        setattr(actor, 'name', actor_name)
        return actor


    @classmethod 
    def deploy(cls, actor=False , skip_ray=False, wrap=False,  **kwargs):
        """
        deploys process as an actor or as a class given the config (config)
        """
        config = kwargs.pop('config', None)
        config = Module.resolve_config(self=cls,config=config, **kwargs)
        if cls.ray_initialized():
            skip_ray = True

        if skip_ray == False:
            ray_config = config.get('ray', {})
            try:
                ray_context =  cls.get_ray_context(init_kwargs=ray_config)
            except ConnectionError:
                cls.ray_start()
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
            # import streamlit as st
            # st.write(actor_config, kwargs)
            try:

                actor_config['name'] =  actor_config.get('name', cls.get_default_actor_name())
                actor_config['resources'] =  actor_config.get('resources', {'num_cpus': 1.0, 'num_gpus': 0.0})
                config['actor'] = actor_config
                kwargs['config'] = config
                actor = cls.deploy_actor(**actor_config, **kwargs)
                actor_id = cls.get_actor_id(actor)  
                actor =  cls.add_actor_metadata(actor)
            except ray.exceptions.RayActorError:
                actor_config['refresh'] = True
                config['actor'] = actor_config
                kwargs['config'] = config
                actor = cls.deploy_actor(**actor_config, **kwargs)
                actor_id = cls.get_actor_id(actor)  
                actor =  cls.add_actor_metadata(actor)

            if wrap:
                actor = cls.wrap_actor(actor)

            return actor 
        else:
            kwargs['config'] = config
            kwargs['config']['actor'] = None
            return cls(**kwargs)

    default_ray_env = {'address':'auto', 'namespace': 'default'}
    @classmethod
    def get_ray_context(cls,init_kwargs=None, reinit=True):
        
        if cls.ray_initialized():
            return

        if init_kwargs == None:
            init_kwargs = cls.default_ray_env

            
        if isinstance(init_kwargs, dict):

            
            for k in cls.default_ray_env.keys():
                default_value= cls.default_ray_env.get(k)
                init_kwargs[k] = init_kwargs.get(k,default_value)
                assert isinstance(init_kwargs[k], str), f'{k} is not in args'
            
            if Module.ray_initialized() and reinit == True:
                ray.shutdown()

            init_kwargs['include_dashboard'] = True
            init_kwargs['dashboard_host'] = '0.0.0.0'
            # init_kwargs['_system_config']={
            #     "object_spilling_config": json.dumps(
            #         {"type": "filesystem", "params": {"directory_path": "/tmp/spill"}},
            #     )
            # }
            init_kwargs['ignore_reinit_error'] = True
            return ray.init(**init_kwargs)
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
 

    def create_actor(cls,
                 name,
                 cls_kwargs,
                 detached=True,
                 resources={'num_cpus': 0.5, 'num_gpus': 0},
                 max_concurrency=5,
                 refresh=False,
                 return_actor_handle=False,
                 verbose = True,
                 redundant=False):
        '''
          params:
              config: configuration of the experiment
              run_dag: run the dag
              token_pairs: token pairs
              resources: resources per actor
              actor_prefix: prefix for the data actors
          '''
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
            if actor_exists(name):
                kill_actor(actor=name,verbose=verbose)

        if redundant:
            # if the actor already exists and you want to create another copy but with an automatic tag
            actor_index = 0
            while not actor_exists(name):
                name =  f'{name}-{actor_index}' 
                actor_index += 1


        if not actor_exists(name):
            
            try:
                actor_class = ray.remote(cls)
                actor_handle = actor_class.options(**options_kwargs).remote(**cls_kwargs)
            except ValueError:
                pass


        
        return ray.get_actor(name)





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
    def kill_actor(actor):
        kill_actor(actor)
        return f'{actor} killed'
    
    @staticmethod
    def actor_exists(actor):
        if isinstance(actor, str):
            return actor in Module.list_actor_names()
        else:
            raise NotImplementedError
    @staticmethod
    def get_actor(actor_name, wrap=False):
        actor =  ray.get_actor(actor_name)
        actor = Module.add_actor_metadata(actor)
        if wrap:
            actor = Module.wrap_actor(actor=actor)
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
        actor_config =  self.config.get('actor', {})
        if actor_config == None:
            actor_config = {}
        return actor_config.get('name')
    
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

    @staticmethod
    def load_object(module:str, __dict__:dict, **kwargs):
        kwargs = kwargs.get('__dict__', kwargs.get('kwargs', {}))
        return Module.import_object(module)(**kwargs)



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
        return self.get_module_path()

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
        # st.write(dir(process))
        memory_info = process.memory_full_info()._asdict()
        memory_info['percent'] = process.memory_percent()
        memory_info['ratio'] = memory_info['percent'] / 100
        return memory_info


    @staticmethod
    def list_objects( *args, **kwargs):
        return list_objects(*args, **kwargs)

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

        return list_tasks(*args, **kwargs)

    @classmethod
    def get_module_path(cls, simple=True):
        module_path =  inspect.getmodule(cls).__file__
        if simple:
            module_path = os.path.dirname(module_path.replace(Module.root, '')).replace('/', '.')[1:]

        return module_path

    @staticmethod
    def list_nodes( *args, **kwargs):
        return list_nodes(*args, **kwargs)