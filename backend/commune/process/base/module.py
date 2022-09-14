import requests
import json
import datetime
import pandas as pd
import os
import sys
from copy import deepcopy
import ray
if os.environ['PWD'] not in sys.path:
    sys.path.append(os.environ['PWD'])
from commune.config import ConfigLoader
from commune.client.manager import ClientManager
from commune.config import ConfigLoader
from commune.ray import ActorBase
from commune.utils import dict_get, dict_put, get_object, dict_has
# function to use requests.post to make an API call to the subgraph url
from commune.ray.utils import create_actor



class BaseModule(ActorBase):
    """
    Manages loading and processing of data

    pairs: the pairs we want to consider
    base: the timeframe we consider buys on
    higher_tfs: all the higher timeframes we will use
    """
    defualt_config_path = None
    default_client_config_path = f'{os.environ["PWD"]}/commune/client/manager.yaml'
    default_client_config = ActorBase.load_config(default_client_config_path)
    module = {}
    loop_running = False






    cache = {} # for caching things
    def __init__(
            self, 
            config=None, 
            ):
        
        ActorBase.__init__(self,config)
        self.connect_clients()
        self.get_sub_modules()


        self.setup()
        self.resolve_queue_config()
        self.template_config = deepcopy(self.config)
        
    def setup(self):
        pass

    def resolve_queue_config(self):
        for mode in ['in', 'out']:
            queue_keys = ['queue', mode]
            queue_topic = f"{self.config['module']}.{mode}"
            if not dict_has(self.config, keys=queue_keys):
                dict_put(self.config, keys=queue_keys, value=queue_topic)

    def override(self, override={}):
        """
        overiide config with override dictionary
        
        """
        for k,v in override.items():
            dict_put(input_dict=self.__dict__,keys=k, value=v)

    def has_submodule(self, key:str):
        has_submodule = dict_has(input_dict=self.sub_modules, keys=key)
        if has_submodule:
            assert dict_has(input_dict=self.__dict__, keys=key)
        return has_submodule

    def list_submodules(self):
        self.sub_modules.keys()

    def rm_submodule(self, key:str):
        if self.has_submodule(key):
            del self.sub_modules[key]
            del self.__dict__[key]

    def add_submodule(self, key:str, config:dict):
        module_key = key
        module_config = config
        if isinstance(module_config, dict):
            if 'config' in module_config:

                '''
                config: config
                override: config override
                '''
                # module key must be a string 

                actor_kwargs = module_config.get('actor', {})
                override_kwargs = module_config.get('override', {})

                if not actor_kwargs:
                    override_kwargs['client'] = self.client_manager


                module = self.get_module(config=module_config['config'], 
                                        actor=actor_kwargs,
                                        override=override_kwargs)

            elif  'module' in module_config:
                '''
                module: string of where module is located

                '''

                # module key must be a string 
                assert isinstance(module_config['module'],str)

                actor_kwargs = module_config.get('actor', {})
                override_kwargs = module_config.get('override', {})

                if not actor_kwargs:
                    override_kwargs['client'] = self.client_manager

                module = self.get_module(config=module_config['module'],
                                            actor= actor_kwargs, 
                                            override=override_kwargs)

        elif isinstance(module_config, str):
            if self.actor_exists(module_config):

                '''
                if the module is a ray actor, then get the handle 
                note that you need to call remote() for every function
                '''
                module = self.get_actor(module_config)
            else:
                
                module = self.get_module(config=module_config, override ={'client': self.client_manager})   

        # inserts the module based on the key of the module in sub_module config
        dict_put(input_dict=self.__dict__,
                            keys= module_key,
                            value= module)
        self.sub_modules[module_key] = module

    def get_sub_modules(self):
        self.sub_modules = {}
        if 'sub_module' in self.config:
            assert isinstance(self.config['sub_module'], dict)
            for module_key,module_config in self.config['sub_module'].items():
                self.add_submodule(module_key, module_config)


    def change_state(self):
        pass

    def object_config_override(self):
        if self.object_dict.get('override'):
            self.override(self.object_dict['override'])


    def get_trigger_queue(self, item=None, mode='in', block=True): 

        
        if item == None:
            item  = deepcopy(self.object_dict)


        queue_path = ['queue',mode]


        topic = dict_get(self.object_dict, queue_path)
        if not isinstance(topic, bool) and bool(topic):
            topic = dict_get(self.config, keys=queue_path)
        # if the topic is false

        if bool(topic):
            if mode == 'in':
                return  self.client['ray'].queue.get(topic=topic, block=block)
            elif mode == 'out':

                return  self.client['ray'].queue.put(topic=topic, item=item , block=block)
            else:
                ('BROOOOO, the mode can only be in or out')
    
        return item


    def run_preprocess(self, **kwargs):
        self.config = deepcopy(self.template_config)
        self.object_config_override()
        self.change_state()
        self.read_state()

    def run_process(self,**kwargs):
        return_obj = self.process(**kwargs)
        if isinstance(return_obj, dict):
            self.object_dict = return_obj
    

        
    def run_postprocess(self, **kwargs):
        self.get_explain()
        self.write_state()   
        self.last_run_timestamp = datetime.datetime.utcnow().timestamp()
        # self.get_trigger_queue(item=self.object_dict, mode='out')



    def run(self, **kwargs):
        self.object_dict = kwargs
        self.run_preprocess(**self.object_dict)
        self.run_process(**self.object_dict)
        self.run_postprocess(**self.object_dict)
        return self.object_dict


    @property
    def last_run_delay(self):
        current_timestamp = datetime.datetime.utcnow().timestamp()
        if not hasattr(self, "last_run_timestamp"):
            return current_timestamp
        return current_timestamp - self.last_run_timestamp

    def run_again(self, threshold=3600):
        return self.last_run_delay > threshold

    @staticmethod
    def default_clients(clients =[], config=None):
        if config is None:
            config =  BaseModule.load_config(BaseModule.default_client_config_path)
        if len(clients)>0 :
            config['client'] = {client: config['client'][client] for client in clients}

        return BaseModule.get_module(config=config).client


    def connect_clients(self):
        
        client_config = self.config.get('client', None)
        if isinstance(client_config, dict):
            self.client_manager = self.get_module(config=client_config)
            self.client = self.client_manager.client
        elif client_config == None:
            self.client, self.client_manager = None, None
        else:
            self.client_manager = self.config['client']
            self.config['client'] = self.client_manager.config['client']
            self.client = self.client_manager.client


    def process(self, **kwargs):
        raise NotImplementedError("Implement this shizzz fam")

    def read_state(self):
        if 'read' not in self.config or self.client == None:
            return
        for object_key, store_dict in self.config['read'].items():

            # only avoid reading state if you are refreshing the object or have an ignore flag

            if store_dict.get('refresh', self.config.get('refresh')):
                continue
            elif store_dict.get('ignore'):
                continue


            object_value = None
            object_value = self.client_manager.load(client=store_dict['client'],
                                                        params=store_dict['params'])
            
            if object_value:
                dict_put(self.__dict__, object_key, object_value)
            else:
                if not dict_has(self.__dict__, object_key):
                    dict_put(self.__dict__, object_key, store_dict.get('fallback_value', object_value))

    


    def write_state(self):
        if 'write' not in self.config or self.client == None:
            return
        

        for object_key, store_dict in self.config['write'].items():
            if  store_dict.get('ignore'):
                continue

            # only supported for 
            assert hasattr(self,object_key.split('.')[0]), f"{object_key} not found in self"
            object_value = dict_get(self.__dict__,object_key)
            self.client_manager.write(data=object_value,client=store_dict['client'],
                                                                params=store_dict['params'])


    def cancel_loop(self):
        self.loop_running = False
        self.loop_count= False

    def loop(self,**kwargs):
        self.loop_count = 0
        self.loop_running = True
        while self.loop_running:
            self.run(**kwargs)
            self.loop_count += 1

    def get_explain(self):
        if hasattr(self, 'explain'):
            self.explain_module.run(override={'module': self})
            self.config['explain'] = explain_module.config
    
        elif  'explain' in self.config:
            explain_config = deepcopy(self.config['explain'])
            if dict_has(explain_config, 'sub_module'):
                del explain_config['sub_module']['process']
            explain_module = self.get_module(explain_config)
            explain_module.run(override={'module': self})
            self.config['explain'] = explain_module.config
