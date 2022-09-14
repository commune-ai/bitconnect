from commune.utils import get_object
from commune.config.loader import ConfigLoader
from commune.ray.actor import ActorModule
import streamlit as st

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
        
        
        print( 'CONFIG', self.config)
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

