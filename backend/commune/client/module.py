


import streamlit as st
import os, sys
sys.path.append(os.getenv('PWD'))
import datasets
from copy import deepcopy
from commune import BaseModule
class ClientModule(BaseModule):
    default_config_path = 'client.module'
    registered_clients = {}

    def __init__(self, config=None ):
        BaseModule.__init__(self, config=config,get_clients=False)
        self.get_default_clients()
        self.register_clients(clients=self.clients_config)

    @property
    def clients_config(self):
        return self.config.get('client', self.config.get('clients', {}))

    @staticmethod
    def client_path_formater(client:str):
        return f"client.{client}.module."
    def get_default_clients(self):
        self.client_path_dict = dict(
        ipfs = 'client.ipfs.module.IPFSModule',
        local = 'client.local.module.LocalModule',
        s3 = 'client.s3.module.S3Module',
        estuary = 'client.estuary.module.EstuaryModule',
        pinata = 'client.pinata.module.PinataModule',
        rest = 'client.rest.module.RestModule',
        # ray='client.ray.module.RayModule'
        )



        self.default_clients = list(self.client_path_dict.keys())

    def register_clients(self, clients=None):
        print(clients, 'FUCKYOU', self.default_clients)

        if clients == None:
            # resort to list of default cleitns if none
            clients = self.default_clients
        elif type(clients) in [list, dict]:
            if len(clients) == 0:
                clients = self.default_clients
        



        if isinstance(clients, bool):
            if clients == True:
                clients = self.default_clients
            else:
                return

        if isinstance(clients, list):
            assert all([isinstance(c,str)for c in clients]), f'{clients} should be all strings'
            for client in clients:
                self.register_client(client=client)
        elif isinstance(clients, dict):
            for client, client_kwargs in clients.items():
                self.register_client(client=client, **client_kwargs)
        else:
            raise NotImplementedError(f'{clients} is not supported')

    def register_all_clients(self):
        self.register_clients()



    def get_client_class(self, client:str):
        assert client in self.client_path_dict, f"{client} is not in {self.default_clients}"
        return self.get_object(self.client_path_dict[client])

    def register_client(self, client, **kwargs):
        if client in self.blocked_clients:
            return
        assert isinstance(client, str)
        assert client in self.default_clients,f"{client} is not in {self.default_clients}"
        
        
        client_module = self.get_client_class(client)(**kwargs)
        setattr(self, client,client_module )
        self.registered_clients[client] = client_module

    def remove_client(client:str):
        self.__dict__.pop(client)
        self.registered_clients.pop(client)
        return client
    
    delete_client = rm_client= remove_client
    
    def remove_clients(clients):
        return [self.remove_client(client) for client in clients]
            
    delete_clients = rm_clients= remove_clients

    def get_registered_clients(self):
        return self.registered_clients


    @property
    def blocked_clients(self):
        v = None
        for k in ['block', 'blocked', 'ignore']:
            v =  self.clients_config.get('block')
            if v == None:
                continue
            elif isinstance(v, list):
                return v
            
        if v == None:
            v = []
        else:
            raise NotImplementedError(f"v: {v} should not have been here")

        return v

    ignored_clients = blocked_clients
if __name__ == '__main__':
    import streamlit as st
    module = ClientModule()
    st.write()
    st.write(ClientModule._config())
    # st.write(module.__dict__)
