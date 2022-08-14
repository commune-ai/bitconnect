
import os, sys
if __name__ == "__main__":
    sys.path[0] = os.environ['PWD']
from copy import deepcopy
from commune.ray import ActorBase

class ClientManager(ActorBase):
    default_cfg_path = f"{os.environ['PWD']}/commune/config/client/manager.yaml"

    def __init__(self, cfg):

        self.cfg = cfg
        self.client = self.get_clients(clients_cfg=cfg.get('client'))


    @staticmethod
    def get_clients(clients_cfg=None):
        '''

        Args:
            config: dict[str, dict]
                mapping the client key with their client initialization parameters

        return
            dict[str, Client]
        '''
        clients = {}
        for client_key, client_cfg in clients_cfg.items():
            if isinstance(client_cfg, dict):
                clients[client_key] = ClientManager.get_module(cfg=client_cfg)
            else:
                clients[client_key] = client_cfg
        return clients



    def read(self, params, client, verbose=False):

        params = deepcopy(params)

        if callable(getattr(self.client[client], 'load', None)):
            # maybe some clients need other clients

            return self.client[client].load(**params)
        else:
            if verbose:
                print(f'{client} does not have load method')

    def write(self, data, params, client,  verbose=False):
        
        params = deepcopy(params)
        if callable(getattr(self.client[client], 'write', None)):
            # maybe some clients need other clients
            #TODO should this be an async call?
            self.client[client].write(**params,
                                    data=data)
        else:
            if verbose:
                print(f'{client} does not have write method')

    def load(*args, **kwargs):
        return self.read(*args, **kwargs)
    def save(*args, **kwargs):
        return self.write(*args, **kwargs)

if __name__ == "__main__":
    ClientManager.deploy(actor=False)
    print(dir(ClientManager))
