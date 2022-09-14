
# Create Ocean instance
import streamlit as st
import os, sys
sys.path.append(os.getenv('PWD'))
from commune import BaseModule
from functools import partial
import ray


class ClientModule(BaseModule):

    default_config_path = 'ray.client.module'
    def __init__(self, config=None, **kwargs):
        BaseModule.__init__(self, config=config)
        self.config['server'] = kwargs.get('server', self.config.get('server'))
        self.server_module =self.get_actor(self.config['server'])
        self.parse()
    
    def parse(self):
        for fn_key in self.server_module._ray_method_signatures.keys():

            def fn(self, fn_key,server, *args, **kwargs):
                ray_get = kwargs.pop('ray_get', False)
                object_id =(getattr(server, fn_key).remote(*args, **kwargs))
                if ray_get == True:
                    return ray.get(object_id)

                else:
                    return object_id

            setattr(self, fn_key, partial(fn, self, fn_key, self.server_module))
        
        
if __name__ == '__main__':
    module = ClientModule.deploy(actor=True)
    # st.write(module.get_functions(module))

