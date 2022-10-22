
import os
import sys
import json
sys.path.append(os.getcwd())
from commune import BaseModule
import ray
import requests


class RayModule(BaseModule):
    def __init__(
        self,
        config=None,
        **kwargs
    ):
        BaseModule.__init__(self, config=config,  **kwargs)
        self.queue = self.get_actor(**self.config['servers']['queue'], wrap=True)
        self.object  = self.get_actor(**self.config['servers']['object'], wrap=True)



    # def load_clients(self):
    #     # load object server

    #     self.

if __name__ == '__main__':
    import streamlit as st
    module = RayModule.deploy(actor={'refresh': False},wrap=True)
    st.write(module.actor)




