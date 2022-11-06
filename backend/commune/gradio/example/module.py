


import os, sys
sys.path.append(os.environ['PWD'])
import gradio
from commune import Module
from inspect import getfile
import inspect
import socket
from commune.utils import SimpleNamespace
import streamlit as st

class ExampleModule(Module):
    default_config_path =  'gradio.example'
    def bro(self, input1=1, inpupt2=10, output_example={'bro': 1}):
        pass

    def gradio(self):
        return None

    @classmethod
    def streamlit(cls):
        st.write(f'### {cls.__name__}')
        self = cls.deploy(actor={'refresh': False}, wrap=True)
        st.write(self.hash({'bro'}))

        st.write(self.account)



