


import os, sys

from commune.gradio.api.module import register
sys.path.append(os.environ['PWD'])
import gradio
from commune import BaseModule
from inspect import getfile
import inspect
import socket
from commune.utils import SimpleNamespace

class ExampleModule(BaseModule):
    default_config_path =  'gradio.example'

    # @register(inputs=['text', 'text'], outputs=['json'])
    def bro(self, input1, input2, image):
        return {"output" : input1 + input2}

    
    
    def __gradio__(self):
        BaseModule.__gradio__(self)
        return gradio.Interface(fn=self.bro, inputs=["text", "text", 'image'], outputs=["text"] )