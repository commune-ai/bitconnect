


import os, sys
sys.path.append(os.environ['PWD'])
import gradio as gr
from commune import Module
from inspect import getfile
import streamlit as st
import inspect
import socket
from commune.utils import SimpleNamespace

class ExampleModule(Module):
    default_config_path =  'gradio.example'
    
    def bro(self, input1 : str, inpupt2 : str) -> str:
        return ""

    def fn(self, input : str) -> str:
        return "Hello World"

    @classmethod
    def gradio(cls):
        return gr.Interface(lambda inputs : f"Hello Welcome, {inputs}. This is a Test", inputs="text", outputs='text')      

    @classmethod
    def streamlit(cls):
        st.write("Hello World")
        st.write("This function is a inherited from Class Module which allows the API to check if this function is Streamlit streamable")

if __name__ == "__main__":
    ExampleModule().run()




