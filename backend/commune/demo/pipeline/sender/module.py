import os, sys
sys.path.append(os.environ['PWD'])
from commune import Module


class SenderModule(Module):
    def send_echo(self, *args, **kwargs):
        return kwargs
    

if __name__ == '__main__':
    import streamlit as st
    module = DemoModule.deploy()
    st.write(module.send_echo())