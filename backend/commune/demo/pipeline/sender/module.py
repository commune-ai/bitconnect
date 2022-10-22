import os, sys
sys.path.append(os.environ['PWD'])
from commune import BaseModule


class SenderModule(BaseModule):
    def send_echo(self, *args, **kwargs):
        return kwargs
    

if __name__ == '__main__':
    import streamlit as st
    module = DemoModule.deploy()
    st.write(module.send_echo())