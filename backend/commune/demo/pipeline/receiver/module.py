import os, sys
sys.path.append(os.environ['PWD'])
from commune import Module


class RecieverModule(Module):
    def receive(self):
        return 'bro'

if __name__ == '__main__':
    import streamlit as st
    module = DemoModule.deploy()
    st.write(module.config)