import os, sys
sys.path.append(os.environ['PWD'])
from commune import Module


class DemoModule(Module):
    def bro(self):
        return 'bro'

if __name__ == '__main__':
    import inspect
    import streamlit as st
    module = DemoModule.deploy(actor=False)
    st.write(DemoModule.get_config_path(False))
    st.write(DemoModule.queue)