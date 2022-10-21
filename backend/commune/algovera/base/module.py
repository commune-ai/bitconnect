from commune import BaseModule


import streamlit as st

class Module(BaseModule):
    default_config_path='algovera.base.module'

    def bro(self, input_text= 'hello'):
        return input_text


if __name__ == '__main__':
    algovera_module = Module.deploy(actor={'refresh': False})

# st.write(['.'.join(v.split('.')[:-2]) for v in Module().module_tree])