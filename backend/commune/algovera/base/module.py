from commune import Module


import streamlit as st

class Module(Module):
    default_config_path='algovera.base.module'

    def bro(self, input_text= 'hello'):
        return input_text


if __name__ == '__main__':
    algovera_module = Module.deploy(actor={'refresh': False})
    st.write()
# st.write(['.'.join(v.split('.')[:-2]) for v in Module().module_tree])