





import os, sys
sys.path.append(os.environ['PWD'])
import gradio
from commune import BaseModule
from commune.utils import *






class ClientModule(BaseModule):
    default_config_path =  'gradio.client.module'

    def bro(self, bro,  fam='bro', output_example={'bro': 'fuck you jesus'}):
        return output_example 

from commune.utils import *



if __name__ == "__main__":

    import streamlit as st
    module = ClientModule()
    output_dict = {}
    
    module_list  = module.client.rest.get(endpoint='list', params={'path_map':False})

    st.write(module_list)
    # st.write(module.client.rest.get(endpoint='module/port2module'))
    # module_path = 'gradio.client.module.ClientModule'
    
    st.write(module.client.rest.get(endpoint='rm_all'))
    st.write(module.client.rest.get(endpoint='port2module'))
    module.client.rest.get(endpoint='add', params=dict(module='gradio.client.module.ClientModule'))
    st.write(module.client.rest.get(endpoint='port2module'))
    module.client.rest.get(endpoint='add', params=dict(module='gradio.client.module.ClientModule'))
    st.write(module.client.rest.get(endpoint='port2module'))

    module.client.rest.get(endpoint='module/list', params={'path_map':False})
    # st.write(module.client.rest.get(endpoint='module/getattr', params={'key':'subprocess_map'}))




    # st.write(module.client.rest.get(endpoint='rm', params={'port': 7865}))
    # st.components.v1.iframe('http://0.0.0.0:7865', width=None, height=500, scrolling=True)

    # st.write(module.load_object(**module_dict['bro']['output'][0]))


