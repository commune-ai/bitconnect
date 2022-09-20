





import os, sys
sys.path.append(os.environ['PWD'])
import gradio
from commune import BaseModule
from commune.utils import *






class ClientModule(BaseModule):
    default_config_path =  'gradio.client.module'
    def bro(self, fam='bro', output_example={'bro': True}):
        pass

from commune.utils import *



if __name__ == "__main__":

    import streamlit as st
    module = ClientModule()
    output_dict = {}
    
    # module_list  = module.client.rest.get(endpoint='module/list', params={'path_map':False})

    st.write(module.client.rest.get(endpoint='module/port2module'))
    # module_path = 'gradio.client.module.ClientModule'
    # st.write(module.client.rest.get(endpoint='module/add', params=dict(module='gradio.client.module.ClientModule')))



    # module.client.rest.get(endpoint='module/list', params={'path_map':False})
    # st.write(module.client.rest.get(endpoint='module/getattr', params={'key':'subprocess_map'}))

    # module_schemas  = module.client.rest.get(endpoint='module/schema')

    # st.write(module.load_object(**module_dict['bro']['output'][0]))


