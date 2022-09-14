





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
    
    st.write(module.client.__dict__)
    module_list  = module.client.rest.get(endpoint='module/list', params={'path_map':False})

    st.write(module_list)
    # module_schemas  = module.client.rest.get(endpoint='module/schemas')
    # module_schemas  = module.client.rest.get(endpoint='module/schema')

    # st.write(output_dict)
    st.write(module.client.__dict__)
    module_dict = module.client.rest.get(endpoint='module/schema', params={'module': 'gradio.client.module.ClientModule','gradio':True})
    st.write(module_dict)
    # st.write(module.load_object(**module_dict['bro']['output'][0]))


    