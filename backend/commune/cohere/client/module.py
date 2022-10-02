import os, sys
import cohere
sys.path.append(os.environ['PWD'])
from commune import BaseModule
from commune.utils import *
import streamlit as st
import pandas as pd
import numpy as np

class ClientModule(BaseModule):
    default_config_path =  'cohere.client.module'

    def __init__(self, config=None):
        BaseModule.__init__(self, config=config)
        
        self.api = self.config.get('api')
        self.credit = self.config.get('credit')
        self.models = self.config.get('baseline')
        self.title =  self.config.get('title') if self.config.get('title') != "" else ""
        self.Examples = []
        self.example_table = None
        self.prompt = ""
        self.input_table = None
        st.session_state["model"]    = ""
        st.session_state["prompts"]  = []
        st.session_state["examples"] = []
        st.session_state["output"]   = []



    def api_key(self):
        try:
            c = cohere.Client(self.api)
        except Exception:
            st.error("The API Key set to this state does not exist within co:here")
            return 
        st.success("Connected to co:here API ", icon="✅")


    def _test_(self):
        for state in st.session_state.items():
            st.write(state)



    
    def __pricing(self, inputs=[]):
        """
        Determine Price of the call to the Classify API 
            - Current Standard per 1000 calls
               Small  - $5
               Medium - $5
        """
        # given our current call count determine with the amount of inputs
        col1, col2 = st.columns([1,2])
        with col1: 
            st.metric(label=f"{st.session_state['model']}($5/1000 queue) ", value=f"{len(inputs)}", delta=f"$ {(0.005*len(inputs))}")
        with col2: 
            st.metric(label=f"balance ", value=f"${self.credit}", delta=f"-{(0.005*len(inputs))}")
        
       

    def __models(self):
        """
         Determine the models
            - Small
            - Medium
            - Large
            - Later (Custom)
        """
        model = st.selectbox("", self.models, label_visibility="collapsed")
        if not "model" in st.session_state:
            st.session_state["model"] = ""
        
        st.session_state["model"] = model



    def __navagation(self):
        """
        Navagation tool to hold 
            - models
                - Example Models
            - buttons
                - export code
                - share
        """
        
        with st.sidebar:
            
            st.markdown("<h1 style='text-align: center;'>co:here SDK</h1>", unsafe_allow_html=True)
            with st.expander("Models"):
                self.__models()
            with st.expander("Presets"):
                with st.container():
                    st.header("Example Presets")
                    st.selectbox("", self.Examples, label_visibility="collapsed")

            with st.expander("State Of The API Call"):
                st.json(st.session_state)
        
    

    def __upload__(self):
        """
        Upload Button to import cvs to examples
        """
        uploaded_file = st.file_uploader("Choose a csv file", type="csv", accept_multiple_files=False)
        if uploaded_file:
            uploaded_data_read = pd.read_csv(uploaded_file)
            data = [(item[0], item[1]) for item in uploaded_data_read.values]
            st.dataframe(data, width=700, height=178)
            if st.button('Save') and len(uploaded_data_read.values):
                st.session_state["examples"] = data

                
    def __example__(self):
        
        col1, col2 = st.column
        st.checkbox(value=False)


    def _save(self):
        st.write("")


    def _clear(self):
        st.write("cleared")

    def __streamlit__(self):
        """
        launcher for streamlit application
        """

            
        
        st.markdown("<h1 style='text-align: center;'>Playground</h1>", unsafe_allow_html=True)

        st.session_state["title"] = st.text_input(label="Title", placeholder="Name", value=f"{self.title}")        
    
        with st.expander("Examples"):
            self.__upload__()

        with st.expander("Prompts"):
                
                self.prompt = st.text_input(label="", placeholder="Enter Prompt", label_visibility="collapsed")
                col1, col2 = st.columns([1,8])
                
                with col1:
                    added = st.button("append")
                    if added:
                        if not self.prompt in st.session_state["prompts"]:
                            st.session_state["prompts"].append(self.prompt)
                
                with col2:
                    removed = st.button("remove")
                    if removed:
                        if not "prompt" in st.session_state:
                            st.session_state["prompt"] = []
                        if self.prompt in st.session_state["prompts"]:
                            st.session_state["prompts"].remove(self.prompt)            
                if "prompts" in st.session_state: 
                    self.input_table = st.dataframe(pd.DataFrame(np.array(st.session_state["prompts"]),columns=['Inputs']), width=1000,height=175)
                    self.__pricing(st.session_state["prompts"])                    

        self.__navagation()

        with st.expander("output"):
                st.json({})

        


        
        btn1, btn2, btn3 = st.columns([1.5,1.1,9])
        with btn1:
            st.button("Execute", on_click=self._save)
            
        with btn2:
            st.button("save", on_click=self._save)

        with btn3:
            st.button("clear all", on_click=self._clear)

        self._test_()

         
    

if __name__ == "__main__":
    import streamlit as st
    ClientModule().__streamlit__()