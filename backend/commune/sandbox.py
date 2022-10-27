import streamlit as st
import os,sys
sys.path[0] = os.environ['PWD']

import bittensor
st.write(os.getcwd())



st.write(bittensor)
