<div align="center">

# **Bitconnect** <!-- omit in toc -->

</div>

***
## Summary 

This is a big endeaver, which is to connect ML developers and avoid Big Tech from rulling the plebs. This node is a set of different nodes that combine into a decentralized package. Please note that its mainly me and my brother, so we are trying our best to squash bugs.
***

## Setup

1. Clone Repo and its Submodules

```
git clone https://github.com/commune-ai/bitconnect.git

```


2. Enter bitconnect folder and pull the submodules.
```
cd bitconnect;
git submodule update --init --recursive;
```


2. Spinnup Docker Compose
```
make up
```

3. Run the Streamlit app
```
make app arg=
```


## Commands

- Run 
    
     ```make up```
-  Enter Backend 
    
     ``` make bash arg=backend```
-  Enter Subtensor 
    
     ``` make bash arg=subtensor```


- Run Streamlit Server
    
     ``` make app arg={path_to_python_file}```

