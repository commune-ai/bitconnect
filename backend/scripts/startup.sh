#/bin/bash
ray start --head;
nohup python commune/gradio/api/module.py --api &> api.out&
# pip install https://github.com/opentensor/cubit/releases/download/v1.1.1/cubit-1.1.1-cp38-cp38-linux_x86_64.whl;
tail -F anything;

