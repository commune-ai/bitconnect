

import os
import sys
from copy import deepcopy
sys.path.append(os.environ['PWD'])
from commune.utils import dict_put, dict_get, get_object, dict_has
from commune import Module
from ocean_lib.ocean.util import get_address_of_type , get_web3


class NetworkModule(Module):

    default_config_path = 'web3.network'

    def __init__(self, config=None, network=None, **kwargs):
        Module.__init__(self, config=config, **kwargs)
        self.set_network(network=network)

    @property
    def network(self):
        network = self.config['network']
        if len(network.split('.')) == 3:
            network = '.'.join(network.split('.')[:-1])
        assert len(network.split('.')) == 2
        return network

    @network.setter
    def network(self, network):
        assert network in self.available_networks
        self.config['network'] = network


    def set_network(self, network:str):
        network = network if network != None else self.config['network']
        
        
        url = self.get_url(network)
        self.network = network
        self.url = url 
        self.web3 = self.get_web3(self.url)
        return self.web3
    connect_network = set_network

    def get_web3_from_url(self, url:str):
        return get_web3(url)
    get_web3 = get_web3_from_url

    @property
    def networks_config(self):
        return self.config['networks']

    @property
    def networks(self):
        return self.get_networks()

    def get_networks(self):

        
        return list(self.networks_config.keys())

    @property
    def available_networks(self):
        return self.get_available_networks()

    def get_available_networks(self):
        networks_config = self.networks_config

        subnetworks = []
        for network in self.networks:
            for subnetwork in networks_config[network].keys():
                subnetworks.append('.'.join([network,subnetwork]))
        return subnetworks
    def get_url_options(self, network):
        assert len(network.split('.')) == 2
        network, subnetwork = network.split('.')
        return list(self.networks_config[network][subnetwork]['url'].keys())

    def get_url(self, network:str='local.main.ganache' ):
        if len(network.split('.')) == 2:
            url_key = self.get_url_options(network)[0]
            network_key, subnetwork_key = network.split('.')
        elif len(network.split('.')) == 3:
            network_key, subnetwork_key, url_key = network.split('.')
        else:
            raise NotImplementedError(network)

        key_path = [network_key, subnetwork_key, 'url',url_key ]
        return dict_get(self.networks_config, key_path )

if __name__ == '__main__':
    import streamlit as st
    module = NetworkModule.deploy(actor=False)
    
    st.write(module.get_url('local.main'))
    st.write(module.set_network('local.main').eth.get_block_number())
    
    # with st.expander('Select Network', True):
    #     network_mode_options = module.network_modes
    #     selected_network_mode = st.selectbox('Select Mode', network_mode_options, 1)
        
    #     if selected_network_mode == 'live':
    #         network2endpoints = {config['name']:config['networks'] for config in module.network_config[selected_network_mode]}
    #         selected_network = st.selectbox('Select Network', list(network2endpoints.keys()), 4)

    #         endpoint2info = {i['name']:i for i in network2endpoints[selected_network]}  
    #         selected_endpoint = st.selectbox('Select Endpoint', list(endpoint2info.keys()) , 4)
    #         network_info = endpoint2info[selected_endpoint]
    #     elif selected_network_mode == 'development':
    #         network2info = {config['name']:config for config in module.network_config[selected_network_mode]}
    #         selected_network = st.selectbox('Select Network', list(network2info.keys()), 4)
    #         network_info = network2info[selected_network]
    #     else:
    #         raise NotImplemented

    #     # st.write(module.run_command("env").stdout)
        
    #     st.write(network_info)
    #     from algocean.account import AccountModule
    #     # os.environ["PRIVATE_KEY"] = "0x8467415bb2ba7c91084d932276214b11a3dd9bdb2930fefa194b666dd8020b99"
    #     account = AccountModule(private_key='PRIVATE_KEY')
    #     st.write(account.account)
    #     def build_command( network_info):
    #         cmd = network_info['cmd']
    #         cmd_settings = network_info['cmd_settings']

    #         for k,v in cmd_settings.items():
    #             cmd += f' --{k} {v}'
            
    #         return  cmd



    #     # st.write(module.run_command(build_command(network_info))) 
    #     # st.write(module.run_command('npx hardhat node', False))
    #     # st.write(network_info['cmd'])
    #     import subprocess
    #     from subprocess import DEVNULL, PIPE
    #     import psutil

    #     # Module.kill_port(8545)

    #     def launch(cmd: str, **kwargs) -> None:
    #         print(f"\nLaunching '{cmd}'...")
    #         out = DEVNULL if sys.platform == "win32" else PIPE

    #         return psutil.Popen(['bash','-c', cmd], stdout=out, stderr=out)



    #     cmd = build_command(network_info)
    #     # # p = subprocess.Popen([sys.executable, '-c', f'"{cmd}"'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    #     # p = launch(cmd)
    #     # import time
    #     # while True:
    #     #     time.sleep(2)
    #     #     st.write( p.stderr.read(1000))
    #     # st.write(module.run_command('npx hardhat node'))

    #     # p = subprocess.call(['ls'], stdout=subprocess.PIPE,
    #     #                                     stderr=subprocess.PIPE,
    #     #                                     shell=True)

    #     # st.write(module.run_command(cmd))
    #     from ocean_lib.ocean.util import get_address_of_type, get_ocean_token_address, get_matic_token_address, get_web3

    #     st.write(os.getenv('GANACHE_URL'))
    #     # # import yaml
    #     # st.write(module.compile())


    #     # st.write(module.client.local.get_yaml(f'{module.root}/web3/data/network-config.yaml'))
    #     # st.write(module.get_abi('token/ERC20/ERC20.sol'))
    #     # st.write(module.get_abi('dex/sushiswap/ISushiswapFactory.sol'))

