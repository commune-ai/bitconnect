

import os
import sys
from copy import deepcopy
sys.path.append(os.environ['PWD'])
from commune.utils import dict_put, get_object, dict_has
from commune import Module


class ContractModule(Module):
    def __init__(self, config=None, network=None, account=None, **kwargs):

        Module.__init__(self, config=config, network=None, **kwargs)

        self.set_network(network=network)
        self.set_account(account=account)


        
    @property
    def address(self):
        return self.contract.address


    @property
    def function_abi_map(self):
        return {f_abi['name']:f_abi for f_abi in self.abi}
    @property
    def function_names(self):
        return list(self.function_abi_map.keys())


    def call(self, function, args=[]):
        if len(args) == 0:
            args.append({'from': self.account})
        output = getattr(self.contract, function)(*args)
        return self.parseOutput(function=function, outputs=output)


    def parseOutput(self, function, outputs):
        output_abi_list = self.function_abi_map[function]['outputs']
        
        parsedOutputs = {}
        for i,output_abi  in enumerate(output_abi_list) :
            output_key = i 
            if output_abi['name']:
                output_key = output_abi['name']
            
            parsedOutputs[output_key] = outputs[i]
            if 'components' in output_abi:
                component_names = [c['name'] for c in output_abi['components']]
                
                parseStruct = lambda o:  dict(zip(component_names, deepcopy(o)))
                if type(outputs[i]) in [list, tuple, set]:
                    parsedOutputs[output_key] = list(map(parseStruct, outputs[i]))
                else:
                    parsedOutputs[output_key] = parseStruct(outputs[i])
        
        return parsedOutputs

    artifacts_path = f'{os.environ["PWD"]}/artifacts/'

    contracts_path = f'{os.environ["PWD"]}/contracts/'
    @property
    def contract_paths(self):
        return list(filter(lambda f: f.endswith('.sol'), self.client.local.glob(self.contracts_path+'**')))
  
    @property
    def contracts(self):
        contracts = []
        for path in self.contract_paths:
            contracts += [os.path.splitext(path)[0].replace('/', '.')]
        return contracts
    @property
    def contract2path(self):
        return dict(zip(self.contracts, self.contract_paths))


    def get_artifact(self, path):
        available_abis = self.contracts + self.interfaces

        if path in self.contract_paths:
            root_dir = os.path.join(self.artifacts_path, 'contracts')
        elif path in self.interface_paths:
            root_dir = os.path.join(self.artifacts_path, 'interfaces')
        else:
            raise Exception(f"{path} not in {available_abis}")
        json_name = os.path.basename(path).replace('.sol', '.json')

        artifact_path = os.path.join(root_dir, path, json_name)
        artifact = self.client.local.get_json(artifact_path)
        return artifact


    def get_abi(self,path):
        return self.get_artifact(path)['abi']
    interfaces_path = f'{os.environ["PWD"]}/interfaces/'
    @property
    def interface_paths(self):
        return list(filter(lambda f: f.endswith('.sol'),self.client.local.glob(self.interfaces_path+'**')))

    @property
    def interfaces(self):
        interfaces = []
        for path in self.interface_paths:
            interfaces += [os.path.splitext(path)[0].replace('/', '.')]
        return interfaces
    @property
    def interface2path(self):
        return dict(zip(self.interfaces, self.interface_paths))


    @property
    def artifact_paths(self): 
        full_path_list = list(filter(lambda f:f.endswith('.json') and not f.endswith('dbg.json') and os.path.dirname(f).endswith('.sol'),
                            self.client.local.glob(f'{self.artifacts_path}**')))
        
        
        return full_path_list
    
    @property
    def artifacts(self):
        artifacts = []
        for path in self.artifact_paths:
            simple_path = deepcopy(path)
            simple_path = simple_path.replace(self.artifacts_path, '')
            artifacts.append(simple_path)
        return artifacts


    def connected(self):
        return bool( self.web3.__class__.__name__ == 'Web3')

    def disconnected(self):
        return not self.connected()

    def set_web3(self, web3=None):
        self.web3 = web3
        return self.web3
    def set_network(self, network = None, web3=None):
        if network.__class__.__name__ == 'NetworkModule':
            network = network
            web3 = network.web3 
        
        st.write('network', network)
        self.network = network
        self.web3 = web3
        return network

    connect_network = set_network

    def set_account(self, private_key):
        private_key = os.getenv(private_key, private_key)
        self.account = AccountModule(private_key=private_key)
        return self.account
    
    def compile(self):
        # compile smart contracts in compile
        return self.run_command('npx hardhat compile')
        
    @property
    def network_modes(self):
        return list(self.network_config.keys())

    @property
    def available_networks(self):
        return ['local', 'ethereum']

    @property
    def network_config(self):
        network_config_path = f'{self.root}/web3/data/network-config.yaml'
        return self.client.local.get_yaml(network_config_path)

    @property
    def contract_paths(self):
        contracts = list(filter(lambda f: f.startswith('contracts'), self.artifacts))
        return list(map(lambda f:os.path.dirname(f.replace('contracts/', '')), contracts))

    @property
    def interfaces(self):
        interfaces = list(filter(lambda f: f.startswith('interfaces'), self.artifacts))
        return list(map(lambda f:os.path.dirname(f.replace('interfaces/', '')), interfaces))


    def resolve_web3(self, web3):
        if web3 == None:
            web3 = self.web3
        self.set_web3(web3)
        return web3

    def resolve_account(self, account):
        if account == None:
            account = self.account
        self.set_account(account)
        return account

    def set_account(self, account):
        self.account = account
        if account != None:
            account.set_web3(self.web3)
    

    def deploy_contract(self, contract = 'token.ERC20.ERC20', args=['AIToken', 'AI'],tag=None, web3=None, account=None, network=None):
        
        simple_contract_path = contract
        contract_path = self.resolve_contract_path(simple_contract_path)
        
        network = self.resolve_network(network)
        web3 = self.resolve_web3(web3)
        account = self.resolve_account(account)

        tag = self.resolve_tag(tag)


        assert contract in self.contracts
        contract_artifact = self.get_artifact(contract_path)
        contract_class = web3.eth.contract(abi=contract_artifact['abi'], 
                                    bytecode= contract_artifact['bytecode'],)
        # st.write(contract_artifact['abi'])

        st.write(account.address)
        nonce = web3.eth.get_transaction_count(account.address) 
        st.write(nonce, 'NONCE')
        construct_txn = contract_class.constructor(*args).buildTransaction(
                            {
                                    'from': account.address,
                                    'gasPrice':web3.eth.generate_gas_price(),
                                    'nonce': nonce
                            }
        )
        # sign the transaction
        signed_tx = account.sign_tx(construct_txn)
        tx_hash = web3.eth.send_raw_transaction(signed_tx)
        tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)

        contract_address = tx_receipt.contractAddress 
        
        self.register_contract(contract_path=simple_contract_path, network=network.network, contract_address=contract_address, tag=tag)
        
        self.address2contract
        return  tx_receipt.contractAddress 
        # st.write(f'Contract deployed at address: { tx_receipt.contractAddress }')


    @property
    def registered_contracts(self):
        return self.get_json('registered_contracts', {})

    def get_contract(self, address=None, contract=None, tag=None , web3=None, account=None):
        web3 = self.resolve_web3(web3)
        account = self.resolve_account(account)
    
        if isinstance(address, str):
            contract_path = self.address2contract[address]
        
        elif isinstance(contract, str):
            # if contract has tag {contract_name}-{tag}
            if len(contract.split('-'))==2:
                tag = contract.split('-')[1] 

            # if the tag is None, default to "default_tag"
            tag = self.resolve_tag(tag)
            contract_path = contract
        else:
            raise Exception('please specify the contract')

        

    @property
    def address2contract(self):
        registered_contracts = self.registered_contracts
        st.write(registered_contracts)
        address2contract = {}
        for network, contract_path_map in registered_contracts.items():
            for contract_path, contract_tag_map in contract_path_map.items():
                for tag, contract_address_list in contract_tag_map.items():
                    for contract_address in contract_address_list:
                        address2contract[contract_address+'-'+tag] = contract_path
        return address2contract


    @property
    def contract2address(self):
        return {v:k for self.address2contract}

    @property
    def address2network(self):
        registered_contracts = self.registered_contracts
        address2network = {}
        for network, contract_path_map in registered_contracts.items():
            for contract_path, contract_tag_map in contract_path_map.items():
                for tag, contract_address_list in contract_tag_map.items():
                    for contract_address in contract_address_list:
                        address2network[contract_address] = network
        
        return address2network

    @property
    def network2address(self):
        network2address = {}
        for address, network in self.address2network.items():
            if network in network2address:
                network2address[network].append(address)
            else:
                network2address[network] = [address]
        return network2address


    @property
    def network2contract(self):
        network2contract = {}
        for network, address_list in self.network2address.items():
            network2contract[network] = address2contract[address] for address in address_list
        return network2contract
    
    def contract2network(self):
        address2contract = self.address2contract
        contract2network ={}
        for address, network in self.address2network.items():
            contract2network[address2contract[address]] = network

        return contract2network


    default_tag = 'v0'
    def resolve_tag(self, tag=None): 
        if tag == None:
            tag = self.default_tag
        return tag

    def register_contract(self, network:str,
                            contract_path:str , 
                            contract_address:str,  
                            tag=None, 
                            refresh=True):

        if refresh:
            self.rm_json('registered_contracts')

        st.write(self.interfaces)

        registered_contracts = self.registered_contracts
        st.write(registered_contracts, 'contracts')
        if network not in registered_contracts:
            registered_contracts[network] = {}
        if contract_path not in registered_contracts[network]:
            registered_contracts[network][contract_path] = {}
        
        if tag not in registered_contracts[network][contract_path]:
            registered_contracts[network][contract_path][tag] = []

        assert isinstance(registered_contracts[network][contract_path][tag], list)
        registered_contracts[network][contract_path][tag].append(contract_address)
    
        self.put_json('registered_contracts', registered_contracts)

        return registered_contracts

    def resolve_network(self, network):
        if network == None:
            network = self.network
        self.set_network(network)
        return network
    
    
    def resolve_contract_path(self,  path):
        contract_path = self.contract2path.get(path, None)
        assert contract_path in self.contract_paths
        return contract_path


        
    def __reduce__(self):
        deserializer = ContractModule
        serialized_data = (self.config)
        return deserializer, serialized_data
if __name__ == '__main__':
    import streamlit as st
    import ray

    contract = ContractModule()
    network = Module.launch('web3.network')
    account = Module.launch('web3.account')
    contract.set_network(network)
    contract.set_account(account)

    st.write(contract.deploy_contract(contract='token.ERC20.ERC20'))



    # st.write(network)
    # st.write('FAM')
    # st.write()
    # st.write(contract.connected())
    # contract.set_web3(network.web3)
    # account.set_web3(network.web3)
    # contract.set_account(account)
    # st.write(contract.connected())
    # st.write(contract.compile())
    # st.write(contract.deploy_contract())
    # st.write(contract.contracts)


    # contract.put_json('contract', {'hey': 'bro'})
    # st.write(contract.get_json('bro', {'hey': 'bro'}))


    st.write()
   

    # st.write(account.get_balance())   
    # st.write(ContractModule.get_actor('web3.account'))
    # st.write(ContractModule.get_actor('web3.network'))


 