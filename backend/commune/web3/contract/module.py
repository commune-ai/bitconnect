

import os
import sys
from copy import deepcopy
sys.path.append(os.environ['PWD'])
from commune.utils import dict_put, get_object, dict_has
from commune import Module
import streamlit as st
from commune.web3.contract.pythonic_contract_wrapper import PythonicContractWrapper
import gradio

class ContractManagerModule(Module):
    def __init__(self, config=None, contract=None, network=None, account=None, compile=True, **kwargs):

        Module.__init__(self, config=config, network=None, **kwargs)

        if compile:
            self.compile()

        self.set_network(network=network)
        self.set_account(account=account)
        self.set_contract(contract=contract)

    @property
    def address(self):
        return self.contract.address

    @property
    def function_abi_map(self):
        return {f_abi['name']:f_abi for f_abi in self.abi}
    @property
    def function_names(self):
        return list(self.function_abi_map.keys())


    @property
    def accounts(self):
        return self.account.accounts

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
        path = self.resolve_contract_path(path)
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
    def set_network(self, network = None):
        if hasattr(self, 'network'):
            assert isinstance(network, str ), f'{network}'
            self.network.set_network(network)
        else:
            if network == None:
                network = self.config['network']
            self.network = self.launch(**network)
        

        self.web3 = self.network.web3

    connect_network = set_network

    def compile(self):
        # compile smart contracts in compile
        return self.run_command('npx hardhat compile')
        
    @property
    def available_networks(self):
        return self.network.available_networks

    @property
    def network_name(self):
        return self.network.network

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
        return web3

    def resolve_account(self, account):
        if account == None:
            account = self.account
        return account

    def set_account(self, account):
        if hasattr(self, 'account'):
            self.account.set_account(account)
        else:
            if account == None:
                account = self.config['account']
            self.account = self.launch(**account)
            if self.account != None:
                self.account.set_web3(self.web3)
        
    
    def get_contract_address(self, contract, version=-1):
        return self.contract2addresses.get(self.network_name, {}).get(contract,[None])[version]
        


    def deploy_contract(self, contract = 'token.ERC20.ERC20', args=['AIToken', 'AI'],  new=True, refresh=False, **kwargs):
        
        simple_contract_path = contract
        contract_path = self.resolve_contract_path(simple_contract_path)
        contract_address =  self.get_contract_address(contract)


        network = self.resolve_network(kwargs.get('network'))
        web3 = self.resolve_web3(kwargs.get('web3'))
        account = self.resolve_account(kwargs.get('account'))
        
        if contract_address == None or new == True:

            assert contract in self.contracts
            contract_artifact = self.get_artifact(contract_path)
            contract_class = web3.eth.contract(abi=contract_artifact['abi'], 
                                        bytecode= contract_artifact['bytecode'],)

            nonce = web3.eth.get_transaction_count(account.address) 
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
            
            self.register_contract(contract_path=simple_contract_path, network=network.network, contract_address=contract_address, refresh=refresh)

        # ensure the contract exists
        assert self.contract_exists(contract_address)
        return self.get_contract(contract_address)
    @property
    def registered_contracts(self):
        return self.get_json('registered_contracts', {})

    @property
    def contract2addresses(self):
        return self.registered_contracts



    def set_contract(self,contract=None, address=None, web3=None, account=None, version=-1):
        if isinstance(contract, str) or isinstance(address, str):
            contract = self.get_contract(contract=contract, address=address , web3=web3, account=account, version=-1)
        elif type(contract).__name__ in ['f']:
            return
        elif contract == None:
            pass
        else:
            raise NotImplementedError

        self.contract = contract
        return self.contract



    def contract_exists(self, contract=''):
        is_address = isinstance(self.address2contract.get(contract), str)
        is_contract = isinstance(self.contract2address.get(contract), str)
        return bool(is_address or is_contract)

    def get_contract(self,contract=None , web3=None, account=None, version=-1, pythonic=True):
        web3 = self.resolve_web3(web3)
        account = self.resolve_account(account)

        # assume theres an address
        address = contract
        contract_path = self.address2contract.get(address)
        if isinstance(contract_path, str):
            contract_path ,  contract_version = contract_path.split('-v')
            contract_version = int(contract_version)
            contract_address = address


        else:
            contract_path = contract
            contract_version_addresses = self.get_contract_address(contract, version)
            if len(contract_version_addresses) > 0:
                contract_address = contract_version_addresses[version]
            else:
                raise NotImplemented(contract_address)
      


        contract_artifact = self.get_artifact(contract_path)
        contract = web3.eth.contract(address=contract_address, abi=contract_artifact['abi'])
        if pythonic:
            contract = PythonicContractWrapper(contract=contract, account = self.account)
        
        return contract
        

    @property
    def address2contract(self):
        registered_contracts = self.registered_contracts
        address2contract = {}
        for network, contract_path_map in registered_contracts.items():
            for contract_path, contract_address_list in contract_path_map.items():
                for i, contract_address in enumerate(contract_address_list):
                    address2contract[contract_address] = contract_path+f'-v{i}'

        return address2contract


    @property
    def contract2address(self):
        return {v:k for k,v in self.address2contract.items()}

    def deployed_contracts(self):
        return list(self.contract2address.keys())
    def deployed_addresses(self):
        return list(self.contract2address.values())

    @property
    def address2network(self):
        registered_contracts = self.registered_contracts
        address2network = {}
        for network, contract_path_map in registered_contracts.items():
            for contract_path, contract_address_list in contract_path_map.items():
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
            network2contract[network] = [address2contract[address] for address in address_list]
        return network2contract
    

    def contract2network(self):
        address2contract = self.address2contract
        contract2network ={}
        for address, network in self.address2network.items():
            contract2network[address2contract[address]] = network

        return contract2network

    def register_contract(self, network:str,
                            contract_path:str , 
                            contract_address:str,  
                            refresh=True):


        registered_contracts = {} if refresh else self.registered_contracts
        if network not in registered_contracts:
            registered_contracts[network] = {}
        if contract_path not in registered_contracts[network]:
            registered_contracts[network][contract_path] = []

        assert isinstance(registered_contracts[network][contract_path], list)
        registered_contracts[network][contract_path].append(contract_address)
    
        self.put_json('registered_contracts', registered_contracts)

        return registered_contracts

    def resolve_network(self, network):
        if network == None:
            network = self.network
        return network
    
    
    def resolve_contract_path(self,  path):

        if path in self.contract_paths:
            contract_path = path
        else:
            contract_path = self.contract2path.get(path, None)
        assert contract_path in self.contract_paths
        return contract_path


        
    def __reduce__(self):
        deserializer = ContractManagerModule
        serialized_data = (self.config)
        return deserializer, serialized_data


    @classmethod
    def streamlit(cls):
        import ray
        st.write("## "+cls.__name__)
        ContractManagerModule.new_event_loop()

        self =  ContractManagerModule.deploy(actor=False, wrap=True)
        self.set_network('polygon.test')
        self.set_account('chris')

        st.write(self.network.network)
        contract = self.deploy_contract(contract='token.ERC20.ModelToken',new=True)
        st.write(contract)


if __name__ == '__main__':
    ContractManagerModule.run()

 