from functools import partial
import web3 
import streamlit as st

class PythonicContractWrapper:
    def __init__(self, contract, account=None):

        for k,v in contract.__dict__.items():
            setattr(self, k, v)
        self.set_account(account)
        # st.write(self.account)
        self.parse()
    
    def set_account(self, account=None):
        if account != None:
            self.account = account
            self.web3 = self.account.web3


    def parse(self):
        for fn_dict in self.functions._functions:
            fn_name = fn_dict['name']
            
            if fn_dict['stateMutability'] == 'view':
                def wrapped_fn(self,fn_name, *args, tx={}, **kwargs):
                    fn  = getattr(self.functions, fn_name)
                    return fn(*args, **kwargs).call()
            elif fn_dict['stateMutability'] in ['payable', 'nonpayable']:
                def wrapped_fn(self,fn_name, *args,tx={}, **kwargs):
                    st.write(args, kwargs, 'FACK')
                    value = tx.pop('value', 0)
                    fn  = getattr(self.functions, fn_name)
                    return self.account.send_contract_tx(fn(*args, **kwargs), value=value)

            else:
                raise NotImplementedError(fn_name)
            
            wrapped_fn_ = partial(wrapped_fn, self, fn_name)
            setattr(self,fn_name, wrapped_fn_)


    @property
    def function_schema(self):
        function_schema = {}
        for fn_abi in self.functions.abi:
            if fn_abi['type'] == 'constructor':
                name = 'constructor'
            elif fn_abi['type'] == 'function':
                name = fn_abi.pop('name')
            else:
                continue

            function_schema[name] = fn_abi
            
            for m in ['inputs', 'outputs']:
                if m in function_schema[name]:
                    function_schema[name][m] =  [{k:i[k] for k in ['type', 'name']} for i in function_schema[name][m]]

        return function_schema

    function_abi = function_schema

    @property
    def function_names(self):
        return list(self.function_schema.keys())


            
