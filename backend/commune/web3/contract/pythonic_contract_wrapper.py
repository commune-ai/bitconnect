from functools import partial
import web3 

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
                def wrapped_fn(self,fn_name, stateMutability, *args, **kwargs):
                    fn  = getattr(self.functions, fn_name)
                    return fn(*args, **kwargs).call()
            elif fn_dict['stateMutability'] == 'nonpayable':
                def wrapped_fn(self,fn_name, value=None, *args, **kwargs):
                    fn  = getattr(self.functions, fn_name)
                    return self.account.send_contract_tx(fn=fn(*args, **kwargs, value=value))
            elif fn_dict['stateMutability'] == 'payable':
                def wrapped_fn(self,fn_name, value=0, *args, **kwargs):
                    fn  = getattr(self.functions, fn_name)
                    return self.account.send_contract_tx(fn(*args, **kwargs))

            else:
                raise NotImplementedError(fn_name)
            
            wrapped_fn_ = partial(wrapped_fn, self, fn_name)
            setattr(self,fn_name, wrapped_fn_)
