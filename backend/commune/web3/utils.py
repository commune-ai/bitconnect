from web3 import Web3

@staticmethod
def hash(input, hash_type='keccak',return_type='str',*args,**kwargs):
    
    hash_fn = AccountModule.resolve_hash_function(hash_type)

    input = AccountModule.python2str(input)
    hash_output = Web3.keccak(text=input, *args, **kwargs)
    if return_type in ['str', str, 'string']:
        hash_output = Web3.toHex(hash_output)
    elif return_type in ['hex', 'hexbytes']:
        pass
    else:
        raise NotImplementedError(return_type)
    
    return hash_output

