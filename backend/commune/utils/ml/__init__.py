import torch


def tensor_info_dict(tensor, 
                    fn_dict):
    
    return {k:v(tensor) for k,v in fn_dict.items()}
def tensor_dict_check(tensor_dict, 
                      fn_dict={'std':( lambda x: torch.std(x)), 
                             'mean': (lambda x: torch.mean(x))}):
    
    '''
    apply function dict fn_dict to each of the tensor_dict elements
    '''
    out_dict = {}
    for k,v in tensor_dict.items():
        if isinstance(v,torch.Tensor):
            out_dict[k] = tensor_info_dict(tensor=v,fn_dict=fn_dict)
        else:
            out_dict[k] = None
    
    return out_dict
            