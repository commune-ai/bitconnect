import datetime
import streamlit as st
def isoformat2datetime(isoformat:str):
    dt, _, us = isoformat.partition(".")
    dt = datetime.datetime.strptime(dt, "%Y-%m-%dT%H:%M:%S")
    us = int(us.rstrip("Z"), 10)
    dt = dt + datetime.timedelta(microseconds=us)
    assert isinstance(dt, datetime.datetime)
    return dt

def isoformat2timestamp(isoformat:str, return_type='int'):
    supported_types = ['int', 'float']
    assert return_type in supported_types, f'return type should in {supported_types} but you put {return_type}'
    dt = isoformat2datetime(isoformat)
    timestamp = eval(return_type)(dt.timestamp())
    assert isinstance(timestamp, int)
    return timestamp


def timedeltatimestamp( **kwargs):
    assert len(kwargs) == 1
    supported_modes = ['hours', 'seconds', 'minutes', 'days']
    mode = list(kwargs.keys())[0]
    assert mode in supported_modes, f'return type should in {supported_modes} but you put {mode}'
    
    current_timestamp = datetime.datetime.utcnow()
    timetamp_delta  =  current_timestamp.timestamp() -  ( current_timestamp- datetime.timedelta(**kwargs)).timestamp()
    return timetamp_delta



import time


class Timer:
    
    def __init__(self, text='time elapsed: {}', return_type='seconds', streamlit=False, ):   
        
        self.__dict__.update(locals())


    def __enter__(self):
        self.start = datetime.datetime.utcnow()
        return self

    def __exit__(self, *args):
        self.end =  datetime.datetime.utcnow()
        interval = (self.end - self.start)

        return_type = self.return_type
        if return_type in ['microseconds', 'ms', 'micro', 'microsecond']:
            div_factor = 1
        elif return_type in ['seconds', 's' , 'second', 'sec']:
            div_factor = 1000
        
        elif return_type in ['minutes', 'm', 'min' , 'minutes']: 
            div_factor = 1000*60
        
        else:
            raise NotImplementedError
        
        self.elapsed_time = self.interval =  interval 


        if self.streamlit and self.text:
            st.write(self.text.format(t=self.elapsed_time))
        else: 
            print(self.text.format(t=self.elapsed_time))

        return self.interval
