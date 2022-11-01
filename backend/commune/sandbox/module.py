##################
##### Import #####
##################
import torch
import concurrent.futures
import time
import psutil
import random
import argparse
from tqdm import tqdm
import bittensor
import streamlit as st
import numpy as np
import sys

##########################
##### Get args ###########
##########################
parser = argparse.ArgumentParser( 
    description=f"Bittensor Speed Test ",
    usage="python3 speed.py <command args>",
    add_help=True
)
bittensor.wallet.add_args(parser)
bittensor.logging.add_args(parser)
bittensor.subtensor.add_args(parser)
config = bittensor.config(parser = parser)
config.wallet.name = 'const'
config.wallet.hotkey = 'Tiberius'
##########################
##### Setup objects ######
##########################
# Sync graph and load power wallet.


from commune import Module
class Sandbox(Module):
    def __init__(self, 
                subtensor=None,
                dataset=None, 
                tokenizer=None,
                wallet = None,
                config=None):
        Module.__init__(self, config=config)

        # config = bittensor.config()
        # st.write(config)
        self.subtensor = self.set_subtensor(subtensor)
        st.write(self.subtensor.network, 'network')
        self.wallet = self.set_wallet(wallet)
        self.dataset = self.set_dataset(dataset)
        self.tokenizer = self.set_tokenizer(tokenizer)

        self.receptor_pool = bittensor.receptor_pool(wallet=self.wallet)


    def set_dataset(self, dataset=None):
        if dataset==None:
            dataset = self.launch_module(**self.config['dataset'])
        self.dataset = dataset
        return self.dataset



    def set_wallet(self, wallet=None):
        if wallet == None:
            wallet = bittensor.wallet(**self.config.get('wallet'))

        self.wallet = wallet
        return self.wallet


    def set_tokenizer(self, tokenizer=None):
        if tokenizer == None:
            tokenizer = self.dataset.tokenizer
        self.tokenizer = tokenizer
        return tokenizer
    
    def set_subtensor(self, subtensor=None):
        if subtensor == None:
            subtensor = bittensor.subtensor( config = config )
            graph = bittensor.metagraph( subtensor = subtensor )
            graph.load()
            self.subtensor = subtensor
            self.graph = graph
        if self.sync_delay > self.config.get('delay_threshold', 100):
            self.graph.sync()
            self.graph.save()
        
        
        
        return self.subtensor
    
    
    @property
    def current_block(self):
        return self.subtensor.block
    
    @property
    def synced_block(self): 
        return self.graph.block.item()

    @property
    def sync_delay(self):
        return self.current_block - self.synced_block
    

    def get_receptors(self, n = 10,uids=None):
        if uids == None:
            uids = list(range(n))
        
        receptors = []
        for uid in uids:
            receptors += [bittensor.receptor( wallet = self.wallet, endpoint = self.graph.endpoint_objs[uid])]
        return receptors
    

    def get_random_endpoints(self, n = 10 ):
        endpoints =self.graph.endpoint_objs
        random_ids =  list(np.random.randint(0, len(endpoints), (n)))
        return [endpoints[i] for i in random_ids]

    def get_endpoints(self, n=10, uids:list=[]):

        if len(uids) == 0:
            uids = list(range(n))
        endpoints =self.graph.endpoint_objs
        selected_endpoints = []
        for uid in uids:
            selected_endpoints += [endpoints[uid]]

        return selected_endpoints


    def tokenize(self, text:str, dtype = torch.int64, device='cpu'):
        # must be a string, or a list of strings
        if isinstance(text, str):
            text = [text]
        assert all(isinstance(t, str) for t in text)
        token_ids =  self.tokenizer(text)['input_ids']
        token_ids = torch.Tensor(token_ids).type(dtype).to(device)
        return token_ids

    @staticmethod
    def str2synapse(synapse:str, *args, **kwargs):
        return getattr(bittensor.synapse, synapse)(*args, **kwargs)
    @property
    def available_synapses(self):
        return [f for f in dir(bittensor.synapse) if f.startswith('Text')]


    @staticmethod
    def errorcode2name(code):
        code2name_map =  {k:f'{v}' for k,v in zip(bittensor.proto.ReturnCode.values(),bittensor.proto.ReturnCode.keys())}
        return code2name_map[code]

    def streamlit(self,
            sequence_length = 10,
            batch_size = 2,
            timeout= 4,
            synapse = 'TextLastHiddenState',
            num_endpoints = 200,
            success_only= True
        ):

        # inputs = torch.zeros([batch_size, sequence_length], dtype=torch.int64)
        inputs = self.dataset.sample( batch_size=batch_size, sequence_length=sequence_length)
        # st.write(inputs)

        synapse = getattr(bittensor.synapse, synapse)()
        endpoints = self.get_random_endpoints(num_endpoints)
        
        # st.write([e.uid for e in endpoints])
        # st.write(self.receptor_pool.wallet)
        # st.write(receptor_kwargs)

        io_1 = psutil.net_io_counters()
        start_bytes_sent, start_bytes_recv = io_1.bytes_sent, io_1.bytes_recv

        with self.timer(text='Querying Endpoints: {t}', streamlit=True) as t:
            results = self.receptor_pool.forward(
                                endpoints=endpoints,
                                synapses= [synapse],
                                timeout=timeout,
                                inputs= [inputs]*len(endpoints),
                                # return_type='dict',
                                # graph=self.graph
                            )
            elapsed_time = t.elapsed_time.total_seconds() 

        io_2 = psutil.net_io_counters()
        total_bytes_sent, total_bytes_recved = io_2.bytes_sent - start_bytes_sent, io_2.bytes_recv - start_bytes_recv

        results = list(results) + [list(map(lambda e:e.uid, endpoints))]
        results = self.process_results(results)
        # tensors =
        
        
        success_indices = torch.argwhere(results['code']==1).squeeze(1).tolist()
        results['elapsed_time'] = elapsed_time
        results['timeout'] = timeout
        results['num_successes'] = len(success_indices)

        results['successes_per_second'] = results['num_successes']/results['elapsed_time'] 
        results['time_over_timeout'] = elapsed_time - timeout
        results['time_over_timeout_ratio'] = (elapsed_time - timeout)/(timeout + 1e-10)
        results['upload_bytes_mb'] =total_bytes_sent / 1000
        results['download_bytes_mb'] =total_bytes_recved / 1000
        results['upload_rate_mb'] =results['upload_bytes_mb']/elapsed_time 
        results['download_rate_mb'] =results['download_bytes_mb']/elapsed_time
        results['num_endpoints'] = num_endpoints
        results['success_rate'] = results['num_successes']/results['num_endpoints']

        # results['output_size'] = sys.getsizeof( results.pop['tensor'])
        results['batch_size'] = batch_size
        results['sequence_length'] = sequence_length
        results['num_tokens'] = batch_size*sequence_length

        for is_success in [True, False]:
            for m in ['min', 'max', 'mean', 'std']:
                if is_success:
                    results[f'success_latency_{m}'] = getattr(torch, m)(results['latency'][success_indices]).item()
                else:
                    results[f'latency_{m}'] = getattr(torch, m)(results['latency']).item()


        if success_only:
            for k in ['tensor', 'latency', 'code']:
                results[k] = results[k][success_indices]


        results['code'] = list(map(self.errorcode2name, results['code'].tolist()))

        st.write(results)

        # df['code'] = df['code'].map(returnid2code)

        
        # if return_type in ['metrics', 'metric']:
        #     metric_dict = {}
        #     metric_dict['success_count'] = int(df['code'].apply(lambda x: x == 'Success').sum())
        #     metric_dict['success_rate'] = df['code'].apply(lambda x: x == 'Success').mean()
        #     metric_dict['num_endpoints'] = num_endpoints
        #     metric_dict['timeout'] = int(df['timeout'].iloc[0])
        #     metric_dict['latency'] = df['latency'].iloc[0]
        #     metric_dict['input_length'] = df['input_length'].iloc[0]
        #     metric_dict['elapsed_time'] = elasped_time.total_seconds()
        #     metric_dict['samples_per_second'] = metric_dict['success_count'] / metric_dict['elapsed_time']
        #     metric_dict['splits'] = splits
        #     metric_dict['min_success'] = min_success
        #     metric_dict['num_responses'] = num_responses

        # for k in ['trust', 'consensus','stake', 'incentive', 'dividends', 'emission', 'latency']:
        #     # for mode in ['mean', 'std', 'max', 'min']:
        #     metric_dict[k] =  getattr(df[k], 'mean')()

        # metric_dict = {k:float(v)for k,v in metric_dict.items()}
        # return metric_dict
        # else:
        #     return df

    def process_results(self, results, ):
        results_dict = {'tensor':[], 'code':[], 'latency':[], 'endpoint': []}

        num_responses = len(results[0])
        for i in range(num_responses):
            tensor = results[0][i][0]
            code = results[1][i][0]
            latency = results[2][i][0]
            endpoint = results[3][i]

            results_dict['tensor'].append(tensor)
            results_dict['code'].append(code)
            results_dict['latency'].append(latency)
            results_dict['endpoint'].append(endpoint)

        if len(results_dict['tensor'])>0:
            results_dict['tensor'] = torch.stack(results_dict['tensor'])
            results_dict['code'] = torch.tensor(results_dict['code'])
            results_dict['latency'] = torch.tensor(results_dict['latency'])
            results_dict['endpoint'] = torch.tensor(results_dict['endpoint'])
        else:
            results_dict['tensor'] = torch.tensor([])
            results_dict['code'] = torch.tensor([])
            results_dict['latency'] = torch.tensor([])
            results_dict['endpoint'] =  torch.tensor([])

        return results_dict




if __name__ == '__main__':
    Sandbox.deploy(actor=False).streamlit()