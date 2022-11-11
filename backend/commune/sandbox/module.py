
##################
##### Import #####
##################
import ray
import torch
import concurrent.futures
import time
import psutil
import random
import argparse
from tqdm import tqdm
import asyncio
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
import bittensor
import streamlit as st
import numpy as np
import sys
import pandas as pd

from commune.utils import chunk

##########################
##### Get args ###########
##########################
from commune.streamlit import StreamlitPlotModule, row_column_bundles

import threading
import time
import queue
from loguru import logger
from commune.dataset.huggingface.module import DatasetModule
st.write(DatasetModule)

from commune import Module


parser = argparse.ArgumentParser( 
    description=f"Bittensor Speed Test ",
    usage="python3 speed.py <command args>",
    add_help=True
)
bittensor.wallet.add_args(parser)
bittensor.logging.add_args(parser)
bittensor.subtensor.add_args(parser)
config = bittensor.config(parser = parser)

class Sandbox(Module):
    sample_example = {}
    def __init__(self, 
                subtensor=None,
                dataset=None, 
                tokenizer=None,
                wallet = None,
                config=None, 
                load=True,
                loop=None):
        Module.__init__(self, config=config)
        # self.loop = self.set_event_loop()
        self.sample_example = {}
        # config = bittensor.config()
        if load:
            self.subtensor = self.set_subtensor(subtensor)
            self.wallet = self.set_wallet(wallet)
            self.receptor_pool =self.set_receptor_pool(receptor_pool=None)
            self.dataset = self.set_dataset(dataset)
            # self.tokenizer = self.set_tokenizer(tokenizer)
            
        self.sync_the_async()
    
    def set_receptor_pool(self, receptor_pool=None, refresh=None, max_active_receptors=0):
        rp_config = self.config['receptor_pool']
        # if refresh:
        #     rp_config['actor'] =  rp_config.get('actor',{})
        #     rp_config['actor']['refresh'] = True
        rp_config['actor'] = rp_config.get('actor', False)
        rp_config['kwargs']['wallet']=self.wallet
        rp_config['kwargs']['max_active_receptors'] = max_active_receptors
        rp_config['kwargs']['compression'] = None

        if receptor_pool == None:
            receptor_pool = self.launch_module( **rp_config)  
        self.receptor_pool = receptor_pool

        return self.receptor_pool

    def set_dataset(self, dataset=None):
        if dataset==None:
            dataset = self.launch_module(**self.config['dataset'])
        
        self.dataset = dataset
        return self.dataset

    def set_wallet(self, wallet=None):
        wallet = wallet if wallet else self.config.get('wallet')
        if isinstance(wallet, dict):
            self.wallet = bittensor.wallet(**wallet)
        elif isinstance(wallet, bittensor.wallet):
            self.wallet = wallet
        else:
            raise NotImplemented(f'{type(wallet)} type of wallet is not available')
    
        return self.wallet

    def set_tokenizer(self, tokenizer=None):
        if tokenizer == None:
            tokenizer = bittensor.tokenizer()
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
    

    @property
    def endpoints(self):
        endpoints =self.graph.endpoint_objs
        return endpoints
    
    @property
    def uids(self):
        return list(map(lambda x: x.uid, self.endpoints))
        

    def get_random_endpoints(self, n = 10 ):
        endpoints =self.endpoints
        random_ids =  np.random.randint(0, len(endpoints), (n))
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

    synapses = all_synapses=available_synapses
    @staticmethod
    def errorcode2name(code):
        code2name_map =  {k:f'{v}' for k,v in zip(bittensor.proto.ReturnCode.values(),bittensor.proto.ReturnCode.keys())}
        return code2name_map[code]

        
    async def async_receptor_pool_forward(self, endpoints, inputs, synapses , timeout, splits=5):
        if synapses == None:
            synapses = self.synapses

        endpoints_split_list = self.chunk(endpoints, num_chunks=splits)

        kwargs_list = []

        for endpoints_split in endpoints_split_list:
            kwargs_list.append(dict(endpoints=endpoints_split, inputs=inputs, synapses=synapses , timeout=timeout))


        job_bundle = asyncio.gather(*[self.receptor_pool.async_forward(**kwargs) for kwargs in kwargs_list])
       
        agg_results = [[],[],[]]
        for results in (await job_bundle):
            for i,result in enumerate(results):
                agg_results[i].extend(result)
        # st.write(len(results[0]), len(results[1]),  len(results[2]))
        # st.write([(len(result), type(result)) for result in results])
        return agg_results


    def resolve_synapse(self, synapse:str, *args,**kwarga):
        return getattr(bittensor.synapse, synapse)()


    async def async_sample(self,
            sequence_length = 256,
            batch_size = 32,
            timeout= 2,
            synapse = 'TextCausalLMNext',
            num_endpoints = 50,
            success_only= True,
            return_type='results',
            return_json = True,
            splits=1, 
        ):
        # inputs = torch.zeros([batch_size, sequence_length], dtype=torch.int64)
        inputs = self.dataset.sample( batch_size=batch_size, sequence_length=sequence_length)
        synapse = self.resolve_synapse(synapse)
        endpoints = self.get_random_endpoints(num_endpoints)
        
        uids = torch.tensor([e.uid for e in endpoints])

        io_1 = psutil.net_io_counters()
        start_bytes_sent, start_bytes_recv = io_1.bytes_sent, io_1.bytes_recv

        with self.timer(text='Querying Endpoints: {t}', streamlit=True) as t:
            
            results = await self.async_receptor_pool_forward(
                                endpoints=endpoints,
                                synapses= [synapse],
                                timeout=timeout,
                                inputs= [inputs]*len(endpoints),
                                splits=splits)

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
        results['splits'] = splits
        # results['output_size'] = sys.getsizeof( results.pop['tensor'])
        results['batch_size'] = batch_size
        results['sequence_length'] = sequence_length
        results['num_tokens'] = batch_size*sequence_length
        if success_only and len(success_indices) == 0:
            return {}
        for is_success in [True, False]:
            for m in ['min', 'max', 'mean', 'std']:
                if is_success:
                    if len(success_indices)>0:
                        results[f'success_latency_{m}'] = getattr(torch, m)(results['latency'][success_indices]).item()
                    else:
                        results[f'success_latency_{m}'] = 0
                else:
                    results[f'latency_{m}'] = getattr(torch, m)(results['latency']).item()



        result_keys = ['tensor', 'latency', 'code', 'uid']

        # results['code'] = list(map(self.errorcode2name, results['code'].tolist()))



        graph_state_dict = self.graph.state_dict()
        graph_keys = ['trust', 'consensus','stake', 'incentive', 'dividends', 'emission']
        for k in graph_keys:
            results[k] =  graph_state_dict[k][results['uid']]
        

        if success_only:
            for k in result_keys + graph_keys:
                if len(success_indices)>0:
                    results[k] = results[k][success_indices]
                else:
                    results[k] = []

        if return_type in ['metric', 'metrics']:
            results = {k:v for k,v  in results.items() if k not in graph_keys+result_keys }

        elif return_type in ['results', 'result']:
            results = {k:v for k,v  in results.items()\
                             if k in (graph_keys+result_keys) }

            results['input'] = inputs


        else:
            raise Exception(f'{return_type} not supported')
        
        self.sample_example = results

        
        return results
    def process_results(self, results):
        results_dict = {'tensor':[], 'code':[], 'latency':[], 'uid': []}

        num_responses = len(results[0])
        for i in range(num_responses):
            tensor = results[0][i][0]
            code = results[1][i][0]
            latency = results[2][i][0]
            endpoint = results[3][i]

            results_dict['tensor'].append(tensor[...,:2])
            results_dict['code'].append(code)
            results_dict['latency'].append(latency)
            results_dict['uid'].append(endpoint)

        if len(results_dict['tensor'])>0:
            results_dict['tensor'] = torch.stack(results_dict['tensor'])
            results_dict['code'] = torch.tensor(results_dict['code'])
            results_dict['latency'] = torch.tensor(results_dict['latency'])
            results_dict['uid'] = torch.tensor(results_dict['uid'])
        else:
            results_dict['tensor'] = torch.tensor([])
            results_dict['code'] = torch.tensor([])
            results_dict['latency'] = torch.tensor([])
            results_dict['uid'] =  torch.tensor([])

        return results_dict


    # def run_experiment()



    def run_experiment(self,
            params = dict(
                sequence_length=[16,32,64, 128, 256 ],
                batch_size=[4,8,16,32, 64],
                num_endpoints=[32,64,128, 256],
                timeout=[4,8,12],
                synapse=['TextCausalLMNext'],
                splits=[1,2,4,8]
            ),
            experiment='experiment3',
            sequence_length=[]):

        # def flatten_hyperparams(hyperparams, flat_list =[]):
        #     for k,v_obj in hyperparams.items():
        #         tmp_params = deepcopy(hyperparams)
        #         if isinstance(v_obj, list):
        #             for v in v_obj:
        #                 tmp_params[k] = v
        #                 flat_list += flatten_hyperparams(hyperparams=tmp_params[k], flat_list=flat_list)
        #         else:
        #             continue

            
        sample_kwargs_list = []
        for sequence_length in params['sequence_length']:
            for num_endpoints in params['num_endpoints']:
                for timeout in params['timeout']:
                    for synapse in params['synapse']:
                        for batch_size in params['batch_size']:
                            for splits in params['splits']:
                                sample_kwargs_list += [dict(
                                    sequence_length = sequence_length,
                                    batch_size = batch_size,
                                    timeout= timeout,
                                    synapse = synapse,
                                    num_endpoints = num_endpoints,
                                    success_only= False,
                                    return_type='metric',
                                    splits=splits
                                )]
        random.shuffle(sample_kwargs_list)
        for i,sample_kwargs in enumerate(tqdm(sample_kwargs_list)):
            self.set_receptor_pool(refresh=True)         
            trial_metrics_result = self.sample(**sample_kwargs)
            self.put_json(f'{experiment}/{i}', trial_metrics_result)
  
    # def streamlit(self):
    #     for k,v_list in params.items():
    def streamlit(self):
        st.write(self.load_experiment('experiment3'))


    def load_experiment(self, path='experiment3'):
        df = []
        
        for p in self.glob_json(path+'/*'):
            df.append(self.client.local.get_json(p))

        df =  pd.DataFrame(df)

        # df = pd.concat(df)
        # returnid2code = {k:f'{v}' for k,v in zip(bittensor.proto.ReturnCode.values(),bittensor.proto.ReturnCode.keys())}
        # df['code'] = df['code'].map(returnid2code)
        return df

    def streamlit_experiment(self, experiment= 'experiment3'):
        df = self.load_experiment(path=experiment)
        from commune.streamlit import StreamlitPlotModule, row_column_bundles

        st.write(df)

        df['tokens_per_second'] = df['num_tokens']*df['num_successes'] / df['elapsed_time']
        df['samples_per_second'] = df['batch_size']*df['num_successes'] / df['elapsed_time']
    
        StreamlitPlotModule().run(df)



    @staticmethod
    def chunk(sequence,
            chunk_size=None,
            append_remainder=False,
            distribute_remainder=True,
            num_chunks= None):
        # Chunks of 1000 documents at a time.

        if chunk_size is None:
            assert (type(num_chunks) == int)
            chunk_size = len(sequence) // num_chunks

        if chunk_size >= len(sequence):
            return [sequence]
        remainder_chunk_len = len(sequence) % chunk_size
        remainder_chunk = sequence[:remainder_chunk_len]
        sequence = sequence[remainder_chunk_len:]
        sequence_chunks = [sequence[j:j + chunk_size] for j in range(0, len(sequence), chunk_size)]

        if append_remainder:
            # append the remainder to the sequence
            sequence_chunks.append(remainder_chunk)
        else:
            if distribute_remainder:
                # distributes teh remainder round robin to each of the chunks
                for i, remainder_val in enumerate(remainder_chunk):
                    chunk_idx = i % len(sequence_chunks)
                    sequence_chunks[chunk_idx].append(remainder_val)

        return sequence_chunks

    def schema(self):
          return {k:v.shape for k,v in self.sample_example.items()}

    @classmethod
    def sync_the_async(cls, obj = None):
        if obj == None:
            obj = cls

        for f in dir(obj):
            if 'async_' in f:
                setattr(obj, f.replace('async_',  ''), cls.sync_wrapper(getattr(obj, f)))


    @staticmethod
    def sync_wrapper(fn):
        def wrapper_fn(*args, **kwargs):
            return asyncio.run(fn(*args, **kwargs))
        return  wrapper_fn

from munch import Munch 
class AyncioManager:
    """ Base threadpool executor with a priority queue 
    """

    def __init__(self,  max_tasks:int=10):
        """Initializes a new ThreadPoolExecutor instance.
        Args:
            max_threads: 
                The maximum number of threads that can be used to
                execute the given calls.
        """
        self.max_tasks = max_tasks
        self.running, self.stopped = False, False
        self.tasks = []
        self.queue = Munch({'in':queue.Queue(), 'out':queue.Queue()})
        self.start()
        st.write(self.background_thread)

    def stop(self):
        while self.running:
            self.stopped = True
        return self.stopped
        
    def start(self):
        self.background_thread = threading.Thread(target=self.run_loop, args={}, kwargs={}, daemon=True)
        self.background_thread.start()

    def run_loop(self):
        return asyncio.run(self.async_run_loop())
    def new_aysnc_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop
    async def async_run_loop(self): 
        loop = self.new_aysnc_loop()
        print(loop)
        self.stopped = False
        self.running = True
        print(loop)

        while self.running and not self.stopped:
            finished_tasks = []
            if len(self.tasks)>0:
                finished_tasks, self.tasks = await asyncio.wait(self.tasks)
            for finished_task in finished_tasks:
                self.queue.out.put(await asyncio.gather(*finished_task))
            if len(self.tasks) <= self.max_tasks:
                new_job = self.queue['in'].get()
                self.submit(**new_job)
                new_job = self.queue.out.get()

        loop.close()
        self.running = False

    def submit(self,fn, *args, **kwargs):
        job = {'fn': fn, 'args': args, 'kwargs': kwargs}
        self.queue['in'].put(job)

    def get(self):
        return self.queue['out'].get()

    def close(self):
        self.stop()
        self.background_thread.join()

    def __del__(self):
        self.close()


if __name__ == '__main__':
    Sandbox.ray_start()
    module = Sandbox.deploy(actor=False, wrap=True)

    async def async_run_jobs(jobs, max_tasks=5, stagger_time=0.5):
        finished_jobs, running_jobs = [],[]
        finished_results = []
        for job in jobs:
            while len(running_jobs)>=max_tasks:
                
                tmp_finished_jobs, running_jobs = await asyncio.wait(running_jobs, return_when=asyncio.FIRST_COMPLETED)
                running_jobs = list(running_jobs)
                if tmp_finished_jobs:
                    finished_jobs += list(tmp_finished_jobs)
                    finished_results += await asyncio.gather(*tmp_finished_jobs)
    
                    st.write(len(finished_results[-1].get('tensor',[])))
    

                else:
                    asyncio.sleep(stagger_time)
                
            
            running_jobs.append(job['fn'](*job.get('args', []),**job.get('kwargs',{})))
            

        finished_results += list(await asyncio.gather(*running_jobs))
        
        return finished_results

    jobs = [{'fn': module.async_sample, 'kwargs': dict(num_endpoints=10, timeout=6, sequence_length=10, batch_size=10) } for i in range(50)]


    with Sandbox.timer() as t:

        results = asyncio.run(async_run_jobs(jobs, max_tasks=10))
        st.write('QPS: ',len(results)/t.seconds)

