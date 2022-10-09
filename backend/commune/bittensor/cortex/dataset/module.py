import streamlit as st
from random import shuffle, seed
from collections import defaultdict
import pandas as pd
import bittensor
import torch
from torch import nn
from tqdm.auto import tqdm
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
# import torchsort
from commune.bittensor import BitModule
from commune import BaseModule
import ray
from commune.bittensor.cortex.metric import causal_lm_loss, ranknet_loss
from commune.utils import *
from sklearn import metrics
from scipy.stats import kendalltau


import torch
from torch import nn

from commune.ray.actor_pool import ActorPool


class DatasetModule(BitModule):
    __file__ = __file__
    default_config_path = 'bittensor.cortex.dataset'
    def __init__(self, config=None, **kwargs):
        load = kwargs.pop('load', True)
        BitModule.__init__(self, config=config,  **kwargs)

        if type(load) in [dict]:
            self.load(**load)
        else:
            self.load(load)


    def load_env(self, path=None, **kwargs):
        return BitModule.load(self=self, path=path, **kwargs)

    def load(self, keys=True, load_kwargs={}, load_args={}, **kwargs):
        

        if keys in [False, None]:
            return

        if keys == True:
            keys = ['env', 'dataset', 'tokenizer', 'receptor_pool']
        
        load_keys = keys

        for load_key in load_keys:
            st.write(load_key, 'load')
            load_kwargs.get(load_key, {}) 
            load_fn = getattr(self, f'load_{load_key}', None)
            assert load_fn != None, f'{load_key} is suppose to be a function'
            load_fn_kwargs = load_kwargs.get(load_key, self.config.get(load_key, {}))
            load_fn(**load_fn_kwargs)


    def load_receptor_pool(self, path='commune.bittensor.receptor.pool.module.ReceptorPoolModule', 
                        actor=False, 
                        params={ 'max_worker_threads': 0 , 'max_active_receptors': 0}, 
                        replicas=None, **kwargs):

        if hasattr(self, 'receptor_pool'):
            del self.receptor_pool
        receptor_pool_module = self.import_object(path)
        if isinstance(replicas, int):
            raise NotImplementedError
        else:
            self.receptor_pool = receptor_pool_module.deploy(actor=actor,
                                                wallet=self.wallet,**kwargs) 


    def load_dataset(self, path, params, **kwargs):
        dataset_class = self.import_object(path)
        self.dataset = dataset_class(**params)

    def load_tokenizer(self, 
                        path='bittensor.tokenizer', 
                        params=dict(version=bittensor.__version__), **kwargs): 
        tokenizer_class = self.import_object(path)
        self.tokenizer = tokenizer_class(**params)

    @property
    def device(self):
        device = self.config.get('device', 'cpu')
        if 'cuda' in device:
            assert torch.cuda.is_available()
        return device

    @device.setter
    def device(self, device):
        if 'cuda' in device:
            assert torch.cuda.is_available()
        self.config['device'] = device
        return device

    def to(self, device):
        self.device = device
        return self.device

    default_receptor_path = 'bittensor.receptor.pool.module.ReceptorPoolModule'


    def tokenize(self, text, padding=True, *args, **kwargs):
        device = kwargs.pop('device', self.device)
        return torch.tensor(self.tokenizer(text=text, padding=padding)['input_ids']).to(device)



    @property
    def num_endpoints(self):
        return self.config.get('num_endpoints')

    def get_endpoints(self, endpoint_ids=None , num_endpoints=None, random_sample=True):
        endpoints =self.metagraph.endpoint_objs
        selected_endpoints = []
        if isinstance(endpoint_ids, list ):
            for i in endpoint_ids:
                assert isinstance(i, int), i
                assert i > 0 and i < len(endpoints), endpoint_ids

                selected_endpoints.append(endpoints[i])

            return selected_endpoints
        
        if num_endpoints == None:
            num_endpoints =self.num_endpoints

        if random_sample == True:
            endpoint_index_list = list(np.random.randint(0, self.n, (num_endpoints)))
            selected_endpoints = [endpoints[idx] for idx in endpoint_index_list]
        else:
            selected_endpoints = endpoints[:num_endpoints]
        return selected_endpoints

    @staticmethod
    def str2synapse(synapse:str):
        return getattr(bittensor.synapse, synapse)()
    
    @property
    def synapses(self):
        synsapses = list(map(self.str2synapse, self.config.get('synapses',self.available_synapses[0])) )
        return synsapses


    @property
    def synapse(self):
        synsapses = list(map(self.str2synapse, self.config.get('synapses',self.available_synapses[0])) )
        return synsapses

    def receptor_pool_forward(self, endpoints, inputs, synapses=None , timeout=1, splits=5, min_success=1, split_type='endpoints'):


  
        endpoints_split_list = chunk(endpoints, num_chunks=splits)
        kwargs_list = []

        for endpoints_split in endpoints_split_list:

            kwargs_list.append(dict(endpoints=endpoints_split, inputs=[inputs]*len(endpoints_split), synapses=synapses , timeout=timeout, min_success=min_success))

        agg_results = [[],[],[]]
        results_generator = self.receptor_pool.map_unordered(lambda a,v: a.forward.remote(**v), kwargs_list)
       
        for results in results_generator:
            for i,result in enumerate(results):
                agg_results[i].extend(result)


        # st.write(len(results[0]), len(results[1]),  len(results[2]))
        # st.write([(len(result), type(result)) for result in results])
        return agg_results
            

    def get_query_metrics(self, results, return_type='df'):

        results = self.add_metrics(results)
        if return_type in ['df']:
            return pd.DataFrame(results)
        elif return_type in ['results']:
            return df
        elif return_type in ['metrics', 'metric']:
            metric_dict = {}
            metric_dict['success_count'] = df['code'].apply(lambda x: x == 'Success').sum()
            metric_dict['success_rate'] = df['code'].apply(lambda x: x == 'Success').mean()
            metric_dict['num_endpoints'] = df['num_endpoints'].iloc[0]
            metric_dict['timeout'] = df['timeout'].iloc[0]
            metric_dict['latency'] = df['latency'].iloc[0]
            metric_dict['input_token_length'] = df['input_length'].iloc[0]
            # metric_dict['elapsed_time'] = elasped_time.total_seconds()
            # metric_dict['queries_per_second'] = metric_dict['success_count'] / metric_dict['elapsed_time']
            
            for k in ['trust', 'consensus','stake', 'incentive', 'dividends', 'emission', 'latency']:
                # for mode in ['mean', 'std', 'max', 'min']:
                metric_dict[k] =  getattr(df[k], 'mean')()

            metric_dict = {k:float(v)for k,v in metric_dict.items()}
            return metric_dict

        else:
            assert NotImplementedError




    @property
    def my_endpoints(self, mode = 'hotkey'):
        endpoints = self.metagraph.endpoint_objs
        
        if mode == 'hotkey':
            endpoints = [e for e in endpoints if (e.hotkey == self.hotkey_address and e.ip != "0.0.0.0") ]
        elif mode == 'coldkey':
            endpoints = [e for e in endpoints if (e.coldkey == self.coldkey_address and e.ip != "0.0.0.0") ]
        else:
            raise NotImplementedError

        return endpoints

    def sample_raw(self, idx=None, tokenize=False):
        text_field = self.config['dataset']['text_field']

        dataset_length = len(self.dataset)
 
        if idx == None:
            idx = random.randint(1,dataset_length)

        assert idx <= dataset_length, f'{idx}<={dataset_length} '
        
        
        sample = self.dataset[idx][text_field]
        if tokenize == True:
            sample = self.tokenize(text=sample, padding=True)
        
        return sample

    def sample_raw_batch(self, batch_size=1, **kwargs):
        return [self.sample_raw(idx=i, **kwargs) for i in range(batch_size)]

    @property
    def available_synapses(self):
        return [f for f in dir(bittensor.synapse) if f.startswith('Text')]

    ls_synapses = all_synapses = available_synapses
    
    @property
    def synapse_map(self):
        return {f:getattr(bittensor.synapse,f) for f in self.available_synapses}


    def get_synapse(self, synapse=None, *args, **kwargs):
        synapse = deepcopy(synapse)
        if synapse == None:
            synapse = self.synapses[0]
        if isinstance(synapse, str):
            synapse = self.synapse_map[synapse]

        return synapse(*args,**kwargs)

    resolve_synapse = get_synapse

    def resolve_device(self, device=None):
        if device == None:
            device = self.device
        
        return device


    def resolve_num_endpoints(self, num_endpoints=None):
        if num_endpoints == None:
            num_endpoints = self.num_endpoints
        return num_endpoints



    @property
    def default_queue_map(self):
        return {i: f'{self.module_path}{i}' for i in ['in','out']}
    @property
    def default_queues(self):
        return list(self.default_queue_map.keys())
        
    def resolve_topic(self):

        return self.module_path

    def stop_sample_loop(self, topic):
        return self.running_loop_dict.pop(topic, None)

    running_loop_dict = {}
    def start_sample_loop(self, topic, 
                        cache_limit=100,cache_fraction=0.5, refresh_cache=False, 
                        refresh_queue=True, condition_fn=None, 
                        max_queue_size = 200,
                        min_results = 10,
                        *args, **kwargs): 
        if topic == None:
            topic  = self.default_queue_map['out']
        assert topic not in self.running_loop_dict
        self.running_loop_dict[topic] = True

        if refresh_cache:
             self.rm_json(f'samples/{topic}')
        if refresh_queue:
            try:
                ray.get(self.queue.rm.remote(topic))
            except KeyError:
                pass
        cnt = 0
        while True:
            if topic in self.running_loop_dict:
                pass
            else:
                break
            
            sample_cache_count = self.sample_cache_count(topic)
            get_cache = random.uniform(0,1)<cache_fraction and sample_cache_count > 0
            if  get_cache:
                cache_idx = random.randint(0, sample_cache_count-1)
                results = self.get_json(f'samples/{topic}/{cache_idx}')
                for i in range(len(results)):
                    results[i]['tensor'] = [torch.tensor(r) for r in results[i]['tensor'] ]
            else:
                cache_idx = sample_cache_count
                with Timer('Sample Time: {t}',streamlit=True):
                    results = self.sample(*deepcopy(args), **deepcopy(kwargs))


            st.write(len(results))

            if len(results)>min_results:

                self.queue.put.remote(topic, results )
                # st.write(ray.get(self.queue.size.remote(topic)))
                with Timer('Save Time 1: {t}',streamlit=True):
                    for i in range(len(results)):
                        results[i]['tensor'] = [r.tolist() for r in results[i]['tensor'] ]
                # with Timer('Save Time 2: {t}',streamlit=True):
                #     self.put_json(f'samples/{topic}/{cache_idx}', results)

    
    sample_dict = {}
    running_generator_dict = {}
    def start_generator(self,
            num_endpoints=10, 
            timeout=1, 
            synapse = 'TextCausalLM', 
            splits=1, 
            batch_size = 6,
            device = None,
            success_only=True,
            min_success=0.5,
            num_batches=10,
            topic='train',
             **kwargs):

        num_endpoints = self.resolve_num_endpoints(num_endpoints)
        st.write('num_endpoints', num_endpoints)
        device = self.resolve_device(device)
        if self.running_generator_dict.get(topic) == True:
            return
        else:
            self.running_generator_dict[topic] = True

        if isinstance(synapse,list):
            synapses = synapse
        else:
            synapses = [synapse]

        synapses = [self.resolve_synapse(s) for s in synapses]

        text_inputs = kwargs.get('text')

        text_inputs = self.sample_raw_batch(batch_size=batch_size, tokenize=False)

        call_kwargs_list = []

        for i in range(num_batches):
            endpoints = self.get_endpoints(num_endpoints=num_endpoints)
            inputs = self.tokenize(text_inputs, padding=True)
            call_kwargs = dict(endpoints=endpoints, 
                                inputs=[inputs]*len(endpoints),
                                synapses=synapses , 
                                timeout=timeout,
                                min_success=min_success)
            call_kwargs_list.append(call_kwargs)
            

        results_generator = self.receptor_pool.map_unordered(lambda a,v: a.forward.remote(**v), call_kwargs_list)
        self.sample_dict[topic] = results_generator


    def sample_generator(self, topic='train'):
        return next(self.sample_dict[topic])

    def sample_cache_count(self, topic):
        return len(self.sample_cache_files(topic))

    def sample_cache_files(self, topic):
        return self.ls_json(f'samples/{topic}')

    def sample_cache_exists(self, topic, idx):
        return self.exists_json(f'samples/{topic}/{idx}')


    def running_loops(self):
        return list(self.running_loop_dict.keys())
    def running_loops_queues(self):
        return {t: ray.get(self.queue.get_queue.remote(t)) for t in running_loops}
    def sample_loop(self, topic, batchsize=1, *args, **kwargs):
        return ray.get(self.queue.get_batch.remote(topic=topic, num_items=batchsize))




    def put(self, key, value):
        return ray.get(self.queue.put.remote(key,value))

    def put_batch(self, key, values, sync=True):
        assert isinstance(values, list), f'{type(values)}'
        jobs = [self.queue.put.remote(key, value) for value in values]
        if sync:
            return ray.get(jobs)
        else:
            return jobs

    def get(self, key):
        return ray.get(self.queue.get.remote(key))

    def get_batch(self, key, batch_size=1, sync=True):
        jobs = [self.queue.get.remote(key) for i in range(batch_size)]
        if sync:
            return ray.get(jobs)
        else :
            return jobs
    
    def sample(self, 
            num_endpoints=None, 
            timeout=1, 
            synapse = 'TextCausalLM', 
            splits=1, 
            batch_size = 10,
            device = None,
            success_only=True,
            min_success=0.9,
            queue_topic = None,
             **kwargs):



        num_endpoints = self.resolve_num_endpoints(num_endpoints)
        st.write('num_endpoints', num_endpoints)
        device = self.resolve_device(device)

        if isinstance(synapse,list):
            synapses = synapse
        else:
            synapses = [synapse]

        synapses = [self.resolve_synapse(s) for s in synapses]


        text_inputs = kwargs.get('text')
        if isinstance(text_inputs, list):
            assert all([isintance(s) for s in text_inputs])
        elif isinstance(text_inputs, str):
            text_inputs = [text_inputs]
        else: 
            text_inputs = self.sample_raw_batch(batch_size=batch_size, tokenize=False)
        inputs = self.tokenize(text_inputs, padding=True)
        endpoints = self.get_endpoints(num_endpoints=num_endpoints)

        elapsed_time = -1
        with self.timer() as t:
            if isinstance(self.receptor_pool, ray.actor.ActorHandle):
                results = ray.get(self.receptor_pool.forward.remote(endpoints=endpoints,
                                                    synapses=synapses, 
                                                    inputs=inputs, 
                                                    timeout=timeout, 
                                                    min_success=min_success))

            else:
                results = self.receptor_pool.forward(endpoints=endpoints,
                                    synapses=synapses, 
                                    inputs=inputs, 
                                    timeout=timeout, 
                                    min_success=min_success)
            elapsed_time = t.elapsed_time.microseconds/1e6

        results_dict = []
        num_responses = len(results[0])
        for i in range(num_responses): 
            # row_dict = e.__dict__
            row_dict = {}
            row_dict['code'] = [DatasetModule.response_id2code_map[c] for c in results[1][i]]
            if row_dict['code'][0] != 'Success' and success_only:
                continue
            row_dict['tensor'] = [synapse_tensor.to(self.device) for synapse_tensor in results[0][i]]
            row_dict['latency'] = results[2][i][0]
            row_dict['text_inputs'] = text_inputs
            row_dict['inputs'] = inputs.tolist()
            row_dict['elapsed_time'] = elapsed_time
            row_dict['synapse'] = list(map(str, synapses))
            row_dict['output_size'] = sys.getsizeof(results[0][i])
            row_dict['output_length'] = results[0][i][0].shape[0]
            row_dict['input_token_length'] = int(inputs.shape[0])
            results_dict.append(row_dict)


        if isinstance(queue_topic, str):
            self.queue.put.remote(queue_topic, results_dict)
        return results_dict

    


if __name__ == '__main__':


    import ray

    # st.write(BenchmarkModule.metagraph)
    # module = BenchmarkModule.deploy(actor={'refresh': False, 'name': f'benchmark'})
    # module = DatasetModule.deploy(actor={'refresh': False}, load=True, wrap=True)
    # DatasetModule.ray_restart()
    module = DatasetModule.deploy(actor=False, load=True, wrap=False)

    # ray.get(module.put_batch.remote('fam', [1]*100 ))
    # st.write(ray.get(module.get_batch.remote('fam', 10)))
    # ray.get(module.load_receptor_pool.remote(actor=False))
    # # st.write(module.actor)
    # # topic='train'

    # ray.get(module.delete.remote('receptor_pool'))
    
    for i in range(10):
        resp = module.sample(num_endpoints=100, timeout=1, batch_size=1, min_success=10, queue_topic=None)
        # del module.receptor_pool
        st.write(len(resp), i)
    
    # finished_results = []
    # while running_jobs:
    #     finished_jobs, running_jobs = ray.wait(running_jobs)
    #     if len(finished_jobs)>0:
    #         for job in  finished_jobs:
    #             finished_results += [ray.get(job)]
    #             st.write(len(finished_results))

    # st.write(ray.get(module.sample_generator.remote('train')))

    # st.write(ray.get(queue_server.delete_all.remote()))

    # all_synapses = ray.get(module.getattr.remote('available_synapses'))

    # selected_synapses = st.multiselect('Select Synapses',all_synapses,  all_synapses[:1])
    # module.start_sample_loop.remote(topic='train', synapse=selected_synapses, timeout=1.0, success_only=True, refresh_cache=True, refresh_queue=True)
    # st.write(ray.get(module.sample_cache_count.remote('train')))

    # all_synapses = ray.get(module.getattr.remote('available_synapses'))
    # selected_synapses = st.multiselect('Select Synapses',all_synapses,  all_synapses[:1])
    # ray.get(module.start_sample_loop.remote(topic='test', synapse=selected_synapses, timeout=1, success_only=True, refresh_cache=True, refresh_queue=True))
    # st.write(ray.get(module.sample_cache_count.remote('train')))
    # # st.write(ray.get(module.stop_sample_loop.remote('train')))
    # # st.write(ray.get(module.running_loops.remote()))
    # st.write(ray.get(module.sample_loop.remote('train')))


    # st.write(ray.get(module.sample_loop.remote('test')))
    # st.write()
    
    # st.write(random.uniform(0,1))
    # random.randint()

    # with st.expander('tensors', False):
    #     st.write(results[0])
    # with st.expander('return type', False):
    #     st.write(results[1])

    
