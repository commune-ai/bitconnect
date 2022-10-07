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
import torchsort
from commune.bittensor import BitModule
from commune import BaseModule

from commune.bittensor.cortex.metric import causal_lm_loss, ranknet_loss
from commune.utils import *
from sklearn import metrics
from scipy.stats import kendalltau


import torch
from torch import nn

from commune.ray.actor_pool import ActorPool


class CortexModule(BitModule):
    __file__ = __file__
    default_config_path = 'bittensor.cortex.base'
    def __init__(self, config=None, **kwargs):
        load = kwargs.get('load')
        BitModule.__init__(self, config=config, load=False, **kwargs)

        

        if type(load) in [dict]:
            self.load(**load)
        else:
            self.load(load)


    @property
    def debug(self):
        return self.config.get('debug', False)

    def load_env(self, path=None, **kwargs):
        return BitModule.load(self=self, path=path, **kwargs)

    def load(self, keys=True, load_kwargs={}, load_args={}, **kwargs):
        if keys in [False, None]:
            return

        if keys == True:
            keys = ['env', 'model', 'dataset' ]
        
        load_keys = keys

        for load_key in load_keys:

            load_kwargs.get(load_key, {}) 
            load_fn = getattr(self, f'load_{load_key}', None)
            assert load_fn != None, f'{load_key} is suppose to be a function'
            load_fn_kwargs = load_kwargs.get(load_key, {})
            load_fn_args = load_args.get(load_key, [])
            load_fn(*load_fn_args, **load_fn_kwargs)


    def load_dataset(self, **kwargs):
        dataset_kwargs = {**self.config.get('dataset'), **kwargs}
        self.dataset = self.get_module(**dataset_kwargs)

    @property
    def device(self):
        if torch.cuda.is_available():
            return 'cpu'
        else:
            return 'cpu'


    def load_model(self):
        model_config = self.config['model']
        model_class = self.import_object(model_config.get('path'))
        self.model = model_class(**model_config['params'])
        self.num_endpoints = self.model.num_endpoints
        self.model  = self.model.to(self.device)
    
    
    def load_metric(self, **kwargs):
        metric_config = self.config['metric']
        metric_class = self.import_object(metric_config.get('path'))
        self.metric = metric_class(**metric_config['params'])




    def load_receptor_pool(self, replicas=1, refresh=True, **kwargs):

        receptor_kwargs = dict(max_worker_threads=150, max_active_receptors=512)
        config_receptor = self.config.get('receptor_pool', {})
        
        config_receptor_kwargs = config_receptor.get('params', config_receptor.get('kwargs', {}) )
        
        receptor_kwargs.update(config_receptor_kwargs)
        receptor_kwargs.update(kwargs)


        default_receptor_path = 'bittensor.receptor.pool.module.ReceptorPoolModule'
        receptor_module_path = config_receptor_kwargs.get('module',default_receptor_path )
        receptor_pool_module = self.get_object(receptor_module_path)

        with Timer(text='Deploying Actors: {t}', streamlit=True) as t:
            actors = [receptor_pool_module.deploy(actor={'refresh': refresh, 'name': f'ReceptorPool{i}'},wallet=self.wallet,**receptor_kwargs) for i in range(replicas)]
            # st.write(ray.get([a.getattr.remote('actor_name') for a in actors]))
            st.write(actors)
            self.receptor_pool = ActorPool(actors=actors)



        return self.receptor_pool

    @property
    def num_receptors(self):
        return self.num_endpoints

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

    # def get_loss_fn(self):
    #     return nn.CrossEntropyLoss()
    
    @staticmethod
    def str2synapse(synapse:str):
        return getattr(bittensor.synapse, synapse)()
    @property
    def synapses(self):
        synsapses = list(map(self.str2synapse, self.config.get('synapses',['TextCausalLM'])) )
        return synsapses

    def resolve_synapses(self, synapses=None):
        if synapses == None:
            synapses = self.synapses
        elif isinstance(synapses, str):
            synapses = [self.str2synapse(synapses)]
        elif isinstance(synapses, list):
            if isinstance(synapses[0], str):
                synapses = list(map(self.str2synapse, synapses))



    def receptor_pool_forward(self, endpoints, inputs, synapses=None , timeout=1, splits=5):
        if synapses == None:
            synapses = self.synapses

        endpoints_split_list = chunk(endpoints, num_chunks=splits)

        kwargs_list = []

        for endpoints_split in endpoints_split_list:

            kwargs_list.append(dict(endpoints=endpoints_split, inputs=[inputs]*len(endpoints_split), synapses=synapses , timeout=timeout))

        agg_results = [[],[],[]]
        results_generator = self.receptor_pool.map_unordered(lambda a,v: a.forward.remote(**v), kwargs_list)
       
        for results in results_generator:
            for i,result in enumerate(results):
                agg_results[i].extend(result)


        # st.write(len(results[0]), len(results[1]),  len(results[2]))
        # st.write([(len(result), type(result)) for result in results])
        return agg_results
            


    def predict(self,text, num_endpoints=100, timeout=1, synapses = None, return_type='result', splits=10,  **kwargs):
        synapses = self.resolve_synapses(synapses=synapses)
        receptor_kwargs = kwargs.get('receptor')

        if  kwargs.get('receptor') not in [None, False]:
            receptor_kwargs = kwargs.get('receptor', {}) 
            if type(receptor_kwargs) not in [dict]:
                receptor_kwargs = {}
            self.load_receptor_pool(**receptor_kwargs)


        endpoints = kwargs.get('endpoints')

        num_endpoints = len(endpoints)

        if isinstance(text, str):
            text = [text]
        inputs = torch.tensor(self.tokenizer(text=text, padding=True)['input_ids'])

        elasped_time = 0
        with Timer(text='Querying Endpoints: {t}', streamlit=True) as t:
            results = self.receptor_pool_forward(endpoints=endpoints, synapses=self.synapses, inputs=inputs, timeout=timeout, splits=splits )
            elasped_time = t.elapsed_time
        
        num_responses = len(results[1])

        if return_type in ['df'] or return_type in ['metric', 'metrics']:
            df = []
   
            for i,e in enumerate(endpoints): 
                if i < num_responses:
                    row_dict = e.__dict__
                    row_dict['code'] = results[1][i][0]
                    row_dict['latency'] = results[2][i][0]
                    # row_dict['elapsed_time'] = elasped_time
                    row_dict['timeout'] = timeout
                    row_dict['return_endpoints'] = num_responses
                    row_dict['query_endpoints'] = num_endpoints
                    row_dict['output_size'] = sys.getsizeof(results[0][i])
                    row_dict['output_length'] = results[0][i][0].shape[0]
                    row_dict['input_length'] = int(inputs.shape[0])
    


                    df.append(row_dict)
            
            df = pd.DataFrame(df)

            returnid2code = {k:f'{v}' for k,v in zip(bittensor.proto.ReturnCode.values(),bittensor.proto.ReturnCode.keys())}
            df['code'] = df['code'].map(returnid2code)
            df = pd.merge(self.metagraph.to_dataframe(), df, on='uid')

        
            if return_type in ['metrics', 'metric']:
                metric_dict = {}
                metric_dict['success_count'] = int(df['code'].apply(lambda x: x == 'Success').sum())
                metric_dict['success_rate'] = df['code'].apply(lambda x: x == 'Success').mean()
                metric_dict['num_endpoints'] = num_endpoints
                metric_dict['timeout'] = int(df['timeout'].iloc[0])
                metric_dict['latency'] = df['latency'].iloc[0]
                metric_dict['input_length'] = df['input_length'].iloc[0]
                metric_dict['elapsed_time'] = elasped_time.total_seconds()
                metric_dict['samples_per_second'] = metric_dict['success_count'] / metric_dict['elapsed_time']
                for k in ['trust', 'consensus','stake', 'incentive', 'dividends', 'emission', 'latency']:
                    # for mode in ['mean', 'std', 'max', 'min']:
                    metric_dict[k] =  getattr(df[k], 'mean')()

                metric_dict = {k:float(v)for k,v in metric_dict.items()}
                return metric_dict
            else:
                return df

        elif return_type in ['results', 'result']:

            # return torch.cat([tensor[0] for tensor in results[0]], 0)
            return results



    def run_experiment(self,  trials=1,
                     timeout_list = [1,2,5], 
                     token_length_list=[ 16, 32, 64],
                     num_endpoints_list=[10,20,50,100,500],
                     max_worker_threads_list=[50, 100, 200],
                     replicas_list = [4, 2, 1],
                     max_active_receptors=[128,512],
                     synapse_list = None,
                     path='experiments') :

        # self.rm_config()
        total_trials = len(timeout_list) *\
                     len(num_endpoints_list)* \
                     len(token_length_list) * \
                     len(max_active_receptors) * \
                     len(replicas_list)

        if synapse_list  == None:
            synapse_list = list(self.synapse_map.values())
        cnt = 0

        text_base = 'hello'
               

        for max_worker_threads in max_worker_threads_list:

            # self.load_receptor_pool(replicas=replicas, max_worker_threads=max_worker_threads , refresh=True)

            for replicas in replicas_list:
                self.load_receptor_pool(replicas=replicas, max_worker_threads=max_worker_threads , refresh=True)
  
                for token_length in token_length_list:
                    text = [text_base]*token_length
                    for timeout in timeout_list:
                        for num_endpoints in num_endpoints_list:
                            for synapse in synapse_list:
                                for i in range(trials):
                                    cnt += 1 
                                    
                                    metrics_dict = self.predict(text = text, num_endpoints=num_endpoints, timeout=timeout, splits=replicas, synapses=[synapse], return_type='metric')
                                    metrics_dict['synapse'] = synapse.__name__
                                    print(f'PROGRESS: {cnt}/{total_trials}')
                                    self.put_json(f'{path}/metrics_dict_{cnt}', metrics_dict)
            

    def load_experiment(self, path='experiments'):
        df = []
        if self.client.local.exists(path):
            for p in self.client.local.ls(path):
                df.append(self.client.local.get_json(p))
        else:
            for p in self.ls_json(path):
                df.append(self.get_json(p))

        df =  pd.DataFrame(df)
        # df = pd.concat(df)
        # returnid2code = {k:f'{v}' for k,v in zip(bittensor.proto.ReturnCode.values(),bittensor.proto.ReturnCode.keys())}
        # df['code'] = df['code'].map(returnid2code)
        return df

    def st_experiment(self, path='experiments'):

        df = self.load_experiment()
        with st.expander('dataframe', True):
            st.write(df.iloc[:50]) 
        with st.expander('Latency Histogram', True):
            fig =  module.plot.histogram(df, x='latency', color="code")
            fig.update_layout(legend={'traceorder':'normal'})
            st.write(fig)
        import plotly.express as px
        with st.expander('Return Code Pie Chart', True):
            code_count_dict = dict(df['code'].value_counts())
            codes_count_df =   pd.DataFrame({'codes': list(code_count_dict.keys()), 'values':  list(code_count_dict.values())})
            fig = px.pie(names=list(code_count_dict.keys()), 
                        values= list(code_count_dict.values()))
            st.write(codes_count_df)
            st.write(fig)

    @property
    def coldkey_address(self):
        return self.wallet.coldkeypub.ss58_address

    @property
    def hotkey_address(self):
        return self.wallet.hotkey.ss58_address

    @property
    def endpoints(self):
        return self.metagraph.endpoint_objs
    @property
    def hotkey_endpoints(self):
        return self.my_endpoints(mode='hotkey')

    @property
    def coldkey_endpoints(self):
        return self.my_endpoints(mode='coldkey')

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

    def run(self, 
            steps=1000, 
            num_endpoints=None, 
            timeout=1, 
            synapses = ['TextCausalLM'], 
            splits=1, 
            experiment='exp1',
            batch_size = 1,
             **kwargs):

        if num_endpoints == 0:
            num_endpoints = self.num_endpoints

        self.model.train()

        synapses = self.resolve_synapses(synapses=synapses)

        perf_df = pd.DataFrame()
        for step in range(steps):
            print("getting next batch of data")

            with Timer(text='RUN: Get Samples: {t}', streamlit=True) as t:
                str_inputs = self.sample_batch(batch_size=batch_size)
                inputs = torch.tensor(self.tokenizer(text=str_inputs, padding=True)['input_ids']).to(self.device)
                endpoints = self.get_endpoints(num_endpoints=num_endpoints)

            with Timer(text='RUN: Querying Endpoints: {t}', streamlit=True) as t:

                results = self.receptor_pool_forward(endpoints=endpoints,
                                                    synapses=synapses, 
                                                    inputs=inputs, 
                                                    timeout=timeout, 
                                                    splits=splits )

                tensors = []
                for tensor in results[0]:
                    tensors.append(tensor[0].to(self.device))

            print("Calculating losses for each endpoint")

            with Timer(text='RUN: Calculating losses for each endpoint : {t}', streamlit=True) as t:

                all_losses = []
                for _, logits in tqdm(enumerate(tensors)):
                    all_losses.append(causal_lm_loss(inputs, logits))

                all_losses_tensor = torch.vstack(all_losses).T  # (batch_size, num_endpoints)
                ideal_rankings = torch.argsort(torch.argsort(all_losses_tensor, axis=1, descending=False), axis=1)

            with Timer(text='RUN: model.get_all_sims : {t}', streamlit=True) as t:

                all_sims = self.model.get_all_sims(str_inputs)
                model_rankings = torch.argsort(torch.argsort(all_sims, axis=1, descending=True), axis=1)



            with Timer(text='RUN: Model B and RankNet Loss : {t}', streamlit=True) as t:
                x1 = [[] for _ in range(all_losses_tensor.shape[0])]
                x2 = [[] for _ in range(all_losses_tensor.shape[0])]
                ys = [[] for _ in range(all_losses_tensor.shape[0])]
                for batch in range(all_losses_tensor.shape[0]):
                    for idx in range(all_losses_tensor.shape[1]):
                        for idx2 in range(all_losses_tensor.shape[1]):
                            # TODO: Contrastive sampling improvements
                            # while len(x1[batch]) != 10:
                            # idx2 = randint(0, all_losses_tensor.shape[1] - 1)
                            if idx == idx2:
                                continue
                            d = all_losses_tensor[batch][idx] - all_losses_tensor[batch][idx2]
                            t = (
                                1.0
                                if all_losses_tensor[batch][idx] < all_losses_tensor[batch][idx2]
                                else 0.0
                                if all_losses_tensor[batch][idx] > all_losses_tensor[batch][idx2]
                                else 0.5
                            )
                            x1[batch].append(idx)
                            x2[batch].append(idx2)
                            ys[batch].append(t)

                x1, x2, ys = torch.tensor(x1).to(self.device), torch.tensor(x2).to(self.device), torch.tensor(ys).to(self.device)
                print(f"Batch size: {x1.shape}")
                print("Model forward")
                s1 = self.model(str_inputs, x1)
                s2 = self.model(str_inputs, x2)
                loss = ranknet_loss(s1, s2, ys)
                print("model backwards")


                self.model.optimizer.zero_grad()
                loss.backward()
                self.model.optimizer.step()

            with Timer(text='RUN: Calculating and Saving  Metrics: {t}', streamlit=True) as t:

                sorted_losses_tensor = all_losses_tensor.clone()
                sorted_by_model_rank = all_losses_tensor.clone()

                ideal_idx = torch.argsort(all_losses_tensor, axis=1, descending=False)
                model_idx = torch.argsort(all_sims, axis=1, descending=True)

                for i in range(sorted_by_model_rank.shape[0]):
                    sorted_losses_tensor[i, :] = sorted_losses_tensor[i, ideal_idx[i]]
                    sorted_by_model_rank[i, :] = sorted_by_model_rank[i, model_idx[i]]

                topk = 10
                ideal_softrank = torchsort.soft_rank(all_losses_tensor, regularization_strength=1e-6)
                model_softrank = torchsort.soft_rank(-all_sims, regularization_strength=1e-6)

                tau_softrank, _p = kendalltau(model_softrank.cpu().detach().numpy(), ideal_softrank.cpu().detach().numpy())
                tau_losses, _p = kendalltau(sorted_losses_tensor.cpu().detach().numpy(), sorted_by_model_rank.cpu().detach().numpy())

                tau_rank_topk, _p = kendalltau(model_softrank[:, :topk].cpu().detach().numpy(), ideal_softrank[:, :topk].cpu().detach().numpy())
                tau_loss_topk, _p = kendalltau(sorted_losses_tensor[:, :topk].cpu().detach().numpy(), sorted_by_model_rank[:, :topk].cpu().detach().numpy())


                ndcg = metrics.ndcg_score(1 / (ideal_rankings.cpu() + 1), 1 / (model_rankings.cpu() + 1))
                
                
                metrics_dict = {
                    "step": step, 
                    "loss": loss.item(),
                    "ndcg": ndcg,
                    "tau_softrank": tau_softrank,
                    "tau_losses": tau_losses,
                    "tau_softrank_topk": tau_rank_topk,
                    "tau_loss_topk": tau_loss_topk
                }
                                        
                self.put_json(f'{experiment}/perf_json_{step}', metrics_dict)




if __name__ == '__main__':


    import ray

    # st.write(BenchmarkModule.metagraph)
    # module = BenchmarkModule.deploy(actor={'refresh': False, 'name': f'benchmark'})
    # CortexModule.ray_restart()
    module = CortexModule.deploy(actor={'refresh': True}, load=True)
    dataset = ray.get(module.getattr.remote('dataset'))

    with Timer('bro: {t}', streamlit=True):
        st.write(len(ray.get(dataset.sample.remote(num_endpoints=100, timeout=1, splits=4))))
    
    # # module.run()
    # st.write()

    # module = BenchmarkModule.deploy(actor={'refresh': False})

