
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

from commune.bittensor import BitModule
from commune import Module

from commune.utils import *


import torch
from torch import nn

from commune.ray.actor_pool import ActorPool

class BenchmarkModule(BitModule):
    __file__ = __file__
    default_config_path = 'bittensor.benchmark.module'
    def __init__(self, config=None, load=True, **kwargs):

        BitModule.__init__(self, config=config, **kwargs)

        
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
        
        if keys == True:
            keys = ['env', 'tokenizer', 'receptor_pool']
        
        if keys in [False, None]:
            return
        
        load_keys = keys

        for load_key in load_keys:

            load_kwargs.get(load_key, {}) 
            load_fn = getattr(self, f'load_{load_key}', None)
            assert load_fn != None, f'{load_key} is suppose to be a function'
            load_fn_kwargs = load_kwargs.get(load_key, {})
            load_fn_args = load_args.get(load_key, [])
            load_fn(*load_fn_args, **load_fn_kwargs)


    def load_dataset(self, **kwargs):
        dataset_kwargs = dict(path='bittensor.dataset', params=dict(block_size=128))
        dataset_kwargs.update(kwargs)
        dataset_kwargs.update(self.config.get('dataset'))
        dataset_class = self.import_object(dataset_kwargs['path'])
        self.dataset = dataset_class(**dataset_kwargs['params'])

    def load_tokenizer(self, **kwargs): 
        if isinstance(getattr(self, 'dataset', None), bittensor.dataset):
            self.tokenizer = self.dataset.tokenizer

        tokenizer_kwargs = dict(path='bittensor.tokenizer',
                            params=dict(version=bittensor.__version__))
        tokenizer_kwargs.update(kwargs)
        tokenizer_kwargs.update(self.config.get('tokenizer'))
        tokenizer_class = self.import_object(tokenizer_kwargs['path'])
        self.tokenizer = tokenizer_class(**tokenizer_kwargs['params'])

    def load_model(self):
        model_config = self.config['model']
        model_class = self.import_object(model_config.get('path'))
        self.model = model_class(**model_config['params'])
        self.num_endpoints = self.model.num_endpoints
    
    def load_optimizer(self,**kwargs):
        optimizer_kwargs = dict(path='torch.optim.Adam', params=dict(lr=0.00032))
        optimizer_kwargs.update(kwargs)
        optimizer_kwargs.update(self.config.get('optimizer', {}))
        optim_class = self.import_object(optimizer_kwargs['path'])
        self.optimizer = optim_class(self.model.parameters(),**optimizer_kwargs['params'])


    def load_metric(self, **kwargs):
        metric_config = self.config['metric']
        metric_class = self.import_object(metric_config.get('path'))
        self.metric = metric_class(**metric_config['params'])




    def load_receptor_pool(self, replicas=3, refresh=False, **kwargs):

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
    @staticmethod
    def causal_lm_loss(labels, logits):
        batch_size = logits.shape[0]
        loss_fct = CrossEntropyLoss()

        losses = []
        for batch in range(batch_size):
            shift_logits = logits[batch, :-1, :].contiguous()
            shift_labels = labels[batch, 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, 50258), shift_labels.view(-1))
            losses.append(loss)
        return torch.tensor(losses)

    @property
    def num_receptors(self):
        return self.num_endpoints

    def get_endpoints(self, num_endpoints=None, random_sample=True):
        if num_endpoints == None:
            num_endpoints =self.num_endpoints
        endpoints =self.metagraph.endpoint_objs

        if random_sample == True:
            endpoint_index_list = list(np.random.randint(0, self.n, (num_endpoints)))
            endpoints = [endpoints[idx] for idx in endpoint_index_list]
        else:
            endpoints = endpoints[:num_endpoints]
        return endpoints

    # def get_loss_fn(self):
    #     return nn.CrossEntropyLoss()
    
    @staticmethod
    def str2synapse(synapse:str):
        return getattr(bittensor.synapse, synapse)()
    @property
    def synapses(self):
        # default_synapses = ['bittensor.synapse.TextCausalLM']
        # synapse_class_strings = self.config.get('synapses', default_synapses)
        # return [self.import_module(s)() for s in synapse_class_strings]
        # return [bittensor.synapse.TextCausalLM()] 
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



    def receptor_pool_forward(self, endpoints, inputs, synapses=None , timeout=1, splits=5, min_success=1.0):
        if synapses == None:
            synapses = self.synapses

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
            
    def predict(self,text, num_endpoints=100, timeout=1, synapses = None, return_type='result', splits=1, return_success_only=True, min_success=1.0, **kwargs):
        
        receptor_kwargs = kwargs.get('receptor')

        if  kwargs.get('receptor') not in [None, False]:
            receptor_kwargs = kwargs.get('receptor', {}) 
            if type(receptor_kwargs) not in [dict]:
                receptor_kwargs = {}
            self.load_receptor_pool(**receptor_kwargs)
        synapses = self.resolve_synapses(synapses=synapses)

        if text == None:
            text = self.raw_sample()

        endpoints = kwargs.get('endpoints')
        if endpoints == None:
            endpoints = self.get_endpoints(num_endpoints=num_endpoints)

        num_endpoints = len(endpoints)

        if isinstance(text, str):
            text = [text]
        inputs = torch.tensor(self.tokenizer(text=text, padding=True)['input_ids'])

        st.write(inputs.shape, 'SHAPE')
        elasped_time = 0
        with Timer(text='Querying Endpoints: {t}', streamlit=True) as t:
            results = self.receptor_pool_forward(endpoints=endpoints, synapses=self.synapses, inputs=inputs, timeout=timeout, splits=splits , min_success=min_success)
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
                    row_dict['splits'] = splits
    


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
                metric_dict['splits'] = splits
                metric_dict['min_success'] = min_success
                metric_dict['num_responses'] = num_responses

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
                     timeout_list = [1,2, 4], 
                     token_length_list=[ 32],
                     num_endpoints_list=[100 , 500, 1000 ],
                     max_worker_threads_list=[100,200,400],
                     min_success_list=[0.1,0.2,0.5, 0.8],
                     replicas_list = [1,2,4],
                     max_active_receptors=[2000],
                     synapse_list = None,
                     path='experiments_2') :

        if synapse_list  == None:
            synapse_list = list(self.synapse_map.values())
        # self.rm_config()
        total_trials = len(timeout_list) *\
                     len(num_endpoints_list)* \
                     len(token_length_list) * \
                     len(max_active_receptors) * \
                     len(replicas_list) * len(max_worker_threads_list) * len(synapse_list)

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
                            for min_success in min_success_list:
                                for synapse in synapse_list:
                                    for i in range(trials):
                                        cnt += 1 
                                        metrics_dict = self.predict(text=text,
                                                                    num_endpoints=num_endpoints,
                                                                    timeout=timeout, 
                                                                    splits=replicas, 
                                                                    synapses=[synapse], 
                                                                    return_type='metric',
                                                                    min_success=min_success)
                                                        
                                        metrics_dict['synapse'] = synapse.__name__
                                        metrics_dict['replicas'] = replicas
                                        print(f'PROGRESS ({path}): {cnt}/{total_trials}')
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

    def run(self):

        loss_fn = nn.CrossEntropyLoss()

        # https://github.com/huggingface/transformers/blob/v4.21.3/src/transformers/models/gptj/modeling_gptj.py#L847

        num_batches = 1
 
        for idx in range(num_batches):
            print("getting next batch of data")
            with Timer(text='Get Batch: {t}', streamlit=True) as t:
                inputs = next(self.dataset)


            with Timer(text='Tokenize: {t}', streamlit=True) as t:
                str_inputs = [self.tokenizer.decode(s) for s in inputs]

            print(f"Querying endpoints")
            # endpoints = self.get_endpoints()
            endpoints = self.get_endpoints()
    

            with Timer(text='Querying Endpoints: {t}', streamlit=True) as t:
                results = ray.get(self.receptor_poolforward.remote(endpoints, synapses=self.synapses, inputs=[inputs] * len(endpoints), timeout=10))

            df = []
            for i,e in enumerate(endpoints): 
                row_dict = e.__dict__
                row_dict['code'] = results[1][i][0]
                row_dict['latency'] = results[2][i][0]
                df.append(row_dict)
            
            df = pd.DataFrame(df)
            st.write(df)

            break

            
            tensors = []
            for tensor in results[0]:
                tensors.append(tensor[0])
            


            codes = []
            codes_count = defaultdict(int)
            for code in results[1]:
                code = code[0]
                codes.append(code)
                codes_count[code] += 1
            for code in sorted(set(codes)):
                print(f"{code}: {codes_count[code]}")
        

            print("Calculating losses for each endpoint")
            all_losses = []
            for _, logits in tqdm(enumerate(tensors)):
                all_losses.append(self.causal_lm_loss(inputs, logits))

            all_losses_tensor = torch.vstack(all_losses).T  # (batch_size, num_endpoints)
            inv_loss_tensor = 1/all_losses_tensor


            print("Model forward")
            sims = self.model(str_inputs)

            print("model backwards")

            ideal_rankings = torch.argsort(all_losses_tensor, axis=1)
            model_rankings = torch.argsort(sims, axis=1)

            loss = loss_fn(sims, inv_loss_tensor)
            #ndcg = metrics.ndcg_score(ideal_rankings, model_rankings)
            print(f"step: {idx} | loss={loss.item():.3f}")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

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

    def raw_sample(self):
        text_field = self.config['dataset']['text_field']
        return self.dataset[random.randint(1,len(self.dataset))][text_field]


    def st_sidebar(self):


        mode = 'Network'
        with st.sidebar.expander(mode):
            with st.form(mode):
                networks = ray.get(self.getattr.remote('networks'))
                network2idx = {n:n_idx for n_idx, n in  enumerate(networks)}
                default_idx = network2idx['nakamoto']
                network = st.selectbox('Select Network', networks ,default_idx)
                # block = st.number_input('Select Block', 0,self.current_block, self.current_block-1)
                
                force_sync = st.checkbox('Force Sync', False)

                submit_button = st.form_submit_button('Sync')
                if submit_button:
                    ray.get(self.set_network.remote(network=network,force_sync=force_sync))
                
        mode = 'Wallet'
        with st.sidebar.expander(mode, True):
            cold2hot_wallets = ray.get(self.getattr.remote('wallets'))
            cold_wallet_options = list(cold2hot_wallets.keys())
            selected_coldwallet = st.selectbox('Cold Wallet:', cold_wallet_options, 0)
            hot_wallet_options = list(cold2hot_wallets[selected_coldwallet].keys())
            selected_hotwallet = st.selectbox('Cold Wallet:', hot_wallet_options, 0)
            



    def st_main(self):

        with st.expander('Query', True):
            with st.form('Query'):
                # input side
                text = st.text_area('Input Text', 'Whadup dawg')
                num_endpoints = st.slider('Number of Endpoints', 0, len(self.endpoints), 50)

                timeout = st.slider('Timeout (seconds)', 0, 20, 2)

                return_type = st.selectbox('Select Return Type', ['df', 'result'], 0)

                submit_button = st.form_submit_button('Query')
                
                if submit_button:
                    output = self.predict(text=text, num_endpoints=num_endpoints, return_type = return_type, timeout=timeout)
                    st.write('Output')
                    st.write(output)
            
        with st.expander('My Endpoints', True):
            my_endpoints = self.my_endpoints
            my_endpoints_df = pd.DataFrame([e.__dict__ for e in my_endpoints])
            
            st.write(my_endpoints_df)
            if len(my_endpoints_df)>0:
                my_endpoints_df.drop(columns=['hotkey', 'coldkey'], inplace=True)
                st.write(my_endpoints_df)
            else:
                st.write('No Endpoints')
            # my_selected_endpoints = st.multiselect('',my_endpoints, my_endpoints)


    @property
    def available_synapses(self):
        return [f for f in dir(bittensor.synapse) if f.startswith('Text')]
    
    @property
    def synapse_map(self):
        return {f:getattr(bittensor.synapse,f) for f in self.available_synapses}


    def get_synapse(self, synapse, *args, **kwargs):
        return self.synapse_map[synapse](*args, **kwargs)

    resolve_synapse = get_synapse
    @classmethod
    def st_terminal(cls):

        input_command = st.text_input('input',  'ls')
        submit_input = st.button('Run')

        if submit_input:
            stdout_output = cls.run_command(input_command).stdout
            output = '\n'.join(stdout_output.split('\n'))
            
            with st.expander('output', True):
                for output in output.split('\n'):
                    st.write(output)
        else:
            output = 'Type in Terminal'

    
        st.write(output)


if __name__ == '__main__':


    import ray

    module = BenchmarkModule.deploy(actor={'refresh': False}, load=['env', 'tokenizer', 'receptor_pool'])

    # st.write(ray.get(module.predict.remote(text=['bro'], timeout=0.5,  return_success_only=True , num_endpoints=20 , return_type='metrics',min_success=10, splits=1)))
    df = ray.get(module.load_experiment.remote(path='experiment'))
    
    st.write(df)
    plot = ray.get(module.getattr.remote('plot'))
    plot.run(df)
    