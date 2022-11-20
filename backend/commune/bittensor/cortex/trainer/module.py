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
import ray
from commune.bittensor.cortex.metric import causal_lm_loss, ranknet_loss
from commune.utils import *
from sklearn import metrics
from scipy.stats import kendalltau

import torch
from torch import nn

from commune.ray.actor_pool import ActorPool


class TrainerModule(Module):
    __file__ = __file__
    default_config_path = 'bittensor.cortex.trainer'

    def __init__(
        self, 
        config = None,
        **kwargs
    ):
        torch.nn.Module.__init__(self)
        Module.__init__(self, config=config, **kwargs)
        self.load()

    def load(self):
        self.load_bitmodule()
        self.load_dataset()
        self.load_model()
        self.load_metric()


    def kill_generators(self, splits=None):
        generator_config_map = self.config['dataset']['generator']
        if splits == None:
            splits = list(generator_config_map.keys())
        for split in splits:
            self.dataset.kill_generator(split)
            
    def start_generators(self, splits=None):
        generator_config_map = self.config['dataset']['generator']
        if splits == None:
            splits = list(generator_config_map.keys())
        for split in splits:
            self.dataset.start_generator(**generator_config_map[split])

    def ls_generators(self):
        return self.dataset.ls_generators()

    def sample(self, split='train'):
        return self.dataset.generate_sample(split=split)


    def load_bitmodule(self, refresh=False, sync=True, **kwargs):
        st.write(self.config)
        bitmodule_config = self.config['bitmodule']
        module_class = Module.get_object(bitmodule_config['module'])
        wallet = kwargs.pop('wallet', bitmodule_config['wallet'])
        network = kwargs.pop('network', bitmodule_config['network'])
        self.bitmodule = module_class.deploy(actor={'refresh': refresh}, override={'network': network, 'wallet': wallet}, load=True, wrap = True)
        
        if sync:
            self.sync()
        return self.bitmodule

    def sync(self):
        self.bitmodule.sync()
        self.metagraph_state = self.bitmodule.getattr('metagraph_state')

    def load_dataset(self, refresh=False):
        dataset_config = self.config['dataset']
        dataset_config['actor']['refresh'] = refresh
        dataset_class = self.get_object(dataset_config.get('module'))
        self.dataset = dataset_class.deploy(actor=dataset_config['actor'])


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
        metrics_config = self.config['metric']
        self.metric = {}
        for metric_key, metric_config in metrics_config.items():
            metric_class = self.import_object(metric_config.get('path'))

            self.metric[metric_key] = metric_class



    # def run(self, 
    #         steps=1000, 
    #         num_endpoints=None, 
    #         timeout=1, 
    #         synapses = ['TextCausalLM'], 
    #         splits=1, 
    #         experiment='exp1',
    #         batch_size = 1,
    #          **kwargs):

    #     if num_endpoints == 0:
    #         num_endpoints = self.num_endpoints

    #     self.model.train()

    #     synapses = self.resolve_synapses(synapses=synapses)

    #     perf_df = pd.DataFrame()
    #     for step in range(steps):
    #         print("getting next batch of data")

    #         with Timer(text='RUN: Get Samples: {t}', streamlit=True) as t:
    #             str_inputs = self.sample_batch(batch_size=batch_size)
    #             inputs = torch.tensor(self.tokenizer(text=str_inputs, padding=True)['input_ids']).to(self.device)
    #             endpoints = self.get_endpoints(num_endpoints=num_endpoints)

    #         with Timer(text='RUN: Querying Endpoints: {t}', streamlit=True) as t:

    #             results = self.receptor_pool_forward(endpoints=endpoints,
    #                                                 synapses=synapses, 
    #                                                 inputs=inputs, 
    #                                                 timeout=timeout, 
    #                                                 splits=splits )

    #             tensors = []
    #             for tensor in results[0]:
    #                 tensors.append(tensor[0].to(self.device))

    #         print("Calculating losses for each endpoint")

    #         with Timer(text='RUN: Calculating losses for each endpoint : {t}', streamlit=True) as t:

    #             all_losses = []
    #             for _, logits in tqdm(enumerate(tensors)):
    #                 all_losses.append(causal_lm_loss(inputs, logits))

    #             all_losses_tensor = torch.vstack(all_losses).T  # (batch_size, num_endpoints)
    #             ideal_rankings = torch.argsort(torch.argsort(all_losses_tensor, axis=1, descending=False), axis=1)

    #         with Timer(text='RUN: model.get_all_sims : {t}', streamlit=True) as t:

    #             all_sims = self.model.get_all_sims(str_inputs)
    #             model_rankings = torch.argsort(torch.argsort(all_sims, axis=1, descending=True), axis=1)



    #         with Timer(text='RUN: Model B and RankNet Loss : {t}', streamlit=True) as t:
    #             x1 = [[] for _ in range(all_losses_tensor.shape[0])]
    #             x2 = [[] for _ in range(all_losses_tensor.shape[0])]
    #             ys = [[] for _ in range(all_losses_tensor.shape[0])]
    #             for batch in range(all_losses_tensor.shape[0]):
    #                 for idx in range(all_losses_tensor.shape[1]):
    #                     for idx2 in range(all_losses_tensor.shape[1]):
    #                         # TODO: Contrastive sampling improvements
    #                         # while len(x1[batch]) != 10:
    #                         # idx2 = randint(0, all_losses_tensor.shape[1] - 1)
    #                         if idx == idx2:
    #                             continue
    #                         d = all_losses_tensor[batch][idx] - all_losses_tensor[batch][idx2]
    #                         t = (
    #                             1.0
    #                             if all_losses_tensor[batch][idx] < all_losses_tensor[batch][idx2]
    #                             else 0.0
    #                             if all_losses_tensor[batch][idx] > all_losses_tensor[batch][idx2]
    #                             else 0.5
    #                         )
    #                         x1[batch].append(idx)
    #                         x2[batch].append(idx2)
    #                         ys[batch].append(t)

    #             x1, x2, ys = torch.tensor(x1).to(self.device), torch.tensor(x2).to(self.device), torch.tensor(ys).to(self.device)
    #             print(f"Batch size: {x1.shape}")
    #             print("Model forward")
    #             s1 = self.model(str_inputs, x1)
    #             s2 = self.model(str_inputs, x2)
    #             loss = ranknet_loss(s1, s2, ys)
    #             print("model backwards")


    #             self.model.optimizer.zero_grad()
    #             loss.backward()
    #             self.model.optimizer.step()

    #         with Timer(text='RUN: Calculating and Saving  Metrics: {t}', streamlit=True) as t:

    #             sorted_losses_tensor = all_losses_tensor.clone()
    #             sorted_by_model_rank = all_losses_tensor.clone()

    #             ideal_idx = torch.argsort(all_losses_tensor, axis=1, descending=False)
    #             model_idx = torch.argsort(all_sims, axis=1, descending=True)

    #             for i in range(sorted_by_model_rank.shape[0]):
    #                 sorted_losses_tensor[i, :] = sorted_losses_tensor[i, ideal_idx[i]]
    #                 sorted_by_model_rank[i, :] = sorted_by_model_rank[i, model_idx[i]]

    #             topk = 10
    #             ideal_softrank = torchsort.soft_rank(all_losses_tensor, regularization_strength=1e-6)
    #             model_softrank = torchsort.soft_rank(-all_sims, regularization_strength=1e-6)

    #             tau_softrank, _p = kendalltau(model_softrank.cpu().detach().numpy(), ideal_softrank.cpu().detach().numpy())
    #             tau_losses, _p = kendalltau(sorted_losses_tensor.cpu().detach().numpy(), sorted_by_model_rank.cpu().detach().numpy())

    #             tau_rank_topk, _p = kendalltau(model_softrank[:, :topk].cpu().detach().numpy(), ideal_softrank[:, :topk].cpu().detach().numpy())
    #             tau_loss_topk, _p = kendalltau(sorted_losses_tensor[:, :topk].cpu().detach().numpy(), sorted_by_model_rank[:, :topk].cpu().detach().numpy())


    #             ndcg = metrics.ndcg_score(1 / (ideal_rankings.cpu() + 1), 1 / (model_rankings.cpu() + 1))
                
                
    #             metrics_dict = {
    #                 "step": step, 
    #                 "loss": loss.item(),
    #                 "ndcg": ndcg,
    #                 "tau_softrank": tau_softrank,
    #                 "tau_losses": tau_losses,
    #                 "tau_softrank_topk": tau_rank_topk,
    #                 "tau_loss_topk": tau_loss_topk
    #             }
                                        
    #             self.put_json(f'{experiment}/perf_json_{step}', metrics_dict)




if __name__ == '__main__':

    module = TrainerModule.deploy(actor={'refresh':False, 'wrap':True})

    st.write(module.sample())

    # st.write('FOOOK')
