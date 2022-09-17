
import streamlit as st
from random import shuffle, seed
from collections import defaultdict

import bittensor
import torch
from torch import nn
from tqdm.auto import tqdm
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from commune.bittensor import BitModule


import torch
from torch import nn
from sentence_transformers import SentenceTransformer

class BenchmarkModule(BitModule):
    __file__ = __file__
    default_config_path = 'bittensor.benchmark'
    def __init__(self, config=None, **kwargs):
        BitModule.__init__(self, config=config, **kwargs)
        if kwargs.get('sync') == False:
            self.sync()
        self.load_state()

    @property
    def debug(self):
        return self.config.get('debug', False)

    def load_state(self):
        self.load_dataset()
        self.load_model()
        self.load_optimizer()
        self.load_metric()
        self.load_receptor_pool()

    def load_dataset(self, **kwargs):
        dataset_kwargs = dict(path='bittensor.dataset', params=dict(block_size=128))
        dataset_kwargs.update(kwargs)
        dataset_kwargs.update(self.config.get('dataset'))
        dataset_class = self.import_object(dataset_kwargs['path'])
        
        self.dataset = dataset_class(**dataset_kwargs['params'])
        self.tokenizer = self.dataset.tokenizer

    def load_model(self):
        model_config = self.config['model']
        self.model = RankingModel(**model_config['params'])
        self.num_endpoints = self.model.num_endpoints
    
    def load_optimizer(self,**kwargs):
        optimizer_kwargs = dict(path='torch.optim.Adam', params=dict(lr=0.00032))
        optimizer_kwargs.update(kwargs)
        optimizer_kwargs.update(self.config.get('optimizer', {}))
        optim_class = self.import_object(optimizer_kwargs['path'])
        self.optimizer = optim_class(self.model.parameters(),**optimizer_kwargs['params'])


    def load_metric(self, **kwargs):
        metric_config = self.config['metric']
        self.metric = RankingLoss(**metric_config['params'])

    def load_receptor_pool(self, **kwargs):

        receptor_kwargs = dict(max_worker_threads=64, max_active_receptors=512)
        receptor_kwargs.update(kwargs)
        receptor_kwargs.update(self.config.get('receptor_pool', {}))
        self.receptor_pool = bittensor.receptor_pool(**receptor_kwargs,wallet=self.wallet)



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

    def get_endpoints(self, num_endpoints=None):
        if num_endpoints == None:
            num_endpoints =self.num_endpoints
        endpoints =self.graph.endpoint_objs
        shuffle(endpoints)
        endpoints = endpoints[:self.num_receptors]
        return endpoints

    # def get_loss_fn(self):
    #     return nn.CrossEntropyLoss()
    
    @property
    def synapses(self):
        # default_synapses = ['bittensor.synapse.TextCausalLM']
        # synapse_class_strings = self.config.get('synapses', default_synapses)
        # return [self.import_module(s)() for s in synapse_class_strings]
        return [bittensor.synapse.TextCausalLM()]   
    def run(self):

        loss_fn = nn.CrossEntropyLoss()

        # https://github.com/huggingface/transformers/blob/v4.21.3/src/transformers/models/gptj/modeling_gptj.py#L847

        num_batches = 100
 
        for idx in range(num_batches):
            print("getting next batch of data")
            inputs = next(self.dataset)
            str_inputs = [self.tokenizer.decode(s) for s in inputs]
            print(f"Querying endpoints")
            endpoints = self.get_endpoints()
            results = self.receptor_pool.forward(endpoints, synapses=self.synapses, inputs=[inputs] * len(endpoints), timeout=20)
            st.write(results)
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
            print()

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


    def priority(self, pubkey:str, request_type:bittensor.proto.RequestType, inputs_x) -> float:
        r"""Calculates the priority on requests based on stake and size of input
            Args:
                pubkey ( str, `required`):
                    The public key of the caller.
                inputs_x ( :obj:`torch.Tensor`, `required`):
                    torch inputs to be forward processed.
                request_type ( bittensor.proto.RequestType, `required`):
                    the request type ('FORWARD' or 'BACKWARD').
        """
        try:        
            uid = self.graph.hotkeys.index(pubkey)
            priority = self.graph.S[uid].item()
        
        except:
            # zero priority for those who are not registered.
            priority =  0

        return priority

    def forward_generate(self, inputs_x:torch.FloatTensor, synapse, model_output = None):
        tokens = self.model.token_remap(inputs_x.to(model.device))
        output = self.model.pre_model.generate(
            input_ids=tokens['input_ids'],
            attention_mask=tokens['attention_mask'],
            max_length=max(tokens['input_ids'].shape[1] + 1, synapse.num_to_generate),
            num_beams=synapse.num_beams,
            no_repeat_ngram_size=synapse.no_repeat_ngram_size,
            early_stopping = synapse.early_stopping,
            do_sample=synapse.do_sample,
            top_p=synapse.top_p,
            num_return_sequences=synapse.num_return_sequences,
            temperature = synapse.temperature,
            repetition_penalty = synapse.repetition_penalty,
            length_penalty = synapse.length_penalty,
            max_time = synapse.max_time,
            num_beam_groups = synapse.num_beam_groups,
        )
        raw_texts = [self.model.tokenizer.decode(out) for out in output]
        tokens = [self.model.std_tokenizer.encode(raw_text, return_tensors="pt")[:,:synapse.num_to_generate].view(-1) for raw_text in raw_texts]
        bittensor_output = pad_sequence(tokens, batch_first=True)
        return None, model_output, bittensor_output

    def forward_hidden_state(inputs_x:torch.FloatTensor, synapse, model_output = None):
        with mutex:
            message, model_output, hidden = self.model.encode_forward(inputs_x.to(model.device), model_output=model_output)
        return message, model_output, hidden

    def forward_casual_lm(self, inputs_x:torch.FloatTensor, synapse, model_output = None):
        with mutex:
            message, model_output, logits = self.model.encode_forward_causallm(inputs_x.to(model.device), model_output=model_output)
        return message, model_output, logits

    def forward_casual_lm_next(self, inputs_x: torch.FloatTensor, synapse, model_output=None):
        with self.mutex:
            message, model_output, topk_token_phrases = self.model.encode_forward_causallmnext(inputs_x.to(model.device),
                                                                                        topk=synapse.topk,
                                                                                        model_output=model_output)
        # topk_token_phrases: [sum_b(sum_k(len(phrase_k) + 1)_b)] contains topk token phrases and probabilities
        #   Compacted 1-D tensor >= batch_size * (2 * topk + 1)
        return message, model_output, topk_token_phrases

    @staticmethod
    def blacklist(pubkey:str, request_type:bittensor.proto.RequestType) -> bool:
        r"""Axon security blacklisting, used to blacklist message from low stake members
            Args:
                pubkey ( str, `required`):
                    The public key of the caller.
                request_type ( bittensor.proto.RequestType, `required`):
                    the request type ('FORWARD' or 'BACKWARD').
        """
        # Check for registrations

        def registration_check():
            # If we allow non-registered requests return False = not blacklisted.
            is_registered = pubkey in metagraph.hotkeys
            if not is_registered:
                if config.neuron.blacklist_allow_non_registered:
                    
                    return False
                raise Exception('Registration blacklist')

        # Check for stake
        def stake_check() -> bool:
            # Check stake.
            uid = metagraph.hotkeys.index(pubkey)
            if metagraph.S[uid].item() < config.neuron.blacklist.stake:
                raise Exception('Stake blacklist')
            return False

        # Check for time
        def time_check():
            current_time = datetime.now()
            # Only check if the request are forward requests
            timecheck = timecheck_dicts[request_type]
            if pubkey in timecheck.keys():
                prev_time = timecheck[pubkey]
                if current_time - prev_time >= timedelta(seconds=config.neuron.blacklist.time):
                    timecheck[pubkey] = current_time
                else:
                    timecheck[pubkey] = current_time
                    raise Exception('Time blacklist')
            else:
                timecheck[pubkey] = current_time
        
            return False


        # Black list or not
        try:
            registration_check()

            time_check()

            #stake_check()
            
            return False

        except Exception as e:
            return True
    
    @staticmethod
    def synapse_check(synapse, hotkey):
        """
            Custom synapse function to protect certain synapse functions depending on the stake and weight.
            Certain synapses require more compute than others. For instance, TEXT_SEQ_2_SEQ requires a significantly
            more commitment by the server than a requeset for TEXT_CAUSAL_LM_NEXT.

            Args:
                synapse (:obj:`bittensor.proto.SynapseArgs`, `required`): 
                    The proto message that contains additional args for individual synapse functions
                hotkey (:obj:`torch.FloatTensor`, `required`):
                    The hotkey that sent the request

        """
        ## Uid that sent the request
        incoming_uid = metagraph.hotkeys.index(hotkey)
        if synapse.synapse_type == bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE:
            
            if metagraph.S[incoming_uid] < config.neuron.lasthidden_stake:
                return False
            
        elif synapse.synapse_type == bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM:

            if metagraph.S[incoming_uid] < config.neuron.causallm_stake:
                return False

        elif synapse.synapse_type == bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM_NEXT:

            if metagraph.S[incoming_uid] < config.neuron.causallmnext_stake:
                return False

        elif synapse.synapse_type == bittensor.proto.Synapse.SynapseType.TEXT_SEQ_2_SEQ:

            if (metagraph.S[incoming_uid] < config.neuron.seq2seq_stake) and (metagraph.S[incoming_uid,  uid]):
                return False     
        else:
            return False

        return True

    @staticmethod
    def backward_callback(inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor, synapses=[] ):
        """
            The default backward callback when no callback is attached: Is used to call specific synapse functions

            Args:
                inputs_x (:obj:`torch.FloatTensor`, `required`): 
                    The inputs that will be passed to the synapse functions
                grads_dy (:obj:`torch.FloatTensor`, `required`):
                    The gradients that will be passed to the synapse functions
                synapses (:obj: list of bittensor.proto.SynapseArgs, 'Optional')
                    The proto message that contains additional args for individual synapse functions

            Returns:
                response_tensors: (:obj: list of bittensor.proto.Tensor, `required`): 
                    serialized tensor response from the nucleus call or None.
                response_codes: (:obj: list of bittensor.proto.ReturnCode, `required`)
                    return code associated with forward call i.e. Success of Timeout.
                response_messages: (:obj: list of strings, `required`)
                    return message associated with synapse call
        """
        # --- initialize response variables --- 
        response_tensors = []
        response_codes = []
        response_messages = []
        
        if not config.neuron.remote_train:
            return response_tensors, response_codes, response_messages

        # --- calling attached synapses ---
        with mutex and torch.enable_grad() and torch.autograd.set_detect_anomaly(True):
            for index, synapse in enumerate(synapses):
                try:
                    if synapse.synapse_type in axon.synapse_callbacks and axon.synapse_callbacks[synapse.synapse_type] != None:
                        model_output, response_tensor = axon.synapse_callbacks[synapse.synapse_type](inputs_x[index], synapse)
                        grads_dy_norm = grads_dy[index]/(grads_dy[index].sum() + 0.00001)
                        torch.autograd.backward (
                            tensors = [ response_tensor ],
                            grad_tensors = [ grads_dy_norm ],
                            retain_graph=True
                        )                        
                        model.backward_gradients_count += inputs_x[index].size(0)
                        response_tensors.append(None)
                        response_codes.append(bittensor.proto.ReturnCode.Success)
                        response_messages.append('Success')
                    else:
                        response_tensors.append(None)
                        response_codes.append(bittensor.proto.ReturnCode.NotImplemented)
                        response_messages.append('Not Implemented')
                except Exception as e:
                    # --- Exception Hit in Synapse ---
                    response_tensors.append(None)
                    response_codes.append(bittensor.proto.ReturnCode.UnknownException)
                    response_messages.append(str(e))

        return response_tensors, response_codes, response_messages


    def optimizer_step():
        optimizer.step()
        optimizer.zero_grad()

    def serve(self, 
            config = None, 
            subtensor = None,
            wallet = None,
            axon= None,
            metagraph = None,
        ):

        subtensor = self.resolve(key='subtensor', value=axon)
        wallet = self.resolve(key='wallet', value=axon)
        axon= self.resolve(key='axon', value=axon)
        metagraph = self.resolve(key='graph', value=axon)

        config.to_defaults()

        metagraph.load().sync().save()

        # Create our optimizer.
        optimizer = torch.optim.SGD(
            [ {"params": model.parameters()} ],
            lr = config.neuron.learning_rate,
            momentum = config.neuron.momentum,
        )
        mutex = Lock()

        timecheck_dicts = {bittensor.proto.RequestType.FORWARD:{}, bittensor.proto.RequestType.BACKWARD:{}}
        n_topk_peer_weights = subtensor.min_allowed_weights

        # Create our axon server and subscribe it to the network.
        if axon == None:
            self.axon = bittensor.axon(
                config = config,
                wallet = wallet,
                synapse_checks=synapse_check,
                synapse_last_hidden = self.forward_hidden_state if model.config.neuron.lasthidden else None,
                synapse_causal_lm = self.forward_casual_lm if model.config.neuron.causallm else None,
                synapse_causal_lm_next = self.forward_casual_lm_next if model.config.neuron.causallmnext else None,
                synapse_seq_2_seq = self.forward_generate if model.config.neuron.seq2seq else None ,
                blacklist = self.blacklist if not model.config.neuron.disable_blacklist else None,
                priority = self.priority if not model.config.neuron.disable_priority else None,
            ).start().serve(subtensor=subtensor)
        
        axon.optimizer_step = optimizer_step
        axon.attach_backward_callback(backward_callback)
        # Training Data
        if config.neuron.local_train:
            dataset = bittensor.dataset(config=config)
            dataset.set_data_size(10, 64)
            data = next(dataset)

        # load our old model
        if not config.neuron.restart :
            model.load(config.neuron.full_path)

        if config.wandb.api_key != 'default':
            # --- Init Wandb.
            bittensor.wandb(
                config = config,
                cold_pubkey = wallet.coldkeypub.ss58_address,
                hot_pubkey = wallet.hotkey.ss58_address,
                root_dir = config.neuron.full_path
            )

        last_set_block = subtensor.get_current_block()
        blocks_per_epoch = subtensor.blocks_per_epoch if config.neuron.blocks_per_epoch == -1 else config.neuron.blocks_per_epoch
        blocks_per_set_weights = subtensor.blocks_per_epoch if config.neuron.blocks_per_set_weights == -1 else config.neuron.blocks_per_set_weights

        # --- Run Forever.
        while True:
            
            iteration = 0
            local_data = {}
            nn = subtensor.neuron_for_pubkey(wallet.hotkey.ss58_address)
            uid = metagraph.hotkeys.index( wallet.hotkey.ss58_address )
            current_block = subtensor.get_current_block()
            end_block = current_block + config.neuron.blocks_per_epoch
            if config.neuron.local_train:
                # --- Training step.
                while end_block >= current_block:
                    if current_block != subtensor.get_current_block():
                        loss, _ = model( next( dataset ).to(model.device) )
                        if iteration > 0 : 
                            losses += loss
                        else:
                            losses = loss
                        iteration += 1
                        current_block = subtensor.get_current_block()
                        logger.info(f'local training\titeration: {iteration}\tloss: {loss}')
                
                if iteration != 0:
                    (losses/iteration).backward()
            
            else:
                while end_block >= current_block:
                    time.sleep(12)
                    current_block = subtensor.get_current_block()


            # --- Update parameters
            if (config.neuron.local_train and iteration > 0) or (config.neuron.remote_train and model.backward_gradients_count > 0):
                # Custom learning rate
                if model.backward_gradients_count > 0:
                    optimizer.param_groups[0]['lr'] =  0.1/(model.backward_gradients_count)
                else:
                    optimizer.param_groups[0]['lr'] =  0.1

                logger.info('Backpropagation Started')
                clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                model.backward_gradients = 0
                logger.info('Backpropagation Successful: Model updated')
                local_data = {'local/loss': losses.detach().item() / iteration}

                if local_data['local/loss'] < model.best_loss:
                    model.best_loss = local_data['local/loss']
                    model.save(config.neuron.full_path)

            wandb_data = {            
                'stake': nn.stake,
                'rank': nn.rank,
                'trust': nn.trust,
                'consensus': nn.consensus,
                'incentive': nn.incentive,
                'emission': nn.emission,
            }
            
            if config.wandb.api_key != 'default':

                df = pandas.concat( [
                    bittensor.utils.indexed_values_to_dataframe( prefix = 'w_i_{}'.format(nn.uid), index = metagraph.uids, values = metagraph.W[:, uid] ),
                    axon.to_dataframe( metagraph = metagraph ),
                ], axis = 1)
                df['uid'] = df.index
                wandb_info_axon = axon.to_wandb()                
                wandb.log( { **wandb_data, **wandb_info_axon, **local_data }, step = current_block )
                wandb.log( { 'stats': wandb.Table( dataframe = df ) }, step = current_block )

            if current_block - last_set_block > blocks_per_set_weights:
                try: 
                    bittensor.__console__.print('[green]Current Status:[/green]', {**wandb_data, **local_data})

                    last_set_block = current_block
                    # Set self weights to maintain activity.
                    # --- query the chain for the most current number of peers on the network
                    chain_weights = torch.zeros(subtensor.n)
                    chain_weights [ uid ] = 1 
                    did_set = subtensor.set_weights(
                        uids=torch.arange(0,subtensor.n),
                        weights = chain_weights,
                        wait_for_inclusion = False,
                        wallet = wallet,
                    )
                    
                    metagraph.sync()
                    if did_set:
                        logger.success('Successfully set weights on the chain')
                    else:
                        logger.error('Failed to set weights on chain. (Timeout)')
                except Exception as e:
                    logger.error('Failure setting weights on chain with error: {}', e)


if __name__ == '__main__':
    module = BenchmarkModule.deploy(actor=False)

    module.sync(force_sync=True)
    # st.write(module.synapses)
    # module.run()

#!/bin/python3
# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
""" The Exodus base client.

Example:
    $ python miners/text/template_client.py

"""
import bittensor
import sys
import time
import datetime
from threading import Lock
from datetime import datetime,timedelta
from loguru import logger; logger = logger.opt(colors=True)
from torch.nn.utils.rnn import pad_sequence

import wandb
import pandas
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
