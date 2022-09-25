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

""" Template server.

Example:
    $ import neurons
    $ neurons.text.core_server.neuron().run()
"""

import bittensor
import os

from .nucleus_impl import server
from .run import serve
from torch.nn.utils import clip_grad_norm_

class neuron:
    r"""
    Creates a bittensor neuron that specializes in the serving. The template server miner
    serves a NLP self.model from huggingface on the bittensor network. By default, the self.model does 
    not train itself and thus requires less memory to run. 

    Args: 
            self.config (:obj:`bittensor.Config`, `optional`): 
                bittensor.server.config()
            self.subtensor (:obj:bittensor.subtensor , `optional`):
                bittensor self.subtensor connection
            self.wallet (:obj:bittensor.wallet, `optional`):
                bittensor self.wallet object
            axon (:obj:bittensor.axon, `optional`):
                bittensor axon object
            self.metagraph (:obj:bittensor.metagraph, `optional`):
                bittensor self.metagraph object
            lasthidden (:obj:bool, `optional`):
                lasthidden synapse control
            causallm (:obj:bool, `optional`):
                causallm synapse control
            causallmnext (:obj:bool, `optional`):
                causallmnext synapse control
            seq2seq (:obj:bittensor.metagraph, `optional`):
                seq2seq synapse control
            synapse_list (:obj:list of int, `optional`):
                

    Examples:: 
            >>> self.subtensor = bittensor.subtensor(network='nakamoto')
            >>> server = bittensor.neuron.text.core_server.neuron(self.subtensor=self.subtensor)
            >>> server.run()
    """
    def __init__(
        self, 
        config: 'bittensor.config' = None,
        subtensor: 'bittensor.subtensor' = None,
        wallet: 'bittensor.wallet' = None,
        axon: 'bittensor.axon' = None,
        metagraph: 'bittensor.metagraph' = None,
        lasthidden = None,
        causallm = None,
        causallmnext = None,
        seq2seq = None,
        synapse_list = None,
    ):
        if config == None: config = server.config()
        config = config; 

        if synapse_list != None:
            config.neuron.lasthidden = False
            config.neuron.causallm = False
            config.neuron.causallmnext = False
            config.neuron.seq2seq = False

            if bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE in synapse_list:
                config.neuron.lasthidden = True

            if bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM in synapse_list:
                config.neuron.causallm = True
            if bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM_NEXT in synapse_list:
                self.config.neuron.causallmnext = True
            if bittensor.proto.Synapse.SynapseType.TEXT_SEQ_2_SEQ in synapse_list:
                config.neuron.seq2seq = True
        config.neuron.lasthidden = lasthidden if lasthidden != None else config.neuron.lasthidden
        config.neuron.causallm = causallm if causallm != None else config.neuron.causallm
        config.neuron.causallmnext = causallmnext if causallmnext is not None else config.neuron.causallmnext
        config.neuron.seq2seq = seq2seq if seq2seq != None else config.neuron.seq2seq

        self.check_config( config )
        bittensor.logging (
            config = config,
            logging_dir = self.config.neuron.full_path,
        )

        self.subtensor = subtensor
        self.wallet = wallet
        self.metagraph = metagraph

        self.model = self.get_model(config=config)
        self.axon = self.get_axon(axon=axon)
        self.config.to_defaults()
        self.timecheck_dicts = {bittensor.proto.RequestType.FORWARD:{}, bittensor.proto.RequestType.BACKWARD:{}}
        self.mutex = Lock()

    def register(self, **kwargs):
        self.wallet.reregister(subtensor=self.subtensor, **kwargs)

    def get_model(self, config=None):
        if config == None:
            config = self.config
        self.model = server(config = config)
        if not self.config.neuron.restart :
            self.model.load(self.config.neuron.full_path)

    def get_axon(axon=None):
                # Create our axon server and subscribe it to the network.
        if axon == None:
            axon = bittensor.axon(
                self.config = self.config,
                self.wallet = self.wallet,
                synapse_checks=self.synapse_check,
                synapse_last_hidden = self.forward_hidden_state if self.model.config.neuron.lasthidden else None,
                synapse_causal_lm = self.forward_casual_lm if self.model.config.neuron.causallm else None,
                synapse_causal_lm_next = self.forward_casual_lm_next if self.model.config.neuron.causallmnext else None,
                synapse_seq_2_seq = self.forward_generate if self.model.config.neuron.seq2seq else None ,
                blacklist = self.blacklist if not self.model.config.neuron.disable_blacklist else None,
                priority = self.priority if not self.model.config.neuron.disable_priority else None,
            ).start().serve(self.subtensor=self.subtensor)
        
        axon.optimizer_step = self.optimizer_step
        axon.attach_backward_callback(self.backward_callback)
        return axon


    run = serve
    @classmethod
    def self.config(cls):
        return server.config()

    @staticmethod
    def check_config( config: 'bittensor.Config' ):
        r""" Checks/validates the self.config namespace object.
        """
        bittensor.logging.check_config(config )
        bittensor.wallet.check_config( config )
        bittensor.subtensor.check_config( config )
        bittensor.metagraph.check_config( config )
        bittensor.dataset.check_config( config )
        bittensor.axon.check_config( config )
        bittensor.wandb.check_config( config )
        full_path = os.path.expanduser('{}/{}/{}/{}'.format( config.logging.logging_dir, self.config.wallet.get('name', bittensor.defaults.wallet.name), self.config.wallet.get('hotkey', bittensor.defaults.wallet.hotkey), self.config.neuron.name ))
        config.neuron.full_path = os.path.expanduser(full_path)
        if not os.path.exists(config.neuron.full_path):
            os.makedirs(config.neuron.full_path)


    @property
    def device(self):
        return self.device

    def forward_generate(self, inputs_x:torch.FloatTensor, synapse, model_output = None):
        tokens = self.model.token_remap(inputs_x.to(self.device))
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

    def forward_hidden_state(self, inputs_x:torch.FloatTensor, synapse, model_output = None):
        with self.mutex:
            message, model_output, hidden = self.model.encode_forward(inputs_x.to(self.device), model_output=model_output)
        return message, model_output, hidden

    def forward_casual_lm(self, inputs_x:torch.FloatTensor, synapse, model_output = None):
        with self.mutex:
            message, model_output, logits = self.model.encode_forward_causallm(inputs_x.to(self.device), model_output=model_output)
        return message, model_output, logits

    def forward_casual_lm_next(self, inputs_x: torch.FloatTensor, synapse, model_output=None):
        with self.mutex:
            message, model_output, topk_token_phrases = self.model.encode_forward_causallmnext(inputs_x.to(self.device),
                                                                                        topk=synapse.topk,
                                                                                        model_output=model_output)
        # topk_token_phrases: [sum_b(sum_k(len(phrase_k) + 1)_b)] contains topk token phrases and probabilities
        #   Compacted 1-D tensor >= batch_size * (2 * topk + 1)
        return message, model_output, topk_token_phrases


    def optimizer_step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()



    def get_uid(self, pubkey:str)
        return self.metagraph.hotkeys.index(pubkey)
    
    @property
    def pubkey(self):
        return self.wallet.hotkey.ss58_address

    @property
    def uid(self):
        return self.get_uid(pubkey)

    def registration_check(self, pubkey:str):
        # If we allow non-registered requests return False = not blacklisted.
        is_registered = pubkey in self.metagraph.hotkeys
        if not is_registered:
            if self.config.neuron.blacklist_allow_non_registered:
                
                return False
            raise Exception('Registration blacklist')

        # Check for stake

    def stake_check(self, pubkey:str) -> bool:
        # Check stake.
        uid = self.get_uid(pubkey=pub_key)
        if self.metagraph.S[uid].item() < self.config.neuron.blacklist.stake:
            raise Exception('Stake blacklist')
        return False

    # Check for time
    def time_check(self):
        current_time = datetime.now()
        # Only check if the request are forward requests
        timecheck = self.timecheck_dicts[request_type]
        if pubkey in timecheck.keys():
            prev_time = timecheck[pubkey]
            if current_time - prev_time >= timedelta(seconds=self.config.neuron.blacklist.time):
                timecheck[pubkey] = current_time
            else:
                timecheck[pubkey] = current_time
                raise Exception('Time blacklist')
        else:
            timecheck[pubkey] = current_time
    
        return False



    def blacklist(self, pubkey:str, request_type:bittensor.proto.RequestType) -> bool:
        r"""Axon security blacklisting, used to blacklist message from low stake members
            Args:
                pubkey ( str, `required`):
                    The public key of the caller.
                request_type ( bittensor.proto.RequestType, `required`):
                    the request type ('FORWARD' or 'BACKWARD').
        """

        # Black list or not
        try:
            self.registration_check()
            self.time_check()
            self.stake_check()
            return False

        except Exception as e:
            return True
    
    def synapse_check(self, synapse, hotkey):
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
        incoming_uid = self.metagraph.hotkeys.index(hotkey)
        if synapse.synapse_type == bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE:
            
            if self.metagraph.S[incoming_uid] < self.config.neuron.lasthidden_stake:
                return False
            
        elif synapse.synapse_type == bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM:

            if self.metagraph.S[incoming_uid] < self.config.neuron.causallm_stake:
                return False

        elif synapse.synapse_type == bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM_NEXT:

            if self.metagraph.S[incoming_uid] < self.config.neuron.causallmnext_stake:
                return False

        elif synapse.synapse_type == bittensor.proto.Synapse.SynapseType.TEXT_SEQ_2_SEQ:

            if (self.metagraph.S[incoming_uid] < self.config.neuron.seq2seq_stake) and (self.metagraph.S[incoming_uid,  uid]):
                return False     
        else:
            return False

        return True

    def backward_callback(self, inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor, synapses=[] ):
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
        
        if not self.config.neuron.remote_train:
            return response_tensors, response_codes, response_messages

        # --- calling attached synapses ---
        with self.mutex and torch.enable_grad() and torch.autograd.set_detect_anomaly(True):
            for index, synapse in enumerate(synapses):
                try:
                    if synapse.synapse_type in self.axon.synapse_callbacks and self.axon.synapse_callbacks[synapse.synapse_type] != None:
                        model_output, response_tensor = self.axon.synapse_callbacks[synapse.synapse_type](inputs_x[index], synapse)
                        grads_dy_norm = grads_dy[index]/(grads_dy[index].sum() + 0.00001)
                        torch.autograd.backward (
                            tensors = [ response_tensor ],
                            grad_tensors = [ grads_dy_norm ],
                            retain_graph=True
                        )                        
                        self.model.backward_gradients_count += inputs_x[index].size(0)
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
            uid = self.metagraph.hotkeys.index(pubkey)
            priority = self.metagraph.S[uid].item()
        
        except:
            # zero priority for those who are not registered.
            priority =  0

        return priority


    def serve(self):

        n_topk_peer_weights = self.subtensor.min_allowed_weights


        # Training Data
        if self.config.neuron.local_train:
            dataset = bittensor.dataset(self.config=self.config)
            dataset.set_data_size(10, 64)
            data = next(dataset)

        last_set_block = self.subtensor.get_current_block()
        blocks_per_epoch = self.subtensor.blocks_per_epoch if self.config.neuron.blocks_per_epoch == -1 else self.config.neuron.blocks_per_epoch
        blocks_per_set_weights = self.subtensor.blocks_per_epoch if self.config.neuron.blocks_per_set_weights == -1 else self.config.neuron.blocks_per_set_weights

        # --- Run Forever.
        while True:
            
            iteration = 0
            local_data = {}
            nn = self.subtensor.neuron_for_pubkey(self.wallet.hotkey.ss58_address)
            uid = self.metagraph.hotkeys.index( self.wallet.hotkey.ss58_address )
            current_block = self.subtensor.get_current_block()
            end_block = current_block + self.config.neuron.blocks_per_epoch
            if self.config.neuron.local_train:
                # --- Training step.
                while end_block >= current_block:
                    if current_block != self.subtensor.get_current_block():
                        loss, _ = self.model( next( dataset ).to(self.device) )
                        if iteration > 0 : 
                            losses += loss
                        else:
                            losses = loss
                        iteration += 1
                        current_block = self.subtensor.get_current_block()
                        logger.info(f'local training\titeration: {iteration}\tloss: {loss}')
                
                if iteration != 0:
                    (losses/iteration).backward()
            
            else:
                while end_block >= current_block:
                    time.sleep(12)
                    current_block = self.subtensor.get_current_block()


            # --- Update parameters
            if (self.config.neuron.local_train and iteration > 0) or (self.config.neuron.remote_train and self.model.backward_gradients_count > 0):
                # Custom learning rate
                if self.model.backward_gradients_count > 0:
                    self.optimizer.param_groups[0]['lr'] =  0.1/(self.model.backward_gradients_count)
                else:
                    self.optimizer.param_groups[0]['lr'] =  0.1

                logger.info('Backpropagation Started')
                clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.model.backward_gradients = 0
                logger.info('Backpropagation Successful: Model updated')
                local_data = {'local/loss': losses.detach().item() / iteration}

                if local_data['local/loss'] < self.model.best_loss:
                    self.model.best_loss = local_data['local/loss']
                    self.model.save(self.config.neuron.full_path)

            if current_block - last_set_block > blocks_per_set_weights:
                try: 
                    bittensor.__console__.print('[green]Current Status:[/green]', {**wandb_data, **local_data})

                    last_set_block = current_block
                    # Set self weights to maintain activity.
                    # --- query the chain for the most current number of peers on the network
                    chain_weights = torch.zeros(self.subtensor.n)
                    chain_weights [ uid ] = 1 
                    did_set = self.subtensor.set_weights(
                        uids=torch.arange(0,self.subtensor.n),
                        weights = chain_weights,
                        wait_for_inclusion = False,
                        wallet = self.wallet,
                    )
                    
                    self.metagraph.sync()
                    if did_set:
                        logger.success('Successfully set weights on the chain')
                    else:
                        logger.error('Failed to set weights on chain. (Timeout)')
                except Exception as e:
                    logger.error('Failure setting weights on chain with error: {}', e)
