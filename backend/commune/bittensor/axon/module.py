""" Implementation of Axon, services Forward and Backward requests from other neurons.
"""
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

import sys
import time as clock
from types import SimpleNamespace
from typing import List, Tuple, Callable

import torch
import grpc
import wandb
import pandas
from loguru import logger
import torch.nn.functional as F
import concurrent

import bittensor
import bittensor.utils.stats as stat_utils

from datetime import datetime

from commune import BaseModule
from .auth import AuthInterceptor

logger = logger.opt(colors=True)

class Axon( bittensor.grpc.BittensorServicer , BaseModule):
    r""" Services Forward and Backward requests from other neurons.
    """
    def __new__(
            cls, 
            config: 'bittensor.config' = None,
            wallet: 'bittensor.Wallet' = None,
            forward_text: 'Callable' = None,
            backward_text: 'Callable' = None,
            synapse_last_hidden: 'Callable' = None,
            synapse_causal_lm: 'Callable' = None,
            synapse_causal_lm_next: 'Callable' = None,
            synapse_seq_2_seq: 'Callable' = None,
            synapse_lasthidden_timeout: int = None,
            synapse_causallm_timeout: int = None,
            synapse_causallmnext_timeout: int = None,
            synapse_seq2seq_timeout: int = None,
            synapse_checks: 'Callable' = None,
            thread_pool: 'futures.ThreadPoolExecutor' = None,
            priority_threadpool: 'bittensor.prioritythreadpool' = None,
            server: 'grpc._Server' = None,
            port: int = None,
            ip: str = None,
            max_workers: int = None, 
            maximum_concurrent_rpcs: int = None,
            blacklist: 'Callable' = None,
            priority: 'Callable' = None,
            forward_timeout: int = None,
            backward_timeout: int = None,
            compression: str = None,
        ) -> 'bittensor.Axon':
        r""" Creates a new bittensor.Axon object from passed arguments.
            Args:
                config (:obj:`bittensor.Config`, `optional`): 
                    bittensor.axon.config()
                wallet (:obj:`bittensor.Wallet`, `optional`):
                    bittensor wallet with hotkey and coldkeypub.
                forward_text (:obj:`callable`, `optional`):
                    function which is called on forward text requests.
                backward_text (:obj:`callable`, `optional`):
                    function which is called on backward text requests.
                synapse_last_hidden (:obj:`callable`, `optional`):
                    function which is called by the last hidden synapse
                synapse_causal_lm (:obj:`callable`, `optional`):
                    function which is called by the causal lm synapse
                synapse_causal_lm_next (:obj:`callable`, `optional`):
                    function which is called by the TextCausalLMNext synapse
                synapse_seq_2_seq (:obj:`callable`, `optional`):
                    function which is called by the seq2seq synapse   
                synapse_checks (:obj:`callable`, 'optional'):
                    function which is called before each synapse to check for stake        
                thread_pool (:obj:`ThreadPoolExecutor`, `optional`):
                    Threadpool used for processing server queries.
                server (:obj:`grpc._Server`, `required`):
                    Grpc server endpoint, overrides passed threadpool.
                port (:type:`int`, `optional`):
                    Binding port.
                ip (:type:`str`, `optional`):
                    Binding ip.
                max_workers (:type:`int`, `optional`):
                    Used to create the threadpool if not passed, specifies the number of active threads servicing requests.
                maximum_concurrent_rpcs (:type:`int`, `optional`):
                    Maximum allowed concurrently processed RPCs.
                blacklist (:obj:`callable`, `optional`):
                    function to blacklist requests.
                priority (:obj:`callable`, `optional`):
                    function to assign priority on requests.
                forward_timeout (:type:`int`, `optional`):
                    timeout on the forward requests. 
                backward_timeout (:type:`int`, `optional`):
                    timeout on the backward requests.              
        """   

        if config == None: 
            config = Axon.config()
        config = copy.deepcopy(config)
        config.axon.port = port if port != None else config.axon.port
        config.axon.ip = ip if ip != None else config.axon.ip
        config.axon.max_workers = max_workers if max_workers != None else config.axon.max_workers
        config.axon.maximum_concurrent_rpcs = maximum_concurrent_rpcs if maximum_concurrent_rpcs != None else config.axon.maximum_concurrent_rpcs
        config.axon.forward_timeout = forward_timeout if forward_timeout != None else config.axon.forward_timeout
        config.axon.backward_timeout = backward_timeout if backward_timeout != None else config.axon.backward_timeout
        config.axon.compression = compression if compression != None else config.axon.compression
        config.axon.lasthidden_timeout = synapse_lasthidden_timeout if synapse_lasthidden_timeout != None else config.axon.lasthidden_timeout
        config.axon.causallm_timeout = synapse_causallm_timeout if synapse_causallm_timeout != None else config.axon.causallm_timeout
        config.axon.causallmnext_timeout = synapse_causallmnext_timeout if synapse_causallmnext_timeout is not None else config.axon.causallmnext_timeout
        config.axon.seq2seq_timeout = synapse_seq2seq_timeout if synapse_seq2seq_timeout != None else config.axon.seq2seq_timeout
        axon.check_config( config )

        # Determine the grpc compression algorithm
        if config.axon.compression == 'gzip':
            compress_alg = grpc.Compression.Gzip
        elif config.axon.compression == 'deflate':
            compress_alg = grpc.Compression.Deflate
        else:
            compress_alg = grpc.Compression.NoCompression

        if wallet == None:
            wallet = bittensor.wallet( config = config )
        if thread_pool == None:
            thread_pool = futures.ThreadPoolExecutor( max_workers = config.axon.max_workers )
        if server == None:
            server = grpc.server( thread_pool,
                                  interceptors=(AuthInterceptor(blacklist=blacklist),),
                                  maximum_concurrent_rpcs = config.axon.maximum_concurrent_rpcs,
                                  options = [('grpc.keepalive_time_ms', 100000),
                                             ('grpc.keepalive_timeout_ms', 500000)]
                                )

        synapses = {}
        synapses[bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE] = synapse_last_hidden
        synapses[bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM] = synapse_causal_lm
        synapses[bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM_NEXT] = synapse_causal_lm_next
        synapses[bittensor.proto.Synapse.SynapseType.TEXT_SEQ_2_SEQ] = synapse_seq_2_seq

        synapse_timeouts = {
            bittensor.proto.Synapse.SynapseType.TEXT_LAST_HIDDEN_STATE: config.axon.lasthidden_timeout,
            bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM: config.axon.causallm_timeout,
            bittensor.proto.Synapse.SynapseType.TEXT_CAUSAL_LM_NEXT: config.axon.causallmnext_timeout,
            bittensor.proto.Synapse.SynapseType.TEXT_SEQ_2_SEQ: config.axon.seq2seq_timeout
        }
        
        synapse_check_function = synapse_checks if synapse_checks != None else axon.default_synapse_check

        if priority != None and priority_threadpool == None:
            priority_threadpool = bittensor.prioritythreadpool(config=config)

        axon_instance = cls(
            wallet = wallet, 
            server = server,
            ip = config.axon.ip,
            port = config.axon.port,
            forward = forward_text,
            backward = backward_text,
            synapses = synapses,
            synapse_checks = synapse_check_function,
            synapse_timeouts = synapse_timeouts,
            priority = priority,
            priority_threadpool = priority_threadpool,
            forward_timeout = config.axon.forward_timeout,
            backward_timeout = config.axon.backward_timeout,
        )
        bittensor.grpc.add_BittensorServicer_to_server( axon_instance, server )
        full_address = str( config.axon.ip ) + ":" + str( config.axon.port )
        server.add_insecure_port( full_address )
        return axon_instance    

    def __init__( 
        self, 
        wallet: 'bittensor.wallet',
        ip: str,
        port: int,
        server: 'grpc._Server',
        forward: 'Callable',
        backward: 'Callable',
        synapses: dict,
        synapse_checks: 'Callable',
        synapse_timeouts: dict,
        priority:  'Callable' = None,
        priority_threadpool: 'bittensor.prioritythreadpool' = None,
        forward_timeout: int = None,
        backward_timeout: int = None,
    ):
        r""" Initializes a new Axon tensor processing endpoint.
            
            Args:
                config (:obj:`bittensor.Config`, `required`): 
                    bittensor.axon.config()
                wallet (:obj:`bittensor.wallet`, `required`):
                    bittensor wallet with hotkey and coldkeypub.
                server (:obj:`grpc._Server`, `required`):
                    Grpc server endpoint.
                forward (:obj:list of `callable`, `optional`):
                    list of functions which is called on forward requests.
                backward (:obj:list of `callable`, `optional`):
                    list of functions which is called on backward requests.
                priority (:obj:`callable`, `optional`):
                    function to assign priority on requests.
                priority_threadpool (:obj:`bittensor.prioritythreadpool`, `optional`):
                    bittensor priority_threadpool.                
        """
        # BaseModule.__init__(self, config=config)
        self.ip = ip
        self.port = port
        self.wallet = wallet
        self.server = server
        self.forward_callback = forward if forward != None else self.default_forward_callback
        self.backward_callback = backward if backward != None else self.default_backward_callback
        self.forward_timeout = forward_timeout
        self.backward_timeout = backward_timeout
        self.synapse_callbacks = synapses
        self.synapse_checks = synapse_checks
        self.synapse_timeouts = synapse_timeouts
        self.stats = self._init_stats()
        self.started = None
        self.optimizer_step = None

        # -- Priority 
        self.priority = priority 
        self.priority_threadpool= priority_threadpool
 
    def config(cls) -> 'bittensor.Config':
        """ Get config from the argument parser
        Return: bittensor.config object
        """
        parser = argparse.ArgumentParser()
        axon.add_args( parser )
        return bittensor.config( parser )
    @classmethod   
    def help(cls):
        """ Print help to stdout
        """
        parser = argparse.ArgumentParser()
        cls.add_args( parser )
        print (cls.__new__.__doc__)
        parser.print_help()
    @classmethod
    def add_args( cls, parser: argparse.ArgumentParser, prefix: str = None  ):
        """ Accept specific arguments from parser
        """
        prefix_str = '' if prefix == None else prefix + '.'
        try:
            parser.add_argument('--' + prefix_str + 'axon.port', type=int, 
                    help='''The port this axon endpoint is served on. i.e. 8091''', default = bittensor.defaults.axon.port)
            parser.add_argument('--' + prefix_str + 'axon.ip', type=str, 
                help='''The local ip this axon binds to. ie. [::]''', default = bittensor.defaults.axon.ip)
            parser.add_argument('--' + prefix_str + 'axon.max_workers', type=int, 
                help='''The maximum number connection handler threads working simultaneously on this endpoint. 
                        The grpc server distributes new worker threads to service requests up to this number.''', default = bittensor.defaults.axon.max_workers)
            parser.add_argument('--' + prefix_str + 'axon.maximum_concurrent_rpcs', type=int, 
                help='''Maximum number of allowed active connections''',  default = bittensor.defaults.axon.maximum_concurrent_rpcs)
            parser.add_argument('--' + prefix_str + 'axon.backward_timeout', type=int,
                help='Number of seconds to wait for backward axon request', default=2*bittensor.__blocktime__)
            parser.add_argument('--' + prefix_str + 'axon.forward_timeout', type=int,
                help='Number of seconds to wait for forward axon request', default=5*bittensor.__blocktime__)
            parser.add_argument('--' + prefix_str + 'axon.priority.max_workers', type = int,
                help='''maximum number of threads in thread pool''', default = bittensor.defaults.axon.priority.max_workers)
            parser.add_argument('--' + prefix_str + 'axon.priority.maxsize', type=int, 
                help='''maximum size of tasks in priority queue''', default = bittensor.defaults.axon.priority.maxsize)
            parser.add_argument('--' + prefix_str + 'axon.compression', type=str, 
                help='''Which compression algorithm to use for compression (gzip, deflate, NoCompression) ''', default = bittensor.defaults.axon.compression)
            parser.add_argument('--' +  prefix_str + 'axon.lasthidden_timeout', type = int, 
            help='Timeout for last hidden synapse', default= bittensor.__blocktime__)
            parser.add_argument('--' +  prefix_str + 'axon.causallm_timeout', type = int, 
            help='Timeout for causallm synapse', default= bittensor.__blocktime__)
            parser.add_argument('--' +  prefix_str + 'axon.causallmnext_timeout', type = int, 
            help='Timeout for causallmnext synapse', default= bittensor.__blocktime__)
            parser.add_argument('--' +  prefix_str + 'axon.seq2seq_timeout', type = int, 
            help='Timeout for seq2seq synapse', default= 3*bittensor.__blocktime__)
        except argparse.ArgumentError:
            # re-parsing arguments.
            pass

        bittensor.wallet.add_args( parser, prefix = prefix )
    @classmethod   
    def add_defaults(cls, defaults):
        """ Adds parser defaults to object from enviroment variables.
        """
        defaults.axon = bittensor.Config()
        defaults.axon.port = os.getenv('BT_AXON_PORT') if os.getenv('BT_AXON_PORT') != None else 8091
        defaults.axon.ip = os.getenv('BT_AXON_IP') if os.getenv('BT_AXON_IP') != None else '[::]'
        defaults.axon.max_workers = os.getenv('BT_AXON_MAX_WORERS') if os.getenv('BT_AXON_MAX_WORERS') != None else 10
        defaults.axon.maximum_concurrent_rpcs = os.getenv('BT_AXON_MAXIMUM_CONCURRENT_RPCS') if os.getenv('BT_AXON_MAXIMUM_CONCURRENT_RPCS') != None else 400
        
        defaults.axon.priority = bittensor.Config()
        defaults.axon.priority.max_workers = os.getenv('BT_AXON_PRIORITY_MAX_WORKERS') if os.getenv('BT_AXON_PRIORITY_MAX_WORKERS') != None else 10
        defaults.axon.priority.maxsize = os.getenv('BT_AXON_PRIORITY_MAXSIZE') if os.getenv('BT_AXON_PRIORITY_MAXSIZE') != None else -1

        defaults.axon.compression = 'NoCompression'
    @classmethod   
    def check_config(cls, config: 'bittensor.Config' ):
        """ Check config for axon port and wallet
        """
        assert config.axon.port > 1024 and config.axon.port < 65535, 'port must be in range [1024, 65535]'
        bittensor.wallet.check_config( config )
    @classmethod   
    def default_synapse_check(cls, synapse, hotkey ):
        """ default synapse check function
        """
        if len(hotkey) == bittensor.__ss58_address_length__:
            return True
        
        return False
    @staticmethod
    def check_backward_callback( backward_callback:Callable, pubkey:str = '_' ):
        """ Check and test axon backward callback function
        """
        if not inspect.ismethod(backward_callback) and not inspect.isfunction(backward_callback):
            raise ValueError('The axon backward callback must be a function with signature Callable[inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor ) -> torch.FloatTensor:, got {}'.format(backward_callback))        
        if len( inspect.signature(backward_callback).parameters) != 3:
            raise ValueError('The axon backward callback must have signature Callable[ inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor, synapses ) -> torch.FloatTensor:, got {}'.format(inspect.signature(backward_callback)))
        if 'inputs_x' not in inspect.signature(backward_callback).parameters:
            raise ValueError('The axon backward callback must have signature Callable[inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor ) -> torch.FloatTensor:, got {}'.format(inspect.signature(backward_callback)))
        if 'grads_dy' not in inspect.signature(backward_callback).parameters:
            raise ValueError('The axon backward callback must have signature Callable[inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor ) -> torch.FloatTensor:, got {}'.format(inspect.signature(backward_callback)))
    @staticmethod
    def check_forward_callback( forward_callback:Callable, synapses:list = []):
        """ Check and test axon forward callback function
        """
        if not inspect.ismethod(forward_callback) and not inspect.isfunction(forward_callback):
            raise ValueError('The axon forward callback must be a function with signature Callable[inputs_x: torch.Tensor] -> torch.FloatTensor:, got {}'.format(forward_callback))   
        if len( inspect.signature(forward_callback).parameters) != 3:
            raise ValueError('The axon forward callback must have signature Callable[ inputs_x: torch.Tensor, synapses, hotkey] -> torch.FloatTensor:, got {}'.format(inspect.signature(forward_callback)))
        if 'inputs_x' not in inspect.signature(forward_callback).parameters:
            raise ValueError('The axon forward callback must have signature Callable[ inputs_x: torch.Tensor] -> torch.FloatTensor:, got {}'.format(inspect.signature(forward_callback)))
        
        sample_input = torch.randint(0,1,(3, 3))
        forward_callback([sample_input], synapses, hotkey='')

    def __str__(self) -> str:
        return "Axon({}, {}, {}, {})".format( self.ip, self.port, self.wallet.hotkey.ss58_address, "started" if self.started else "stopped")

    def __repr__(self) -> str:
        return self.__str__()

    def Forward(self, request: bittensor.proto.TensorMessage, context: grpc.ServicerContext) -> bittensor.proto.TensorMessage:
        r""" The function called by remote GRPC Forward requests from other neurons.
            Forward is equivalent to a 'forward' pass through a neural network.
            After checking request validity, this function passes the request to the nucleus for processing.
            See :obj:`bittensor.proto.ReturnCode` for all possible return codes.
            
            Args:
                request (:obj:`bittensor.proto`, `required`): 
                    Tensor request proto.
                context (:obj:`grpc.ServicerContext`, `required`): 
                    grpc server context.
            
            Returns:
                response (bittensor.proto.TensorMessage): 
                    proto response carring the nucleus forward output or None under failure.
        """
        forward_response_tensors, code, synapses = self._forward( request )
        response = bittensor.proto.TensorMessage(
            version = bittensor.__version_as_int__, 
            hotkey = self.wallet.hotkey.ss58_address, 
            return_code = code,
            tensors = forward_response_tensors if forward_response_tensors is not None else [],
            requires_grad = request.requires_grad,
            synapses = synapses,
        )
        return response

    def Backward( self, request: bittensor.proto.TensorMessage, context: grpc.ServicerContext ) -> bittensor.proto.TensorMessage:
        r""" The function called by remote GRPC Backward requests from other neurons.
            Backward is equivalent to a 'backward' gradient descent pass through a neural network.
            After checking request validity, passes the request to the nucleus for processing.
            See :obj:`bittensor.proto.ReturnCode` for all possible return codes.
            
            Args:
                request (:obj:`bittensor.proto`, `required`): 
                    Tensor request proto.
                context (:obj:`grpc.ServicerContext`, `required`): 
                    grpc server context.
            
            Returns:
                response (:obj:`bittensor.proto.TensorMessage`): 
                    proto response carring the nucleus backward output or None under failure.
        """
        backward_response_tensors, code, synapses = self._backward( request )
        response = bittensor.proto.TensorMessage(
            version = bittensor.__version_as_int__, 
            hotkey = self.wallet.hotkey.ss58_address, 
            return_code = code,
            tensors = backward_response_tensors,
            requires_grad = request.requires_grad,
            synapses = synapses
        )
        return response

    def _forward(self, request):
        r""" Performs validity checks on the grpc request before passing the tensors to the forward queue.
            Returns the outputs and synapses from the backend forward call.
            
            Args:
                request (:obj:`bittensor.proto`, `required`): 
                    Tensor request proto.
            Returns:
                response (:obj:`bittensor.proto.Tensor, `required`): 
                    serialized tensor response from the nucleus call or None.
                code (:obj:`bittensor.proto.ReturnCode`, `required`):
                    Code from the call. This specifies if the overall function call was a success. 
                    This is separate from the synapse returns codes which relate to the individual synapse call. 
                synapses (:obj:`List[ 'bittensor.proto.Synapse' ]` of shape :obj:`(num_synapses)`, `required`):
                    Synapse wire protos with return codes from forward request.
        """
        # ===================================================================
        # ==== First deserialize synapse wire protos to instance objects ====        
        # ===================================================================
        synapses: List['bittensor.Synapse'] = []
        for synapse_wire_proto in request.synapses:
            synapses.append( bittensor.synapse.deserialize( synapse_wire_proto ) )


        # ===================================
        # ==== Init params from synapses ====        
        # ===================================
        # These items are filled through the call and the function returns 
        # when all codes are non-success or the function finishes completely.
        synapse_messages = [ "Success" for _ in synapses ]
        synapse_codes = [ bittensor.proto.ReturnCode.Success for _ in synapses ]
        synapse_inputs = [ None for _ in synapses ]
        synapse_responses = [ synapse.empty() for synapse in synapses ] # We fill nones for non success.
        synapse_is_response = [ False for _ in synapses ]
        synapse_call_times = [ 0 for _ in synapses ]
        synapse_timeout = min( [self.synapse_timeouts[s.synapse_type] for s in synapses] + [bittensor.__blocktime__] )
        start_time = clock.time()

        # ==================================================================
        # ==== Function which returns true if all codes are non success ====
        # ==================================================================
        def check_if_should_return() -> bool:
            for code in synapse_codes:
                if code == bittensor.proto.ReturnCode.Success:
                    return False
            return True


        # ==============================================================
        # ==== Function which prints all log statements per synapse ====
        # ==============================================================
        def finalize_codes_stats_and_logs( message = None):
            for index, synapse in enumerate( synapses ):
                request.synapses [ index ].return_code = synapse_codes[ index ] # Set synapse wire proto codes.
                request.synapses [ index ].message = synapse_messages[ index ] # Set synapse wire proto message
                bittensor.logging.rpc_log ( 
                    axon = True, 
                    forward = True, 
                    is_response = synapse_is_response [index], 
                    code = synapse_codes[ index ], 
                    call_time = synapse_call_times[ index ], 
                    pubkey = request.hotkey, 
                    inputs = synapse_inputs [index] , 
                    outputs = None if synapse_responses[index] == None else list( synapse_responses[index].shape ), 
                    message = synapse_messages[ index ] if message == None else message,
                    synapse = synapse.synapse_type
                )

        # ======================================
        # ==== Check Empty request ====
        # ======================================
        if len(request.tensors) == 0:
            code = bittensor.proto.ReturnCode.EmptyRequest
            message = "Forward request contains {} tensors, expected 1 tensor in the forward call".format(len(request.tensors))
            call_time = clock.time() - start_time
            synapse_codes = [code for _ in synapses ]
            synapse_call_times = [call_time for _ in synapses ]
            synapse_messages = [ message for _ in synapses ]
            finalize_codes_stats_and_logs()
            return [], code, request.synapses

        
        # ======================================
        # ==== Check request length ====
        # ======================================
        if len( request.tensors ) != len( synapses ):
            # Not enough responses per request.
            code = bittensor.proto.ReturnCode.RequestShapeException
            call_time = clock.time() - start_time
            message = "Request length doesn't match synape length."
            synapse_codes = [code for _ in synapses ]
            synapse_call_times = [call_time for _ in synapses ]
            synapse_messages = [ message for _ in synapses ]
            finalize_codes_stats_and_logs()
            return [], bittensor.proto.ReturnCode.RequestShapeException, request.synapses


        # ===================================
        # ==== Deserialize/Check inputs ====
        # ===================================
        deserialized_forward_tensors = [ None for _ in synapses]
        for index, synapse in enumerate( synapses ):
            try:
                deserialized_forward_tensors [index] = synapse.deserialize_forward_request_tensor ( request.tensors [index] )

            except ValueError as e:
                synapse_codes [index] = bittensor.proto.ReturnCode.RequestShapeException
                synapse_call_times [index] = clock.time() - start_time
                synapse_messages [index] = 'Input shape exception with error:{}'.format(str(e))

            except Exception as e:
                synapse_codes [index] = bittensor.proto.ReturnCode.RequestDeserializationException
                synapse_call_times [index] = clock.time() - start_time
                synapse_messages [index] = 'Input deserialization exception with error:{}'.format(str(e))
        # Check if the call can stop here.
        if check_if_should_return():
            finalize_codes_stats_and_logs()
            return [], synapse_codes[0] , request.synapses


        # ===================================
        # ==== Make forward calls. =========
        # ===================================
        try:
            finalize_codes_stats_and_logs()
            if self.priority != None:
                priority = self.priority( request.hotkey, inputs_x = deserialized_forward_tensors, request_type = bittensor.proto.RequestType.FORWARD )
                future = self.priority_threadpool.submit (
                    self.forward_callback,
                    inputs_x = deserialized_forward_tensors, 
                    synapses = synapses,
                    priority = priority,
                    hotkey = request.hotkey
                )
                forward_response_tensors, forward_codes, forward_messages = future.result( timeout = synapse_timeout - (clock.time() - start_time) )
            else:
                
                forward_response_tensors, forward_codes, forward_messages = self.forward_callback(
                    inputs_x = deserialized_forward_tensors,
                    synapses = synapses,
                    hotkey= request.hotkey
                )
            synapse_is_response = [ True for _ in synapses ]
            # ========================================
            # ==== Fill codes from forward calls ====
            # ========================================
            for index, synapse in enumerate(synapses):
                synapse_codes [ index ] = forward_codes [ index ]
                synapse_messages [index] = forward_messages [ index ]
        # ========================================
        # ==== Catch forward request timeouts ====
        # ========================================
        except concurrent.futures.TimeoutError:
            if self.priority != None:
                future.cancel()
            code = bittensor.proto.ReturnCode.Timeout
            call_time = clock.time() - start_time
            message = "Request reached timeout"
            synapse_codes = [code for _ in synapses ]
            synapse_call_times = [call_time for _ in synapses ]
            synapse_messages = [ message for _ in synapses ]
            finalize_codes_stats_and_logs()
            return [], bittensor.proto.ReturnCode.Timeout, request.synapses

        # ==================================
        # ==== Catch unknown exceptions ====
        # ==================================
        except Exception as e:
            code = bittensor.proto.ReturnCode.UnknownException
            call_time = clock.time() - start_time
            synapse_codes = [code for _ in synapses ]
            synapse_call_times = [call_time for _ in synapses ]
            synapse_messages = [ 'Exception on Server' for _ in synapses ]
            finalize_codes_stats_and_logs(message = str(e))
            return [], bittensor.proto.ReturnCode.UnknownException, request.synapses

        # =================================================
        # ==== Encode/serialize responses and synapses ====
        # ==================================================
        response_synapses = []
        for index, synapse in enumerate( synapses ):
            try:
                if synapse_codes[index] == bittensor.proto.ReturnCode.Success:
                    synapse_responses [ index ] = synapse.serialize_forward_response_tensor( deserialized_forward_tensors[ index ], forward_response_tensors [ index ] )
                else:
                    synapse_responses [ index ] = synapse.empty()

            except ValueError as e:
                if str(e) == 'Empty Response':
                    synapse_codes [ index ]= bittensor.proto.ReturnCode.EmptyResponse
                else:
                    synapse_codes [ index ]= bittensor.proto.ReturnCode.ResponseShapeException

                synapse_call_times [ index ] = clock.time() - start_time
                synapse_messages [index] = "Synapse response shape exception with error: {}".format( str( e ) )
                synapse_responses [ index ] = synapse.empty()

            except Exception as e:
                synapse_codes [ index ]= bittensor.proto.ReturnCode.ResponseSerializationException
                synapse_call_times [ index ] = clock.time() - start_time
                synapse_messages [index] = "Synapse response serialization exception with error: {}".format( str( e ) )
                synapse_responses [ index ] = synapse.empty()

            response_synapses.append(synapse.serialize_to_wire_proto(code = synapse_codes[index], message= synapse_messages[index] ))

            
        # Check if the call can stop here.
        if check_if_should_return():
            finalize_codes_stats_and_logs()
            return [], synapse_codes[0], request.synapses

        # =========================================================
        # ==== Set return times for successfull forward ===========
        # =========================================================
        for index, _ in enumerate( synapses ):
            if synapse_codes[index] == bittensor.proto.ReturnCode.Success:
                synapse_call_times[index] = clock.time() - start_time

        finalize_codes_stats_and_logs()
        return synapse_responses, bittensor.proto.ReturnCode.Success, response_synapses
 
    def _backward(self, request):
        r""" Performs validity checks on the grpc request before piping the request to the backend queue.
            Returns the outputs and synapses (with codes and messages from the backward call.)
            Args:
                request (:obj:`bittensor.proto`, `required`): 
                    Tensor request proto.
            Returns:
                response: (:obj:`bittensor.proto.Tensor, `required`): 
                    serialized tensor gradient responses. This is always an empty vector until gradients are allowed.
                code (:obj:`bittensor.proto.ReturnCode`, `required`):
                    Code from the call. This specifies if the overall function call was a success. 
                    This is separate from the synapse returns codes which relate to the individual synapse call. 
                synapses (:obj:`List[ 'bittensor.proto.Synapse' ]` of shape :obj:`(num_synapses)`, `required`):
                    Synapse wire protos with return codes from forward request.
        """

        # ===================================================================
        # ==== First deserialize synapse wire protos to instance objects ====        
        # ===================================================================
        synapses: List['bittensor.Synapse'] = []
        for synapse_wire_proto in request.synapses:
            synapses.append( bittensor.synapse.deserialize( synapse_wire_proto ) )

        # ===================================
        # ==== Init params from synapses ====        
        # ===================================
        # These items are filled through the call and the function returns 
        # when all codes are non-success or the function finishes completely.
        synapse_messages = [ "Success" for _ in synapses ]
        synapse_codes = [ bittensor.proto.ReturnCode.Success for _ in synapses ]
        deserialized_forward_tensors = [ None for _ in synapses ]
        deserialized_forward_gradients = [ None for _ in synapses ]
        synapse_is_response = [ False for _ in synapses ]
        synapse_call_times = [ 0 for _ in synapses ]
        start_time = clock.time()

        # ==================================================================
        # ==== Function which returns true if all codes are non success ====
        # ==================================================================
        def check_if_should_return() -> bool:
            for code in synapse_codes:
                if code == bittensor.proto.ReturnCode.Success:
                    return False
            return True

        # ==============================================================
        # ==== Function which prints all log statements per synapse ====
        # ==============================================================
        def finalize_codes_stats_and_logs():
            for index, synapse in enumerate( synapses ):
                request.synapses [ index ].return_code = synapse_codes[ index ] # Set synapse wire proto codes.
                request.synapses [ index ].message = synapse_messages[ index ] # Set synapse wire proto message
                bittensor.logging.rpc_log ( 
                    axon = True, 
                    forward = False, 
                    is_response = synapse_is_response [index], 
                    code = synapse_codes[ index ], 
                    call_time = synapse_call_times[ index ], 
                    pubkey = request.hotkey, 
                    inputs = None if deserialized_forward_gradients[index] == None else deserialized_forward_gradients[index].shape  , 
                    outputs = None, # we never return from backward. 
                    message = synapse_messages[ index ],
                    synapse = synapse.synapse_type
                )

        # ======================================
        # ==== Check Empty request ====
        # ======================================
        if len(request.tensors) == 0:
            code = bittensor.proto.ReturnCode.EmptyRequest
            message = "Empty Request"
            call_time = clock.time() - start_time
            synapse_codes = [code for _ in synapses ]
            synapse_call_times = [call_time for _ in synapses ]
            synapse_messages = [ message for _ in synapses ]
            finalize_codes_stats_and_logs()
            return [], code, request.synapses

        # ======================================
        # ==== Check Invalid request ====
        # ======================================
        if len(request.tensors) < 2:
            code = bittensor.proto.ReturnCode.InvalidRequest
            message = "Backward request contains {} tensors, expected atleast 2 tensor in the backward call".format(len(request.tensors))
            call_time = clock.time() - start_time
            synapse_codes = [code for _ in synapses ]
            synapse_call_times = [call_time for _ in synapses ]
            synapse_messages = [ message for _ in synapses ]
            finalize_codes_stats_and_logs()
            return [], code, request.synapses

        # ======================================
        # ==== Check request length ====
        # ======================================
        if len( request.tensors ) != 2 * len( synapses ): # 2 per synapse (1 input + 1 grad).
            # Not enough responses per request.
            code = bittensor.proto.ReturnCode.RequestShapeException
            call_time = clock.time() - start_time
            message = "Request length doesn't match synape length."
            synapse_codes = [code for _ in synapses ]
            synapse_call_times = [call_time for _ in synapses ]
            synapse_messages = [ message for _ in synapses ]
            finalize_codes_stats_and_logs()
            return [], code, request.synapses

        # ===================================
        # ==== Deserialize/Decode inputs ====
        # ===================================
        for index, synapse in enumerate( synapses ):
            try:
                deserialized_forward_tensors [index] = synapse.deserialize_forward_request_tensor ( request.tensors [index] )
                deserialized_forward_gradients [index] = synapse.deserialize_backward_request_gradient ( deserialized_forward_tensors [index],  request.tensors [ len( synapses ) + index ] )

            except ValueError as e:
                synapse_codes [index] = bittensor.proto.ReturnCode.RequestShapeException
                synapse_call_times [index] = clock.time() - start_time
                synapse_messages [index] = 'Input shape exception with error:{}'.format(str(e))

            except Exception as e:
                synapse_codes [index] = bittensor.proto.ReturnCode.RequestDeserializationException
                synapse_call_times [index] = clock.time() - start_time
                synapse_messages [index] = 'Input deserialization exception with error:{}'.format(str(e))
        # Check if the call can stop here.
        if check_if_should_return():
            finalize_codes_stats_and_logs()
            return [], synapse_codes[0], request.synapses


        # ===================================
        # ==== Make backward calls. =========
        # ===================================
        try:
            finalize_codes_stats_and_logs()
            synapse_is_response = [ True for _ in synapses ]
            if self.priority != None:
                # No wait on backward calls.
                priority = self.priority( request.hotkey, inputs_x = deserialized_forward_tensors, request_type = bittensor.proto.RequestType.BACKWARD )
                self.priority_threadpool.submit(
                    self.backward_callback, 
                    inputs_x = deserialized_forward_tensors, 
                    grads_dy = deserialized_forward_gradients,
                    synapses = synapses,
                    priority = priority
                )

            else:
                # Calling default
                backward_response_tensors, backward_codes, backward_messages = self.backward_callback ( deserialized_forward_tensors, deserialized_forward_gradients, synapses = synapses )
            
                # ========================================
                # ==== Fill codes from forward calls ====
                # ========================================
                for index, synapse in enumerate(synapses):
                    synapse_codes [ index ] = backward_codes [ index ]
                    synapse_messages [index] = backward_messages [ index ]

        # ========================================
        # ==== Catch backward request timeouts ====
        # ========================================
        except concurrent.futures.TimeoutError:
            code = bittensor.proto.ReturnCode.Timeout
            call_time = clock.time() - start_time
            message = "Request reached timeout"
            synapse_codes = [code for _ in synapses ]
            synapse_call_times = [call_time for _ in synapses ]
            synapse_messages = [ message for _ in synapses ]
            finalize_codes_stats_and_logs()
            return [], bittensor.proto.ReturnCode.Timeout, request.synapses

        # ==================================
        # ==== Catch unknown exceptions ====
        # ==================================
        except Exception as e:
            code = bittensor.proto.ReturnCode.UnknownException
            call_time = clock.time() - start_time
            message = str ( e )
            synapse_codes = [code for _ in synapses ]
            synapse_call_times = [call_time for _ in synapses ]
            synapse_messages = [ message for _ in synapses ]
            finalize_codes_stats_and_logs()
            return [], bittensor.proto.ReturnCode.UnknownException, request.synapses

        # Check if the call can stop here.
        if check_if_should_return():
            finalize_codes_stats_and_logs()
            return [], synapse_codes[0], request.synapses

        # ==============================
        # ==== Finalize call times =====
        # ==============================
        for index, _ in enumerate( synapses ):
            if synapse_codes[index] == bittensor.proto.ReturnCode.Success:
                synapse_call_times[index] = clock.time() - start_time

        finalize_codes_stats_and_logs()
        return [], bittensor.proto.ReturnCode.Success, request.synapses

    def default_forward_callback(self, inputs_x:torch.FloatTensor, synapses=[], hotkey = None):
        """
            The default forward callback when no callback is attached: Is used to call specific synapse functions

            Args:
                inputs_x (:obj:`torch.FloatTensor`, `required`): 
                    The inputs that will be passed to the synapse functions
                
                synapses (:obj: list of bittensor.proto.SynapseArgs, 'Optional')
                    The proto message that contains additional args for individual synapse functions

                hotkey (:obj: str of the hotkey, 'Optional')
                    The hotkey of the validator who sent the request

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
        model_output = None
        
        # --- calling attached synapses ---
        for index, synapse in enumerate(synapses):
            try:
                synapse_check =  self.synapse_checks(synapse, hotkey)

                if synapse.synapse_type in self.synapse_callbacks and self.synapse_callbacks[synapse.synapse_type] != None and synapse_check:
                    message, model_output, response_tensor = self.synapse_callbacks[synapse.synapse_type](inputs_x[index], synapse, model_output)
                    response_tensors.append(response_tensor)
                    response_codes.append(bittensor.proto.ReturnCode.Success)
                    response_messages.append('Success' if message is None else message)
                
                elif not synapse_check:
                    response_tensors.append(None)
                    response_codes.append(bittensor.proto.ReturnCode.UnknownException)
                    response_messages.append('Synapse Check Failed')

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

    def default_backward_callback(self, inputs_x:torch.FloatTensor, grads_dy:torch.FloatTensor, synapses=[] ):
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
        
        # --- calling attached synapses ---
        with torch.enable_grad() and torch.autograd.set_detect_anomaly(True):
            for index, synapse in enumerate(synapses):
                try:
                    if synapse.synapse_type in self.synapse_callbacks and self.synapse_callbacks[synapse.synapse_type] != None:
                        message, model_output, response_tensor = self.synapse_callbacks[synapse.synapse_type](inputs_x[index], synapse)
                        torch.autograd.backward (
                            tensors = [ response_tensor ],
                            grad_tensors = [ grads_dy[index] ],
                            retain_graph=True
                        )                        
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

        if self.optimizer_step != None:
            self.optimizer_step()
        
        return response_tensors, response_codes, response_messages

    def attach_forward_callback(self, forward_callback: Callable[ [str, torch.Tensor, int], torch.Tensor ]):
        """ Assigns the forward_callback.

            Returns:
                forward_callback (:callabl:`Callable[ [str, torch.Tensor, int], torch.Tensor `, `required`): 
                    Forward function called on recieving a forward request.
        """
        bittensor.axon.check_forward_callback(forward_callback)
        self.forward_callback = forward_callback

    def attach_synapse_callback(self, synapse_callback: Callable[[str, torch.Tensor, int],torch.Tensor], synapse_type ):
        """ Assigns the callback to a specific synapse.

            Args:
                synapse_callback (:callabl:`Callable[ [str, torch.Tensor, int], torch.Tensor `, `required`): 
                    function called for a specific synapse.
        """
        self.synapse_callbacks[synapse_type] = synapse_callback

    def attach_backward_callback(self, backward_callback: Callable[ [str, torch.Tensor, torch.Tensor, int], torch.Tensor ] ):
        """ Assigns the backward_callback call to this neuron.

            Returns:
                backward_callback (:callabl:`Callable[ [torch.Tensor, torch.Tensor], torch.Tensor `, `required`): 
                     Backward callback called on recieving a backward request.
        """
        bittensor.axon.check_backward_callback(backward_callback)
        self.backward_callback = backward_callback

    def __del__(self):
        r""" Called when this axon is deleted, ensures background threads shut down properly.
        """
        self.stop()

    def serve( 
            self, 
            use_upnpc: bool = False, 
            subtensor: 'bittensor.Subtensor' = None,
            network: str = None,
            chain_endpoint: str = None,
            prompt: bool = False
        ) -> 'Axon':
        r""" Subscribes this Axon servicing endpoint to the passed network using it's wallet.
            Args:
                use_upnpc (:type:bool, `optional`): 
                    If true, serves the axon attempts port forward through your router before 
                    subscribing.
                subtensor (:obj:`bittensor.Subtensor`, `optional`): 
                    Chain connection through which to serve.
                network (default='local', type=str)
                    If subtensor is not set, uses this network flag to create the subtensor connection.
                chain_endpoint (default=None, type=str)
                    Overrides the network argument if not set.
                prompt (bool):
                    If true, the call waits for confirmation from the user before proceeding.

        """   
        if subtensor == None: subtensor = bittensor.subtensor( network = network, chain_endpoint = chain_endpoint) 
        serv_success = subtensor.serve_axon( axon = self, use_upnpc = use_upnpc, prompt = prompt )
        if not serv_success:
            raise RuntimeError('Failed to serve neuron.')
        return self

    def start(self) -> 'Axon':
        r""" Starts the standalone axon GRPC server thread.
        """
        if self.server != None:
            self.server.stop( grace = 1 )  
            logger.success("Axon Stopped:".ljust(20) + "<blue>{}</blue>", self.ip + ':' + str(self.port))

        self.server.start()
        logger.success("Axon Started:".ljust(20) + "<blue>{}</blue>", self.ip + ':' + str(self.port))
        self.started = True
        return self

    def stop(self) -> 'Axon':
        r""" Stop the axon grpc server.
        """
        if self.server != None:
            self.server.stop( grace = 1 )
            logger.success("Axon Stopped:".ljust(20) + "<blue>{}</blue>", self.ip + ':' + str(self.port))
        self.started = False
        return self
    
    def check(self):
        r""" Checks axon's forward and backward callbacks 
        """
        pubkey = self.wallet.hotkey.ss58_address
        if self.forward_callback != None:
            bittensor.axon.check_forward_callback(self.forward_callback,index,pubkey)

        if self.backward_callback != None:
            bittensor.axon.check_backward_callback(backward,index,pubkey)
        return self

    def _init_stats(self):
        return SimpleNamespace(
            # Queries per second.
            qps = stat_utils.EventsPerSecondRollingAverage( 0, 0.01 ),
            # Total requests.
            total_requests = 0,
            # Total bytes recieved per second.
            total_in_bytes = 0,
            # Total bytes responded per second.
            total_out_bytes = 0,
            # Bytes recieved per second.
            avg_in_bytes_per_second = stat_utils.AmountPerSecondRollingAverage( 0, 0.01 ),
            # Bytes responded per second.
            avg_out_bytes_per_second = stat_utils.AmountPerSecondRollingAverage( 0, 0.01 ),
            # Requests per pubkey.
            requests_per_pubkey = {},
            # Success per pubkey.
            successes_per_pubkey = {},
            # Query time per pubkey.
            query_times_per_pubkey = {},
            # Queries per second per pubkey.
            qps_per_pubkey = {},
            # Codes recieved per pubkey.
            codes_per_pubkey = {},
            # Bytes recieved per pubkey.
            avg_in_bytes_per_pubkey = {},
            # Bytes sent per pubkey.
            avg_out_bytes_per_pubkey = {}
        )

    #TODO: Replace/update axon and dendrite stats 
    def update_stats_for_request(self, request, response, time, code):
        r""" Updates statistics for this request and response.
            Args:
                requests ( bittensor.proto.TensorMessage, `required`):
                    The request.
                response ( bittensor.proto.TensorMessage, `required`):
                    The response.
                time (:type:`float`, `required`):
                    Length of call in seconds.
                code (:obj:`bittensor.proto.ReturnCode, `required`)
                    Return code associated with the call i.e. Success of Timeout.
        """
        self.stats.qps.event()
        self.stats.total_requests += 1
        self.stats.total_in_bytes += sys.getsizeof(request) 
        self.stats.total_out_bytes += sys.getsizeof(response) 
        self.stats.avg_in_bytes_per_second.event( float(sys.getsizeof(request)) )
        self.stats.avg_out_bytes_per_second.event( float(sys.getsizeof(response)) )
        pubkey = request.hotkey
        if pubkey not in self.stats.requests_per_pubkey:
            self.stats.requests_per_pubkey[ pubkey ] = 0
            self.stats.successes_per_pubkey[ pubkey ] = 0
            self.stats.query_times_per_pubkey[ pubkey ] = stat_utils.AmountPerSecondRollingAverage(0, 0.05)
            self.stats.qps_per_pubkey[ pubkey ] = stat_utils.EventsPerSecondRollingAverage(0, 0.05)
            self.stats.codes_per_pubkey[ pubkey ] = dict([(k,0) for k in bittensor.proto.ReturnCode.keys()])
            self.stats.avg_in_bytes_per_pubkey[ pubkey ] = stat_utils.AmountPerSecondRollingAverage(0, 0.01)
            self.stats.avg_out_bytes_per_pubkey[ pubkey ] = stat_utils.AmountPerSecondRollingAverage(0, 0.01)

        # Add values.
        self.stats.requests_per_pubkey[ pubkey ] += 1
        self.stats.successes_per_pubkey[ pubkey ] += 1 if code == 1 else 0
        self.stats.query_times_per_pubkey[ pubkey ].event( float(time) )
        self.stats.avg_in_bytes_per_pubkey[ pubkey ].event( float(sys.getsizeof(request)) )
        self.stats.avg_out_bytes_per_pubkey[ pubkey ].event( float(sys.getsizeof(response)) )
        self.stats.qps_per_pubkey[ pubkey ].event()    
        try:
            if bittensor.proto.ReturnCode.Name( code ) in self.stats.codes_per_pubkey[ pubkey ].keys():
                self.stats.codes_per_pubkey[ pubkey ][bittensor.proto.ReturnCode.Name( code )] += 1
        except:
            pass  

    def to_dataframe ( self, metagraph ):
        r""" Return a stats info as a pandas dataframe indexed by the metagraph or pubkey if not existend.
            Args:
                metagraph: (bittensor.Metagraph):
                    Indexes the stats data using uids.
            Return:
                dataframe (:obj:`pandas.Dataframe`)
        """
        # Reindex the pubkey to uid if metagraph is present.
        try:
            index = [ metagraph.hotkeys.index(pubkey) for pubkey in self.stats.requests_per_pubkey.keys() if pubkey in metagraph.hotkeys ]
            columns = [ 'axon_n_requested', 'axon_n_success', 'axon_query_time','axon_avg_inbytes','axon_avg_outbytes', 'axon_qps' ]
            dataframe = pandas.DataFrame(columns = columns, index = index)
            for pubkey in self.stats.requests_per_pubkey.keys():
                if pubkey in metagraph.hotkeys:
                    uid = metagraph.hotkeys.index(pubkey)
                    dataframe.loc[ uid ] = pandas.Series( {
                        'axon_n_requested': int(self.stats.requests_per_pubkey[pubkey]),
                        'axon_n_success': int(self.stats.requests_per_pubkey[pubkey]),
                        'axon_query_time': float(self.stats.query_times_per_pubkey[pubkey].get()),             
                        'axon_avg_inbytes': float(self.stats.avg_in_bytes_per_pubkey[pubkey].get()),
                        'axon_avg_outbytes': float(self.stats.avg_out_bytes_per_pubkey[pubkey].get()),
                        'axon_qps': float(self.stats.qps_per_pubkey[pubkey].get())
                    } )
            dataframe['uid'] = dataframe.index
            return dataframe

        except Exception as e:
            bittensor.logging.error(prefix='failed axon.to_dataframe()', sufix=str(e))
            return pandas.DataFrame()

    def to_wandb( self ):
        r""" Return a dictionary of axon stat info for wandb logging
            Args:
                metagraph: (bittensor.Metagraph):
                    If not None, indexes the wandb data using int uids rather than string pubkeys.
            Return:
                wandb_info (:obj:`Dict`)
        """
        try:
            avg_query_time = 0.0
            for pubkey in self.stats.query_times_per_pubkey:
                avg_query_time += self.stats.query_times_per_pubkey[pubkey].get() / len( self.stats.query_times_per_pubkey )
            # ---- Axon summary for wandb
            wandb_data = {
                'axon/qps': self.stats.qps.get(),
                'axon/avg_query_time': avg_query_time,
                'axon/total_requests': self.stats.total_requests,
                'axon/total_in_bytes' : self.stats.total_in_bytes,
                'axon/total_out_bytes' : self.stats.total_out_bytes,
                'axon/avg_in_bytes_per_second' : self.stats.avg_in_bytes_per_second.get(),
                'axon/avg_out_bytes_per_second' : self.stats.avg_out_bytes_per_second.get(),
            }
            return wandb_data
        except Exception as e:
            bittensor.logging.error(prefix='failed during axon.to_wandb()', sufix=str(e))
            return {} 