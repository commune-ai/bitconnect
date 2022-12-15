import argparse
import os
import copy
import inspect
import time
from concurrent import futures
from typing import Dict, List, Callable, Optional, Tuple, Union
import streamlit as st
import sys
import torch
import grpc
from substrateinterface import Keypair
from loguru import logger
import sys
import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())
sys.path.append(os.getenv('PWD'))
import commune
from commune.server.server_interceptor import ServerInterceptor
from commune.proto import CommuneServicer
import bittensor


class ServerModule(CommuneServicer):
    """ The factory class for bittensor.Axon object
    The Axon is a grpc server for the bittensor network which opens up communication between it and other neurons.
    The server protocol is defined in bittensor.proto and describes the manner in which forward and backwards requests
    are transported / encoded between validators and servers
    """

    def __init__(
            self,
            config: Optional['commune.Config'] = None,
            wallet: Optional['bittensor.Wallet'] = None,
            server: Optional['grpc._Server'] = None,
            port: Optional[int] = None,
            ip: Optional[str] = None,
            module: 'AxonModule'= None,
            external_ip: Optional[str] = None,
            external_port: Optional[int] = None,
            max_workers: Optional[int] = None, 
            maximum_concurrent_rpcs: Optional[int] = None,
            blacklist: Optional['Callable'] = None,
            priority: Optional['Callable'] = None,

            thread_pool: Optional[futures.ThreadPoolExecutor] = None,
            timeout: Optional[int] = None,
            compression:Optional[str] = None,
            coldkey='bit',
            hotkey='connect'
        ) -> 'bittensor.Axon':
        r""" Creates a new bittensor.Axon object from passed arguments.
            Args:
                config (:obj:`Optional[bittensor.Config]`, `optional`): 
                    bittensor.Server.config()
                wallet (:obj:`Optional[bittensor.Wallet]`, `optional`):
                    bittensor wallet with hotkey and coldkeypub.    
                thread_pool (:obj:`Optional[ThreadPoolExecutor]`, `optional`):
                    Threadpool used for processing server queries.
                server (:obj:`Optional[grpc._Server]`, `required`):
                    Grpc server endpoint, overrides passed threadpool.
                port (:type:`Optional[int]`, `optional`):
                    Binding port.
                ip (:type:`Optional[str]`, `optional`):
                    Binding ip.
                external_ip (:type:`Optional[str]`, `optional`):
                    The external ip of the server to broadcast to the network.
                external_port (:type:`Optional[int]`, `optional`):
                    The external port of the server to broadcast to the network.
                max_workers (:type:`Optional[int]`, `optional`):
                    Used to create the threadpool if not passed, specifies the number of active threads servicing requests.
                maximum_concurrent_rpcs (:type:`Optional[int]`, `optional`):
                    Maximum allowed concurrently processed RPCs.
                blacklist (:obj:`Optional[callable]`, `optional`):
                    function to blacklist requests.
                priority (:obj:`Optional[callable]`, `optional`):
                    function to assign priority on requests.
                forward_timeout (:type:`Optional[int]`, `optional`):
                    timeout on the forward requests. 
                backward_timeout (:type:`Optional[int]`, `optional`):
                    timeout on the backward requests.              
        """ 

        self.wallet = wallet if wallet else bittensor.wallet(name=coldkey, hotkey=hotkey)  

        config = copy.deepcopy(config if config else self.default_config())
        self.port = config.port = port if port != None else config.port
        self.ip = config.ip = ip if ip != None else config.ip
        self.external_ip = config.external_ip = external_ip if external_ip != None else config.external_ip
        self.external_port = config.external_port = external_port if external_port != None else config.external_port
        self.max_workers = config.max_workers = max_workers if max_workers != None else config.max_workers
        self.maximum_concurrent_rpcs  = config.maximum_concurrent_rpcs = maximum_concurrent_rpcs if maximum_concurrent_rpcs != None else config.maximum_concurrent_rpcs
        self.compression = config.compression = compression if compression != None else config.compression
        self.timeout = timeout if timeout else config.timeout
        ServerModule.check_config( config )
        self.config = config

        # Determine the grpc compression algorithm
        if config.compression == 'gzip':
            compress_alg = grpc.Compression.Gzip
        elif config.compression == 'deflate':
            compress_alg = grpc.Compression.Deflate
        else:
            compress_alg = grpc.Compression.NoCompression
        if thread_pool == None:
            thread_pool = futures.ThreadPoolExecutor( max_workers = config.max_workers )


        if server == None:
            server = grpc.server( thread_pool,
                                  interceptors=(ServerInterceptor(blacklist=blacklist,receiver_hotkey=self.wallet.hotkey.ss58_address),),
                                  maximum_concurrent_rpcs = config.maximum_concurrent_rpcs,
                                  options = [('grpc.keepalive_time_ms', 100000),
                                             ('grpc.keepalive_timeout_ms', 500000)]
                                )
        self.server = server
        self.module = module
        self.started = False
        commune.proto.add_servicer_to_server( self, server )
        full_address = str( config.ip ) + ":" + str( config.port )
        server.add_insecure_port( full_address )

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
            parser.add_argument('--' + prefix_str + 'port', type=int, 
                    help='''The local port this axon endpoint is bound to. i.e. 8091''', default = bittensor.defaults.Server.port)
            parser.add_argument('--' + prefix_str + 'ip', type=str, 
                help='''The local ip this axon binds to. ie. [::]''', default = bittensor.defaults.Server.ip)
            parser.add_argument('--' + prefix_str + 'external_port', type=int, required=False,
                    help='''The public port this axon broadcasts to the network. i.e. 8091''', default = bittensor.defaults.Server.external_port)
            parser.add_argument('--' + prefix_str + 'external_ip', type=str, required=False,
                help='''The external ip this axon broadcasts to the network to. ie. [::]''', default = bittensor.defaults.Server.external_ip)
            parser.add_argument('--' + prefix_str + 'max_workers', type=int, 
                help='''The maximum number connection handler threads working simultaneously on this endpoint. 
                        The grpc server distributes new worker threads to service requests up to this number.''', default = bittensor.defaults.Server.max_workers)
            parser.add_argument('--' + prefix_str + 'maximum_concurrent_rpcs', type=int, 
                help='''Maximum number of allowed active connections''',  default = bittensor.defaults.Server.maximum_concurrent_rpcs)
            parser.add_argument('--' + prefix_str + 'priority.max_workers', type = int,
                help='''maximum number of threads in thread pool''', default = bittensor.defaults.Server.priority.max_workers)
            parser.add_argument('--' + prefix_str + 'priority.maxsize', type=int, 
                help='''maximum size of tasks in priority queue''', default = bittensor.defaults.Server.priority.maxsize)
            parser.add_argument('--' + prefix_str + 'compression', type=str, 
                help='''Which compression algorithm to use for compression (gzip, deflate, NoCompression) ''', default = bittensor.defaults.Server.compression)
        except argparse.ArgumentError:
            # re-parsing arguments.
            pass
        bittensor.wallet.add_args( parser, prefix = prefix )

    @classmethod
    def default_config(cls):
        config = commune.Config()
        config.port = 8091
        config.ip =  '[::]'
        config.external_port =  None
        config.external_ip =  None
        config.max_workers = 10
        config.maximum_concurrent_rpcs =  400
        config.priority = commune.Config()
        config.priority.max_workers =  10
        config.priority.maxsize =  -1
        config.compression = 'NoCompression'
        config.timeout = 10
        return config

    @classmethod   
    def check_config(cls, config: 'commune.Config' ):
        """ Check config for axon port and wallet
        """
        assert config.port > 1024 and config.port < 65535, 'port must be in range [1024, 65535]'
        assert config.external_port is None or (config.external_port > 1024 and config.external_port < 65535), 'external port must be in range [1024, 65535]'


    def __str__(self) -> str:
        return "Server({}, {}, {}, {})".format( self.ip, self.port, self.wallet.hotkey.ss58_address, "started" if self.started else "stopped")

    def __repr__(self) -> str:
        return self.__str__()

    def Forward(self, request: commune.proto.DataBlock, context: grpc.ServicerContext) -> commune.proto.DataBlock:
        r""" The function called by remote GRPC Forward requests. The Datablock is a generic formatter.
            
            Args:
                request (:obj:`bittensor.proto`, `required`): 
                    Tensor request proto.
                context (:obj:`grpc.ServicerContext`, `required`): 
                    grpc server context.
            
            Returns:
                response (commune.proto.DataBlock): 
                    proto response carring the nucleus forward output or None under failure.
        """
        forward_response_tensors, code, synapses = self.call_module( request )
        response = bittensor.proto.TensorMessage(
            version = bittensor.__version_as_int__, 
            hotkey = self.wallet.hotkey.ss58_address, 
            return_code = code,
            tensors = forward_response_tensors if forward_response_tensors is not None else [],
            requires_grad = request.requires_grad,
            synapses = synapses,
        )
        return response


    def call_module(self, request):
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
        # ==== Deserialize/Check inputs ====
        # ===================================
        deserialized_forward_tensors = [ None for _ in synapses]
        for index, synapse in enumerate( synapses ):
            try:
                deserialized_forward_tensors [index] = synapse.deserialize_forward_request_tensor ( request.tensors [index] )

            except Exception as e:
                synapse_codes [index] = bittensor.proto.ReturnCode.RequestDeserializationException
                synapse_call_times [index] = clock.time() - start_time
                synapse_messages [index] = 'Input deserialization exception with error:{}'.format(str(e))
        # Check if the call can stop here.
        if self.check_if_should_return(synapse_codes=synapse_codes):
            self.finalize_codes_stats_and_logs()
            return [], synapse_codes[0] , request.synapses


        # ===================================
        # ==== Make forward calls. =========
        # ===================================
        try:
            self.finalize_codes_stats_and_logs()

            forward_response_tensors, forward_codes, forward_messages = self.module(
                inputs = deserialized_forward_tensors,
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
            code = bittensor.proto.ReturnCode.Timeout
            call_time = clock.time() - start_time
            message = "Request reached timeout"
            synapse_codes = [code for _ in synapses ]
            synapse_call_times = [call_time for _ in synapses ]
            synapse_messages = [ message for _ in synapses ]
            self.finalize_codes_stats_and_logs()
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
            self.finalize_codes_stats_and_logs(message = str(e))
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
        if self.check_if_should_return():
            self.finalize_codes_stats_and_logs()
            return [], synapse_codes[0], request.synapses

        # =========================================================
        # ==== Set return times for successfull forward ===========
        # =========================================================
        for index, _ in enumerate( synapses ):
            if synapse_codes[index] == bittensor.proto.ReturnCode.Success:
                synapse_call_times[index] = clock.time() - start_time

        self.finalize_codes_stats_and_logs()
        return synapse_responses, bittensor.proto.ReturnCode.Success, response_synapses
 
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

    def start(self) -> 'ServerModule':
        r""" Starts the standalone axon GRPC server thread.
        """
        st.write(self.__dict__)
        if self.server != None:
            self.server.stop( grace = 1 )  
            logger.success("Axon Stopped:".ljust(20) + "<blue>{}</blue>", self.ip + ':' + str(self.port))

        self.server.start()
        logger.success("Axon Started:".ljust(20) + "<blue>{}</blue>", self.ip + ':' + str(self.port))
        self.started = True

        return self

    def stop(self) -> 'ServerModule':
        r""" Stop the axon grpc server.
        """
        if self.server != None:
            self.server.stop( grace = 1 )
            logger.success("Axon Stopped:".ljust(20) + "<blue>{}</blue>", self.ip + ':' + str(self.port))
        self.started = False

        return self
    
if __name__ == '__main__':
    module = ServerModule()
    module.start()
    st.write(module.__dict__)