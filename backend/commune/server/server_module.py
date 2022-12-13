import argparse
import os
import copy
import inspect
import time
from concurrent import futures
from typing import Dict, List, Callable, Optional, Tuple, Union

import torch
import grpc
from substrateinterface import Keypair

import bittensor


class Server:
    """ The factory class for bittensor.Axon object
    The Axon is a grpc server for the bittensor network which opens up communication between it and other neurons.
    The server protocol is defined in bittensor.proto and describes the manner in which forward and backwards requests
    are transported / encoded between validators and servers
    
    Examples:: 
            >>> config = bittensor.Server.config()
            >>> axon = bittensor.axon( config = config )
            >>> subtensor = bittensor.subtensor( network = 'nakamoto' )
            >>> Server.serve( subtensor = subtensor )
    """

    def __new__(
            cls,
            config: Optional['bittensor.config'] = None,
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
        ) -> 'bittensor.Axon':
        r""" Creates a new bittensor.Axon object from passed arguments.
            Args:
                config (:obj:`Optional[bittensor.Config]`, `optional`): 
                    bittensor.Server.config()
                wallet (:obj:`Optional[bittensor.Wallet]`, `optional`):
                    bittensor wallet with hotkey and coldkeypub.
                forward_text (:obj:`Optional[callable]`, `optional`):
                    function which is called on forward text requests.
                backward_text (:obj:`Optional[callable]`, `optional`):
                    function which is called on backward text requests.
                synapse_last_hidden (:obj:`Optional[callable]`, `optional`):
                    function which is called by the last hidden synapse
                synapse_causal_lm (:obj:`Optional[callable]`, `optional`):
                    function which is called by the causal lm synapse
                synapse_causal_lm_next (:obj:`Optional[callable]`, `optional`):
                    function which is called by the TextCausalLMNext synapse
                synapse_seq_2_seq (:obj:`Optional[callable]`, `optional`):
                    function which is called by the seq2seq synapse   
                synapse_checks (:obj:`Optional[callable]`, 'optional'):
                    function which is called before each synapse to check for stake        
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

        if config == None: 
            config = Server.config()
        config = copy.deepcopy(config)
        config.Server.port = port if port != None else config.Server.port
        config.Server.ip = ip if ip != None else config.Server.ip
        config.Server.external_ip = external_ip if external_ip != None else config.Server.external_ip
        config.Server.external_port = external_port if external_port != None else config.Server.external_port
        config.Server.max_workers = max_workers if max_workers != None else config.Server.max_workers
        config.Server.maximum_concurrent_rpcs = maximum_concurrent_rpcs if maximum_concurrent_rpcs != None else config.Server.maximum_concurrent_rpcs
        config.Server.compression = compression if compression != None else config.Server.compression

        Server.check_config( config )

        # Determine the grpc compression algorithm
        if config.Server.compression == 'gzip':
            compress_alg = grpc.Compression.Gzip
        elif config.Server.compression == 'deflate':
            compress_alg = grpc.Compression.Deflate
        else:
            compress_alg = grpc.Compression.NoCompression

        if wallet == None:
            wallet = bittensor.wallet( config = config )

        if thread_pool == None:
            thread_pool = futures.ThreadPoolExecutor( max_workers = config.Server.max_workers )
        if server == None:
            server = grpc.server( thread_pool,
                                  interceptors=(AuthInterceptor(blacklist=blacklist),),
                                  maximum_concurrent_rpcs = config.Server.maximum_concurrent_rpcs,
                                  options = [('grpc.keepalive_time_ms', 100000),
                                             ('grpc.keepalive_timeout_ms', 500000)]
                                )


        if module == None:
            module = AxonModule()
        axon_instance = axon_general_impl.AxonGeneral(
            wallet = wallet, 
            server = server,
            module = module,
            ip = config.Server.ip,
            port = config.Server.port,
            external_ip=config.Server.external_ip, # don't use internal ip if it is None, we will try to find it later
            external_port=config.Server.external_port or config.Server.port, # default to internal port if external port is not set
            timeout = timeout,
        )
        bittensor.grpc.add_BittensorServicer_to_server( self, server )
        full_address = str( config.Server.ip ) + ":" + str( config.Server.port )
        server.add_insecure_port( full_address )
        return axon_instance 

    @classmethod   
    def config(cls) -> 'bittensor.Config':
        """ Get config from the argument parser
        Return: bittensor.config object
        """
        parser = argparse.ArgumentParser()
        Server.add_args( parser )
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
            parser.add_argument('--' + prefix_str + 'Server.port', type=int, 
                    help='''The local port this axon endpoint is bound to. i.e. 8091''', default = bittensor.defaults.Server.port)
            parser.add_argument('--' + prefix_str + 'Server.ip', type=str, 
                help='''The local ip this axon binds to. ie. [::]''', default = bittensor.defaults.Server.ip)
            parser.add_argument('--' + prefix_str + 'Server.external_port', type=int, required=False,
                    help='''The public port this axon broadcasts to the network. i.e. 8091''', default = bittensor.defaults.Server.external_port)
            parser.add_argument('--' + prefix_str + 'Server.external_ip', type=str, required=False,
                help='''The external ip this axon broadcasts to the network to. ie. [::]''', default = bittensor.defaults.Server.external_ip)
            parser.add_argument('--' + prefix_str + 'Server.max_workers', type=int, 
                help='''The maximum number connection handler threads working simultaneously on this endpoint. 
                        The grpc server distributes new worker threads to service requests up to this number.''', default = bittensor.defaults.Server.max_workers)
            parser.add_argument('--' + prefix_str + 'Server.maximum_concurrent_rpcs', type=int, 
                help='''Maximum number of allowed active connections''',  default = bittensor.defaults.Server.maximum_concurrent_rpcs)
            parser.add_argument('--' + prefix_str + 'Server.backward_timeout', type=int,
                help='Number of seconds to wait for backward axon request', default=2*bittensor.__blocktime__)
            parser.add_argument('--' + prefix_str + 'Server.forward_timeout', type=int,
                help='Number of seconds to wait for forward axon request', default=5*bittensor.__blocktime__)
            parser.add_argument('--' + prefix_str + 'Server.priority.max_workers', type = int,
                help='''maximum number of threads in thread pool''', default = bittensor.defaults.Server.priority.max_workers)
            parser.add_argument('--' + prefix_str + 'Server.priority.maxsize', type=int, 
                help='''maximum size of tasks in priority queue''', default = bittensor.defaults.Server.priority.maxsize)
            parser.add_argument('--' + prefix_str + 'Server.compression', type=str, 
                help='''Which compression algorithm to use for compression (gzip, deflate, NoCompression) ''', default = bittensor.defaults.Server.compression)
            parser.add_argument('--' +  prefix_str + 'Server.lasthidden_timeout', type = int, 
            help='Timeout for last hidden synapse', default= bittensor.__blocktime__)
            parser.add_argument('--' +  prefix_str + 'Server.causallm_timeout', type = int, 
            help='Timeout for causallm synapse', default= bittensor.__blocktime__)
            parser.add_argument('--' +  prefix_str + 'Server.causallmnext_timeout', type = int, 
            help='Timeout for causallmnext synapse', default= bittensor.__blocktime__)
            parser.add_argument('--' +  prefix_str + 'Server.seq2seq_timeout', type = int, 
            help='Timeout for seq2seq synapse', default= 3*bittensor.__blocktime__)
            parser.add_argument('--' + prefix_str + 'Server.prometheus.level', 
                required = False, 
                type = str, 
                choices = [l.name for l in list(bittensor.prometheus.level)], 
                default = bittensor.defaults.Server.prometheus.level, 
                help = '''Prometheus logging level Server. <OFF | INFO | DEBUG>''')
        except argparse.ArgumentError:
            # re-parsing arguments.
            pass

        bittensor.wallet.add_args( parser, prefix = prefix )

    @classmethod   
    def add_defaults(cls, defaults):
        """ Adds parser defaults to object from enviroment variables.
        """
        defaults.axon = bittensor.Config()
        defaults.Server.port = os.getenv('BT_AXON_PORT') if os.getenv('BT_AXON_PORT') != None else 8091
        defaults.Server.ip = os.getenv('BT_AXON_IP') if os.getenv('BT_AXON_IP') != None else '[::]'
        defaults.Server.external_port = os.getenv('BT_AXON_EXTERNAL_PORT') if os.getenv('BT_AXON_EXTERNAL_PORT') != None else None
        defaults.Server.external_ip = os.getenv('BT_AXON_EXTERNAL_IP') if os.getenv('BT_AXON_EXTERNAL_IP') != None else None
        defaults.Server.max_workers = os.getenv('BT_AXON_MAX_WORERS') if os.getenv('BT_AXON_MAX_WORERS') != None else 10
        defaults.Server.maximum_concurrent_rpcs = os.getenv('BT_AXON_MAXIMUM_CONCURRENT_RPCS') if os.getenv('BT_AXON_MAXIMUM_CONCURRENT_RPCS') != None else 400
        
        defaults.Server.priority = bittensor.Config()
        defaults.Server.priority.max_workers = os.getenv('BT_AXON_PRIORITY_MAX_WORKERS') if os.getenv('BT_AXON_PRIORITY_MAX_WORKERS') != None else 10
        defaults.Server.priority.maxsize = os.getenv('BT_AXON_PRIORITY_MAXSIZE') if os.getenv('BT_AXON_PRIORITY_MAXSIZE') != None else -1

        defaults.Server.compression = 'NoCompression'

    @classmethod   
    def check_config(cls, config: 'bittensor.Config' ):
        """ Check config for axon port and wallet
        """
        assert config.Server.port > 1024 and config.Server.port < 65535, 'port must be in range [1024, 65535]'
        assert config.Server.external_port is None or (config.Server.external_port > 1024 and config.Server.external_port < 65535), 'external port must be in range [1024, 65535]'
        bittensor.wallet.check_config( config )
