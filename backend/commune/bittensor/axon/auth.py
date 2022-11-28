
import grpc

class AuthInterceptor(grpc.ServerInterceptor):
    """ Creates a new server interceptor that authenticates incoming messages from passed arguments.
    """
    def __init__(self, key:str = 'Bittensor',blacklist:List = []):
        r""" Creates a new server interceptor that authenticates incoming messages from passed arguments.
        Args:
            key (str, `optional`):
                 key for authentication header in the metadata (default= Bittensor)
            black_list (Fucntion, `optional`): 
                black list function that prevents certain pubkeys from sending messages
        """
        super().__init__()
        self._valid_metadata = ('rpc-auth-header', key)
        self.nounce_dic = {}
        self.message = 'Invalid key'
        self.blacklist = blacklist
        def deny(_, context):
            context.abort(grpc.StatusCode.UNAUTHENTICATED, self.message)

        self._deny = grpc.unary_unary_rpc_method_handler(deny)

    def intercept_service(self, continuation, handler_call_details):
        r""" Authentication between bittensor nodes. Intercepts messages and checks them
        """
        meta = handler_call_details.invocation_metadata

        try: 
            #version checking
            self.version_checking(meta)

            #signature checking
            self.signature_checking(meta)

            #blacklist checking
            self.black_list_checking(meta)

            return continuation(handler_call_details)

        except Exception as e:
            self.message = str(e)
            return self._deny

    def vertification(self,meta):
        r"""vertification of signature in metadata. Uses the pubkey and nounce
        """
        variable_length_messages = meta[1].value.split('bitxx')
        nounce = int(variable_length_messages[0])
        pubkey = variable_length_messages[1]
        message = variable_length_messages[2]
        unique_receptor_uid = variable_length_messages[3]
        _keypair = Keypair(ss58_address=pubkey)

        # Unique key that specifies the endpoint.
        endpoint_key = str(pubkey) + str(unique_receptor_uid)
        
        #checking the time of creation, compared to previous messages
        if endpoint_key in self.nounce_dic.keys():
            prev_data_time = self.nounce_dic[ endpoint_key ]
            if nounce - prev_data_time > -10:
                self.nounce_dic[ endpoint_key ] = nounce

                #decrypting the message and verify that message is correct
                verification = _keypair.verify( str(nounce) + str(pubkey) + str(unique_receptor_uid), message)
            else:
                verification = False
        else:
            self.nounce_dic[ endpoint_key ] = nounce
            verification = _keypair.verify( str( nounce ) + str(pubkey) + str(unique_receptor_uid), message)

        return verification

    def signature_checking(self,meta):
        r""" Calls the vertification of the signature and raises an error if failed
        """
        if self.vertification(meta):
            pass
        else:
            raise Exception('Incorrect Signature')

    def version_checking(self,meta):
        r""" Checks the header and version in the metadata
        """
        if meta[0] == self._valid_metadata:
            pass
        else:
            raise Exception('Incorrect Metadata format')

    def black_list_checking(self,meta):
        r"""Tries to call to blacklist function in the miner and checks if it should blacklist the pubkey 
        """
        variable_length_messages = meta[1].value.split('bitxx')
        pubkey = variable_length_messages[1]
        
        if self.blacklist == None:
            pass
        elif self.blacklist(pubkey,int(meta[3].value)):
            raise Exception('Black listed')
        else:
            pass
