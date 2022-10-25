#
# Copyright 2022 Ocean Protocol Foundation
# SPDX-License-Identifier: Apache-2.0
#
import logging
import os
from typing import Dict, Optional, Union

from enforce_typing import enforce_types
from eth_account.datastructures import SignedMessage
from eth_account.messages import SignableMessage
from hexbytes.main import HexBytes
from web3.main import Web3

from commune import Module
from eth_account.messages import encode_defunct
from ocean_lib.integer import Integer
from ocean_lib.web3_internal.constants import ENV_MAX_GAS_PRICE, MIN_GAS_PRICE
from ocean_lib.web3_internal.utils import (
    private_key_to_address,
    private_key_to_public_key,
)

logger = logging.getLogger(__name__)
from eth_account.account import Account

class AccountModule(Module):

    """
    The AccountModule is responsible for signing transactions and messages by using an self.account's
    private key.

    The use of this AccountModule allows Ocean tools to send rawTransactions which keeps the user
    key and password safe and they are never sent outside. Another advantage of this is that
    we can interact directly with remote network nodes without having to run a local parity
    node since we only send the raw transaction hash so the user info is safe.

    Usage:
    ```python
    AccountModule = AccountModule(
        ocean.web3,
        private_key=private_key,
        block_confirmations=ocean.config.block_confirmations,
        transaction_timeout=config.transaction_timeout,
    )
    ```

    """

    _last_tx_count = dict()
    ENV_PRIVATE_KEY = 'PRIVATE_KEY'
    @enforce_types
    def __init__(
        self,
        private_key: str= 'PRIVATE_KEY',
        web3: Web3 = None,
        **kwargs
    ) -> None:
        """Initialises AccountModule object."""
        # assert private_key, "private_key is required."


        self.account = self.set_account(private_key = private_key)

    @property
    @enforce_types
    def address(self) -> str:
        return self.account.address

    def private_key(self):
        return self.account.private_key
        
    def set_account(self, private_key=None):
        private_key = os.getenv(private_key, private_key)
        assert isinstance(private_key, str), type(private_key)
        self.account = Account.from_key(private_key)
        return self.account

    def set_web3(self, web3):
        self.web3 = web3
        return self.web3

    @property
    @enforce_types
    def key(self) -> str:
        return self.private_key

    @staticmethod
    @enforce_types
    def reset_tx_count() -> None:
        AccountModule._last_tx_count = dict()

    def validate(self, address:str) -> bool:
        return self.account.address == address

    @staticmethod
    @enforce_types
    def _get_nonce(web3: Web3, address: str) -> int:
        # We cannot rely on `web3.eth.get_transaction_count` because when sending multiple
        # transactions in a row without wait in between the network may not get the chance to
        # update the transaction count for the self.account address in time.
        # So we have to manage this internally per self.account address.
        if address not in AccountModule._last_tx_count:
            AccountModule._last_tx_count[address] = web3.eth.get_transaction_count(address)
        else:
            AccountModule._last_tx_count[address] += 1

        return AccountModule._last_tx_count[address]


    @property
    def address(self):
        return self.account.address

    @enforce_types
    def sign_tx(
        self,
        tx: Dict[str, Union[int, str, bytes]],
        fixed_nonce: Optional[int] = None,
        gas_price: Optional[int] = None,
    ) -> HexBytes:
        
        if fixed_nonce is not None:
            nonce = fixed_nonce
            logger.debug(
                f"Signing transaction using a fixed nonce {fixed_nonce}, tx params are: {tx}"
            )
        else:
            nonce = AccountModule._get_nonce(self.web3, self.address)
        
        if not gas_price:
            gas_price = int(self.web3.eth.gas_price * 1.1)
            gas_price = max(gas_price, MIN_GAS_PRICE)

        max_gas_price = os.getenv(ENV_MAX_GAS_PRICE, None)
        if gas_price and max_gas_price:
            gas_price = min(gas_price, max_gas_price)

        logger.debug(
            f"`AccountModule` signing tx: sender address: {self.account.address} nonce: {nonce}, "
            f"eth.gasPrice: {self.web3.eth.gas_price}"
        )
        tx["gasPrice"] = gas_price
        tx["nonce"] = nonce
        signed_tx = self.web3.eth.account.sign_transaction(tx, self.private_key)
        logger.debug(f"Using gasPrice: {gas_price}")
        logger.debug(f"`AccountModule` signed tx is {signed_tx}")
        return signed_tx.rawTransaction

    @enforce_types
    def sign(self, msg_hash: Union[SignableMessage,str]) -> SignedMessage:
        """Sign a transaction."""
        if isinstance(msg_hash, str):
            msg_hash = encode_defunct(msg_hash)
        elif isinstance(msg_hash, SignableMessage):
            pass
        else:
            raise NotImplemented
            
        return self.account.sign_message(msg_hash)

    @property
    def public_key(self):
        return private_key_to_public_key(self.private_key)
        
    @enforce_types
    def keys_str(self) -> str:
        s = []
        s += [f"address: {self.address}"]
        if self.private_key is not None:
            s += [f"private key: {self.private_key}"]
            s += [f"public key: {self.public_key}"]
        s += [""]
        return "\n".join(s)


    hash_fn_dict = {
        'keccak': Web3.keccak
    }
    @staticmethod
    def resolve_hash_function(cls, hash_type='keccak'):
        hash_fn = AccountModule.hash_fn_dict.get(hash_type)
        assert hash_fn != None, f'hash_fn: {hash_type} is not found'
        return hash_fn

    @staticmethod
    def hash(input, hash_type='keccak',return_type='str',*args,**kwargs):
        
        hash_fn = AccountModule.resolve_hash_function(hash_type)
        hash_output = Web3.keccak(input, *args, **kwargs)
        if return_type in ['str', str, 'string']:
            hash_output = Web3.toHex(hash_output)
        elif return_type in ['hex', 'hexbytes']:
            pass
        else:
            raise NotImplementedError(return_type)
        
        return hash_output