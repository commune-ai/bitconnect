""" Implementation for the dataset and GenesisTextDataset class, which handles dataloading from ipfs
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

import json
import os
import random
import time
from typing import Union

import sys 
sys.path.append(os.getenv('PWD'))

from commune import Module
import requests
import torch
from loguru import logger
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from torch.utils.data.dataloader import DataLoader
import bittensor
from commune.bittensor.dataset.thread_queue import ThreadQueue
logger = logger.opt(colors=True)

class Dataset(Module):
    """ Implementation for the dataset class, which handles dataloading from ipfs
    """
    def __init__(self, config=None):
        Module.__init__(self, config=config)
        # Used to retrieve directory contentx
        self.dataset_dir = 'http://global.ipfs.opentensor.ai/api/v0/cat' 
        self.text_dir = 'http://global.ipfs.opentensor.ai/api/v0/object/get'
        self.mountain_hash = 'QmSdDg6V9dgpdAFtActs75Qfc36qJtm9y8a7yrQ1rHm7ZX'
        # Used when current corpus has been exhausted
        self.refresh_corpus = False

    @staticmethod
    def requests_retry_session(
            retries=1,
            backoff_factor=0.5,
            status_forcelist=(104, 500, 502, 504),
            session=None,
        ):
        """ Creates a retriable session for request calls. This enables
        automatic retries and back-off retries should any request calls fail.

        Args:
            retries (int, optional): Maximum number of retries. Defaults to 3.
            backoff_factor (float, optional): Factor by which to back off if a retry fails. Defaults to 0.3.
            status_forcelist (tuple, optional): A set of integer HTTP status codes that we should force a retry on. Defaults to (500, 502, 504).
            session ([type], optional): Session for which to set up the retries. Defaults to None.

        Returns:
            requests.Session(): A Requests Session object set up for retries and backoff.
        """
        session = session or requests.Session()
        retry = Retry(
            total=retries,
            read=retries,
            connect=retries,
            backoff_factor=backoff_factor,
            status_forcelist=status_forcelist,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def get_ipfs_directory(self, address: str, file_meta: dict, action: str = 'post', timeout : int = 180):
        r"""Connects to IPFS gateway and retrieves directory.
        Args:
            address: (:type:`str`, required):
                The target address of the request. 
            params: (:type:`tuple`, optional):
                The arguments of the request. eg. (('arg', dataset_hash),)
            action: (:type:`str`, optional):
                POST or GET.
            timeout: (:type:`int`, optional):
                Timeout for getting the server's response. 
        Returns:
            dict: A dictionary of the files inside of the genesis_datasets and their hashes.
        """
        session = requests.Session()
        session.params.update((('arg', file_meta['Hash']), ))
        
        try:
            if action == 'get':
                response = self.requests_retry_session(session=session).get(address, timeout=timeout)
            elif action == 'post':
                response = self.requests_retry_session(session=session).post(address, timeout=timeout)
            logger.success("Loaded from IPFS:".ljust(20) + "<blue>{}</blue>".format(file_meta['Name']))

        except Exception as E:
            logger.error(f"Failed to get from IPFS {file_meta['Name']} {E}")
            return None

        return response

    def __len__(self):
        """ Returns length of the dataset that the dataset is processing
        """

    def __getitem__(self, idx):
        """ Returns the next batch from the dataset.
        """
