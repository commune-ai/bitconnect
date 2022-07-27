#!/bin/bash

# fetches ethereum blocks and their utc timestamps
# python commune/extract/crypto/ethereum_block.py &&

# fetches sushiswap liquidity pool data
# python commune/extract/crypto/sushiswap/multiple_pairs.py &&

# # get backtest explainable data (stord in ipfs)
python commune/validate/crypto/backtest.py &&


# # spawn market and 
python commune/contract/model/portfolio/task/spawnMultipleTraders.py

