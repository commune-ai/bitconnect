#!/bin/bash
docker exec -it backend python extract/crypto/ethereum_block.py
docker exec -it backend python commune/extract/crypto/sushiswap/.py
