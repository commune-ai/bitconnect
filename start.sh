#!/usr/bin/env bash
#
# Copyright (c) 2020 Ocean Protocol contributors
# SPDX-License-Identifier: Apache-2.0
#
# Usage: ./start_ocean.sh
#
export DOCKER_BUILDKIT=0
export COMPOSE_DOCKER_CLI_BUILD=0


# Specify the ethereum default RPC container provider

export NETWORK_RPC_HOST="172.15.0.3"
export NETWORK_RPC_PORT="8545"
export NETWORK_RPC_URL="http://"${NETWORK_RPC_HOST}:${NETWORK_RPC_PORT}
export GANACHE_PORT="8545"
export GANACHE_HOST="172.15.0.3"
export GANACHE_URL="http://"${GANACHE_HOST}:${GANACHE_PORT}

# export NETWORK_RPC_URL='https://polygon-mumbai.g.alchemy.com/v2/YtTw29fEGWDXcMKpljSM63DbOrgXgJRx'
# Use this seed on ganache to always create the same wallets
export GANACHE_MNEMONIC=${GANACHE_MNEMONIC:-"taxi music thumb unique chat sand crew more leg another off lamp"}
export WEB3_INFURA_PROJECT_ID="4b1e6d019d6644de887db1255319eff8"
export WEB3_INFURA_URL=" https://mainnet.infura.io/v3/${WEB3_INFURA_PROJECT_ID}"
export WEB3_ALCHEMY_PROJECT_ID="RrtpZjiUVoViiDEaYxhN9o6m1CSIZvlL"
export WEB3_ALCHEMY_URL="https://eth-mainnet.g.alchemy.com/v2/${WEB3_INFURA_PROJECT_ID}"
# Ocean contracts

export PRIVATE_KEY="0x8467415bb2ba7c91084d932276214b11a3dd9bdb2930fefa194b666dd8020b99"


IP="localhost"
optspec=":-:"
set -e

# Patch $DIR if spaces
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
DIR="${DIR/ /\\ }"

# Default versions of Aquarius, Provider
export COMPOSE_FILES=""
export PROJECT_NAME="wholetensor"
export FORCEPULL="false"
export FORCEBUILD="false"

# Export User UID and GID
export LOCAL_USER_ID=$(id -u)
export LOCAL_GROUP_ID=$(id -g)


# Add aquarius to /etc/hosts
# Workaround mainly for macOS


# colors
COLOR_R="\033[0;31m"    # red
COLOR_G="\033[0;32m"    # green
COLOR_Y="\033[0;33m"    # yellow
COLOR_B="\033[0;34m"    # blue
COLOR_M="\033[0;35m"    # magenta
COLOR_C="\033[0;36m"    # cyan

# reset
COLOR_RESET="\033[00m"

while :; do
    case $1 in
        #################################################
        # Cleaning switches
        #################################################
        --purge)
            printf "$COMPOSE_FILES"
            printf $COLOR_R'Doing a deep clean ...\n\n'$COLOR_RESET
            eval docker-compose --project-name=$PROJECT_NAME "$COMPOSE_FILES" down;
            docker network rm ${PROJECT_NAME}_default || true;
            docker network rm ${PROJECT_NAME}_backend || true;
            shift
            break
            ;;

        --subtensor)
        COMPOSE_FILES+=" -f subtensor/docker-compose.yml"

        ;;

        --ganache)
        COMPOSE_FILES+=" -f ganache/ganache.yml"

        ;;

        --frontend)
        COMPOSE_FILES+=" -f frontend/frontend.yml"
        ;;
        --backend)
        COMPOSE_FILES+=" -f backend/backend.yml"
        
        ;;

        --ipfs)
        COMPOSE_FILES+=" -f ipfs/ipfs.yml"
        
        ;;

        --light)
        COMPOSE_FILES=""
        COMPOSE_FILES+=" -f frontend/frontend.yml"
        COMPOSE_FILES+=" -f backend/backend.yml"
        # COMPOSE_FILES+=" -f ipfs/ipfs.yml"
        # COMPOSE_FILES+=" -f ganache/ganache.yml"
        
        ;;

        --all)
        COMPOSE_FILES=""
        COMPOSE_FILES+=" -f backend/backend.yml"
        COMPOSE_FILES+=" -f ipfs/ipfs.yml"
        COMPOSE_FILES+=" -f ganache/ganache.yml"
        COMPOSE_FILES+=" -f subtensor/docker-compose.yml"
        
        ;;
        --pull)
        FORCEPULL="true"
        
        ;;

        --build)
        FORCEBUILD="true"
        
        ;;

        --down)
            printf $COLOR_R'Doing a deep clean ...\n\n'$COLOR_RESET
            # eval docker network rm ${PROJECT_NAME}_default || true;
            eval docker-compose --project-name=$PROJECT_NAME "$COMPOSE_FILES" down;
            break;
        ;;

        
        --restart)
            printf $COLOR_R'Doing a deep clean ...\n\n'$COLOR_RESET
            eval docker-compose --project-name=$PROJECT_NAME "$COMPOSE_FILES" down;
            # docker network rm ${PROJECT_NAME}_default || true;

            # [ ${FORCEPULL} = "true" ] && eval docker-compose  --project-name=$PROJECT_NAME "$COMPOSE_FILES" pull
            # [ ${FORCEPULL} = "true" ] && eval docker-compose  --project-name=$PROJECT_NAME "$COMPOSE_FILES" build
            
            eval docker-compose "$DOCKER_COMPOSE_EXTRA_OPTS" --project-name=$PROJECT_NAME "$COMPOSE_FILES" up --remove-orphans -d
           
           
            break
            ;;

        --) # End of all options.
            shift
            break
            ;;
        -?*)
            printf $COLOR_R'WARN: Unknown option (ignored): %s\n'$COLOR_RESET "$1" >&2
            break
            ;;
        *)
            [ ${FORCEPULL} = "true" ] && eval docker-compose "$DOCKER_COMPOSE_EXTRA_OPTS" --project-name=$PROJECT_NAME "$COMPOSE_FILES" pull
            [ ${FORCEBUILD} = "true" ] && eval docker-compose "$DOCKER_COMPOSE_EXTRA_OPTS" --project-name=$PROJECT_NAME "$COMPOSE_FILES" build
            eval docker-compose "$DOCKER_COMPOSE_EXTRA_OPTS" --project-name=$PROJECT_NAME  "$COMPOSE_FILES" up -d
            break
    esac
    shift
done



