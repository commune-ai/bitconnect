#!/usr/bin/env bash
#
# Copyright (c) 2020 Ocean Protocol contributors
# SPDX-License-Identifier: Apache-2.0
#
# Usage: ./start_ocean.sh
#
export DOCKER_BUILDKIT=0
export COMPOSE_DOCKER_CLI_BUILD=0

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



function check_if_owned_by_root {
    if [ -d "$OCEAN_HOME" ]; then
        uid=$(ls -nd "$OCEAN_HOME" | awk '{print $3;}')
        if [ "$uid" = "0" ]; then
            printf $COLOR_R"WARN: $OCEAN_HOME is owned by root\n"$COLOR_RESET >&2
        else
            uid=$(ls -nd "$OCEAN_ARTIFACTS_FOLDER" | awk '{print $3;}')
            if [ "$uid" = "0" ]; then
                printf $COLOR_R"WARN: $OCEAN_ARTIFACTS_FOLDER is owned by root\n"$COLOR_RESET >&2
            fi
        fi
    fi
}


function check_max_map_count {
  vm_max_map_count=$(docker run --rm busybox sysctl -q vm.max_map_count)
  vm_max_map_count=${vm_max_map_count##* }
  vm_max_map_count=262144
  if [ $vm_max_map_count -lt 262144 ]; then
    printf $COLOR_R'vm.max_map_count current kernel value ($vm_max_map_count) is too low for Elasticsearch\n'$COLOR_RESET
    printf $COLOR_R'You must update vm.max_map_count to at least 262144\n'$COLOR_RESET
    printf $COLOR_R'Please refer to https://www.elastic.co/guide/en/elasticsearch/reference/6.6/vm-max-map-count.html\n'$COLOR_RESET
    exit 1
  fi
}

check_if_owned_by_root

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
        COMPOSE_FILES+=" -f subtensor/subtensor.yml"

        ;;

        --backend)
        COMPOSE_FILES+=" -f backend/backend.yml"
        
        ;;

        --frontend)
        COMPOSE_FILES+=" -f ModuleFlow/frontend.yml"

        ;;

        --ipfs)
        COMPOSE_FILES+=" -f ipfs/ipfs.yml"
        
        ;;

        --all)
        COMPOSE_FILES+=" -f backend/backend.yml"
        COMPOSE_FILES+=" -f ipfs/ipfs.yml"
        COMPOSE_FILES+=" -f subtensor/subtensor.yml"
        
        ;;
        --pull)
        FORCEPULL="true"
        
        ;;

        --build)
        FORCEBUILD="true"
        
        ;;

        --down)
            printf $COLOR_R'Doing a deep clean ...\n\n'$COLOR_RESET
            eval docker network rm ${PROJECT_NAME}_default || true;
            eval docker-compose --project-name=$PROJECT_NAME "$COMPOSE_FILES" down;
            break;
        ;;

        
        --restart)
            printf $COLOR_R'Doing a deep clean ...\n\n'$COLOR_RESET
            eval docker-compose --project-name=$PROJECT_NAME "$COMPOSE_FILES" down;
            docker network rm ${PROJECT_NAME}_default || true;

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
            eval docker-compose "$DOCKER_COMPOSE_EXTRA_OPTS" --project-name=$PROJECT_NAME  "$COMPOSE_FILES" up --remove-orphans -d
            break
    esac
    shift
done



