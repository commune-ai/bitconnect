#
import os, sys
sys.path[0] = os.getcwd()
from commune.client.pulsar import PulsarManager

if __name__ == "__main__":
    client = PulsarManager()