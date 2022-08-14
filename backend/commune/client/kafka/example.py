#
import os, sys
sys.path[0] = os.getcwd()
from commune.client.pulsar import KafkaManager

if __name__ == "__main__":
    client = KafkaManager()