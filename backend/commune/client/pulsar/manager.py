
import argparse
import csv
import os
import json
import pulsar
import types
from time import sleep
from commune.ray.actor import ActorBase


PULSAR_IP = os.getenv("PULSAR_IP", "pulsar")
PULSAR_PORT = os.getenv("PULSAR_PORT", "6650")
PULSAR_TOPIC = os.getenv("PULSAR_TOPIC", "commune")

class PulsarManager(ActorBase):
    default_cfg_path = f"{os.path.dirname(__file__)}/manager.yaml"
    def __init__(self, cfg={}):

        # self.client_kwargs = kwargs
        # self.client = Minio(**kwargs)
        self.ip = cfg.get('ip', IP)
        self.port = cfg.get('port', PULSAR_PORT)
        self.topic = cfg.get('topic', PULSAR_TOPIC)

    @property
    def client(self):
        self._client = self.get_client(default=True)

    def get_client(self, default=True):
        if default:
            if  not hasattr(self, '_client'):
                self._client = pulsar.Client("pulsar://" + str(ip) + ":" + str(port))
            client = self._client 
        else:
            client = pulsar.Client("pulsar://" + str(ip) + ":" + str(port))
    
        return client

    def get_producer(self,topic=None):
        return self.client.create_producer(topic)
        

    def read(self, topic, *args, **kwargs):
        self.get(topic=topic, *args, **kwargs)
    def write(self, data, topic, *args, **kwargs):
        self.send(x=data, topic=topic, *args, **kwargs)


    def get(topic=None):

        if topic is None:
            topic = self.topic
        
        consumer = self.client.subscribe(
            topic, "backend-subscription", consumer_type=pulsar.ConsumerType.Shared
        )
        while True:
            msg = consumer.receive()
            message = json.loads(msg.data().decode("utf8"))
            log.info(message)
            try:
                consumer.acknowledge(msg)
            except Exception as error:
                log.info(f"`{message}`, {repr(error)}")
                consumer.negative_acknowledge(msg)
                self.client.close()


    def send(self, topic=None, x =None, stream_delay=2.0):
        if topic is None:
            topic = self.topic

        producer = self.get_producer(topic=topic)

        if isinstance(x, types.Generator):
            msg_generator = x()
            while True:
                try:
                    msg = next(msg_generator)
                    self._producer_send(msg=msg, producer=producer) 
                except Exception as e:
                    print(f"Error: {e}")
        else:
            self._producer_send(msg=x, producer=producer)

    @staticmethod
    def _producer_send(msg, producer, msg_type=dict):
        if isinstance(msg, dict):
            producer.send(msg.encode("utf8"))
            producer.flush()
        else:
            raise NotImplementedError(f'{type(x)} is not implemeneted')


if __name__ == '__main__':
    pass