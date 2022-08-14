
import argparse
import csv
import os
import json
from kafka import KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError, NoBrokersAvailable
from time import sleep


KAFKA_IP = os.getenv("KAFKA_IP", "kafka")
KAFKA_PORT = os.getenv("KAFKA_PORT", "9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "created_objects")
class KafkaManager(BaseProcess):
    default_cfg_path = f"{os.path.dirname(__file__)}/manager.yaml"
    def __init__(self, cfg={}):

        # self.client_kwargs = kwargs
        # self.client = Minio(**kwargs)
        self.ip = cfg.get('ip', KAFKA_IP)
        self.port = cfg.get('port', KAFKA_PORT)
        self.topic = cfg.get('topic', KAFKA_TOPIC)

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