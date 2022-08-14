
from minio import Minio
import os, sys
sys.path[0] = os.getcwd()
from commune.client.minio.manager import MinioManager
from commune.process import BaseProcess


class Module(BaseProcess):
    default_cfg_path = f'{os.getcwd()}/commune/client/minio/create_bucket.yaml'
    def process(self, **kwargs):
        for bucket_init_kwargs in self.cfg['buckets']:
            self.client['minio'].get_minio_bucket(**bucket_init_kwargs)



if __name__ == '__main__':
    Module.deploy(actor=False).run()



