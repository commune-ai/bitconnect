import os
from .utils import graphql_query
from commune.ray import ActorBase
import ray

class GraphQLManager(ActorBase):
    default_cfg_path = f"{os.environ['PWD']}/commune/config/client/block/graphql.yaml"

    def __init__(
        self,
        cfg,
        # host= 'endpoints',
        # port= 8000
    ):
        self.url = f"http://{cfg['host']}:{cfg['port']}"


    def query(self,query, url=None, return_one=False):
        if url != None:
            self.url = url

        
        output = graphql_query(url=self.url, query=query)
        if return_one:
            output = list(output.values())[0]
        
        return output


    def query_list(sef, query_list, num_actors=2, url=None):
        if url != None:
            self.url = url
        
        ray_graphql_query = ray.remote(graphql_query)
        ready_jobs = []
        for query in query_list:
            ready_jobs.append(ray_graphql_query.remote(url=self.url, query=query))
        
        finished_jobs_results  = []
        while ready_jobs:
            ready_jobs, finished_jobs = ray.wait(ready_jobs)
            finished_jobs_results.extend(ray.get(finished_jobs))

        return finished_jobs_results


