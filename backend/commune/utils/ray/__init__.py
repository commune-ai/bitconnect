import ray
import torch




# def limit_parallel_jobs()

def limit_parallel_wait(running_job_handles,
                        parallel_limit=2,
                        remote_fn=None,
                        fn_kwargs={}):
    finished_job_handles = []
    while True:
        if len(running_job_handles) > 0:
            finished_job_handles, running_job_handles = ray.wait(running_job_handles)
        else:
            finished_job_handles, running_job_handles = [], []

        if len(running_job_handles) <=parallel_limit:
            running_job_handles.append(remote_fn.remote(**fn_kwargs))
            break
    return finished_job_handles, running_job_handles


def kill_actor(actor_name, verbose=True):
    try:
        ray.kill(actor_name)
        if verbose:
            print(f'{actor_name} deleted')
    except ValueError:
        if verbose:
            print(f'{actor_name} does not exist')


def actor_exists(actor_name):
    try:
        ray.get_actor(actor_name)
    except ValueError:
        return False


def create_actor(cls,
                 actor_name,
                 actor_kwargs,
                 detached=True,
                 resources={'num_cpus': 1, 'num_gpus': 0},
                 max_concurrency=1,
                 refresh=True,
                 return_actor_handle=False,
                 verbose = True):
        '''
          params:
              config: configuration of the experiment
              run_dag: run the dag
              token_pairs: token pairs
              resources: resources per actor
              actor_prefix: prefix for the data actors
          '''
        if not torch.cuda.is_available() and 'num_gpus' in resources:
            del resources['num_gpus']

        # configure the option_kwargs

        options_kwargs = {'name': actor_name,
                          'max_concurrency': max_concurrency,
                           **resources}
        if detached:
            options_kwargs['lifetime'] = 'detached'

        # setup class init config

        try:
            # refresh the actor by killing it and starting it (assuming they have the same name)
            if refresh:
                if actor_exists(actor_name):
                    kill_actor(actor_name=actor_name,
                                      verbose=verbose)

            idx = 0
            while actor_exists(actor_name):
                idx += 1
                options_kwargs['name'] = f"{actor_name}.{idx}"
                
            actor_class = ray.remote(cls)
            actor_handle = actor_class.options(**options_kwargs).remote(**actor_kwargs)

            if verbose:
                print(f"{actor_name} actor was sucsessfully created")


            if return_actor_handle:
                actor_handle = ray.get_actor(actor_name)

        except Exception as e :
            print(e)
        finally:
            if return_actor_handle:
                return actor_handle