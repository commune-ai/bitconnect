


import os, sys
sys.path.append(os.environ['PWD'])
import gradio
import socket
from signal import SIGKILL
from psutil import process_iter
from commune.utils import *
from commune import Module
from copy import deepcopy
# from commune.thread import PriorityThreadPoolExecutor
import argparse
import streamlit as st

class bcolor:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
import commune

class GradioModule(commune.Module):
    def __init__(self, **kwargs):
        commune.Module.__init__(self, **kwargs)

        self.subprocess_manager = self.get_object('subprocess.module.SubprocessModule')()

        self.host  = self.config.get('host', '0.0.0.0')
        self.port  = self.config.get('port', 8000)
        self.num_ports = self.config.get('num_ports', 10)
        self.port_range = self.config.get('port_range', [7865, 7871])
        
        # self.thread_manager = PriorityThreadPoolExecutor()
        # self.process_manager = self.get_object('cliProcessManager()

    # without '__reduce__', the instance is unserializable.
    def __reduce__(self):
        deserializer = GradioModule
        serialized_data = (self.config,)
        return deserializer, serialized_data


    @property
    def active_modules(self):
        return self._modules

    @property
    def gradio_modules(self):
        return self._modules

    @property
    def module2port(self):
        module2port = {}
        for port, module in self.port2module.items():
            module2port[module] = port
        return module2port

    @property
    def port2module(self):
        port2module = {}
        for k, v in self.subprocess_map.items():
            port2module[k] = v['module']
        return port2module

    @property
    def port2module(self):
        port2module = {}
        for k, v in self.subprocess_map.items():
            port2module[k] = v['module']

        return port2module


    def rm_module(self, port:str=10, output_example={'bro': True}):
        visable.remove(current)
        return jsonify({"executed" : True,
                        "ports" : current['port']})


    @staticmethod
    def get_funcs(self):
        return [func for func in dir(self) if not func.startswith("__") and callable(getattr(self, func, None)) ]


    def active_port(self, port:int=1):
        return port in self.port2module


    def port_connected(self ,port : int):
        """
            Check if the given param port is already running
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)       
        result = s.connect_ex((self.host, int(port)))
        return result == 0

    @property
    def subprocess_map(self):
        # self.subprocess_manager.load_state()
        # for port,port_subprocess in self.subprocess_manager.subprocess_map.items():
        #     if not self.port_connected(port=port):
        #         self.subprocess_manager.rm(key=port)
        return self.subprocess_manager.subprocess_map

    def port_available(self, port:int):
        # print(self.subprocess_map)
        print(str(port) in self.subprocess_map.keys(), self.port_connected(port))
        return not str(port) in self.subprocess_map.keys() and not self.port_connected(port)
        # subprocess_map = self.subprocess_map

        
        # if str(port) in subprocess_map and self.port_connected(port):
        #     return False
        # else:
        #     return True
        
    def suggest_port(self, max_trial_count=100):

        for port in range(*self.port_range):
            if self.port_available(port):
                return port
        '''
        TODO: kill a port when they are all full
        '''
        raise Exception(f'There does not exist an open port between {self.port_range}')
        

    @property
    def simple2path(self):
        module_list = self.module_list
        simple2path = {}
        for m in module_list:
            assert m.split('.')[-2] == 'module', f'{m}'
            simple_path = '.'.join(m.split('.')[:-2])
            simple2path[simple_path] = m

        return simple2path


    def get_modules(self, force_update=True):
        modules = []
        failed_modules = []
        for root, dirs, files in os.walk('/app/commune'):
            if all([f in files for f in ['module.py', 'module.yaml']]):
                try:
                    cfg = self.config_loader.load(root)   
                    if cfg == None:
                        cfg = {}           
                except Exception as e:
                    cfg = {}

                module_path = root.lstrip(os.environ['PWD']).replace('/', '.')
                module_path = '.'.join(module_path.split('.')[1:])
                if isinstance(cfg.get('module'), str):
                    module_name = cfg.get('module').split('.')[-1]
                    modules.append(f"{module_path}.module.{module_name}")
                elif module_path == None: 
                    failed_modules.append(root)

        return modules

    @property
    def module_list(self):
        return self.get_modules()

    def get_gradio_modules(self):
        return list(self.get_module_schemas().keys())

    @staticmethod
    def get_module_function_schema(module):
        if isinstance(module,str):
            module = get_object(module)
        module_schema = get_module_function_schema(module)
        return module_schema
        
    @staticmethod
    def schema2gradio(fn_schema, return_type='dict'):
        gradio_schema = {}
        fn_example = fn_schema['example']
        gradio_schema['example'] = fn_example

        for m in ['input', 'output']:
            gradio_schema[m] = []
            for k,v in fn_example[m].items():
                v_type = type(v).__name__
                
                if v_type == 'int':
                    gradio_schema[m] += [gradio.Number(value=v, label=k)]
                elif v_type == 'str':
                    gradio_schema[m] += [gradio.Textbox(value=v, label=k)]
                elif v_type == 'bool':
                    gradio_schema[m] += [gradio.Checkbox(value=v, label=k)]
                elif v_type == 'dict':
                    gradio_schema[m] += [gradio.JSON(value=v, label=k)]
                else:
                    raise NotImplementedError(v_type)

                


        # print('GRADIO:', gradio_schema['input'][0].__dict__)
        return gradio_schema
    

    @classmethod
    def streamlit(cls):
        st.write(cls.subprocess_map)       


    def get_gradio_function_schemas(self, module, return_type='gradio'):
        if isinstance(module, str):
            module = get_object(module)
        function_defaults_dict = get_module_function_defaults(module)
        function_defaults_dict = get_full_functions(module_fn_schemas=function_defaults_dict)

        gradio_fn_schema_dict = {}

        for fn, fn_defaults in function_defaults_dict.items():
            module_fn_schema = get_function_schema(defaults_dict=fn_defaults)
            module_fn_schema['example'] = fn_defaults
            gradio_fn_schema_dict[fn] = self.schema2gradio(module_fn_schema)

            gradio_fn_list = []
            if return_type in ['json', 'dict']:
                for m in ['input', 'output']:
                    for gradio_fn in gradio_fn_schema_dict[fn][m]:
                        gradio_fn_list += [{'__dict__': gradio_fn.__dict__, 
                                            'module': f'gradio.{str(gradio_fn.__class__.__name__)}'}]
                        print('DEBUG',GradioModule.load_object(**gradio_fn_list[-1]))
                    gradio_fn_schema_dict[fn][m] =  gradio_fn_list
            elif return_type in ['gradio']:
                pass
            else:
                raise NotImplementedError


        return gradio_fn_schema_dict

    def get_module_schemas(self,filter_complete=False):
        module_schema_map = {}
        module_paths = self.get_modules()

        for module_path in module_paths:

            module_fn_schemas = get_module_function_schema(module_path)

            if len(module_fn_schemas)>0:
                module_schema_map[module_path] = module_fn_schemas
        

        return module_schema_map


    def rm(self, port:int=None, module:str=None):
        module2port = self.port2module
        if port == None:
            port = None
            for p, m in self.port2module.items():
                if module == m:
                    port = p
                    break
        
        assert type(port) in [str, int], f'{type(port)}'
        port = str(port)

        if port not in self.port2module:
            print(f'rm: {port} is already deleted')
            return None
        return self.subprocess_manager.rm(key=port)

    def rm_all(self):
        for port in self.port2module:
            self.rm(port=port)

    def ls_ports(self):
        return self.subprocess_manager.ls()

    def add(self,module:str, port:int, mode:str):
        print("DEBUG", module)
        module = self.resolve_module_path(module)
        module_filepath = self.get_object(module).get_module_filepath(include_pwd=False)
        print(module, __file__ , 'DEBUG')
        print(module_filepath)
        command_map ={
            'gradio':  f'python {module_filepath} -fn=run_gradio -args="[{port}]"',
            'streamlit': f'python {module_filepath} -fn=run_streamlit -args="[{port}]"'
        }

        command  = command_map[mode]
        print(command)
        process = self.subprocess_manager.add(key=str(port), command=command, add_info= {'module':module })
        return {
            'module': module,
            'port': port,
        }
    submit = add

    def stdout(self, modules : list):
        dict_stdout = {}

        for simple, full in modules.items():
            try:
                print(full)
                module = self.get_object(full)
                dict_stdout[simple] = { 'gradio' : hasattr(module, 'gradio'), 'streamlit' : hasattr(module, 'streamlit'), 'fn' :   [fn for fn in Module.functions(module)]   }
            except Exception as e:
                print(e)
                continue
        return dict_stdout

    def resolve_module_path(self, module):
        simple2path = deepcopy(self.simple2path)
        module_list = list(simple2path.values())

        if module in simple2path.keys():
            module = simple2path[module]
    
        assert module in module_list, f'{module} not found in {module_list}'
        return module

    
    module = None
    @classmethod
    def get_instance(cls, config = {}):
        if cls.module == None:
            cls.module = cls(config=config)
        return cls.module


    @classmethod
    def argparse(cls):
        parser = argparse.ArgumentParser(description='Gradio API and Functions')
        parser.add_argument('--api', action='store_true')

        '''
        if --no-api is chosen
        '''
        parser.add_argument('--module', type=str, default='nothing my guy')
        parser.add_argument('--port', type=int, default=8000)
        
        return parser.parse_args()

    def run_command(command:str):

        process = subprocess.run(shlex.split(command), 
                            stdout=subprocess.PIPE, 
                            universal_newlines=True)
        
        return process

import socket
import argparse
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

args = GradioModule.argparse()

@app.get("/")
async def root():
    module = GradioModule.get_instance()
    return {"message": "GradioFlow MothaFucka"}


global graph 
graph = dict()

@app.get("/test")
async def test():
    module = GradioModule.get_instance()
    return module.stdout(module.module_list)

@app.get("/list")
async def module_list(mode='simple'):
    module = GradioModule.get_instance()

    # if mode == 'full':
    #     module_list = module.module_list
    if mode == "streamable":
        module_list = module.stdout(module.simple2path) 
    elif mode == 'simple':
        module_list = list(module.simple2path) 
    else:
        raise NotImplementedError()
    return module_list

@app.get("/simple2path")
async def module_list(path_map:bool=False):
    module = GradioModule.get_instance()

    return module.simple2path

@app.get("/schemas")
async def module_schemas():
    module = GradioModule.get_instance()
    modules = module.get_module_schemas()
    return modules


@app.get("/schema")
async def module_schema(module:str, gradio:bool=True):
    if gradio:
        self = GradioModule.get_instance()
        module_schema = self.get_gradio_function_schemas(module, return_type='dict')
    else:
        module_schema = GradioModule.get_module_function_schema(module)
    return module_schema


@app.get("/ls_ports")
async def ls():
    self = GradioModule.get_instance()
    return self.ls_ports()


@app.get("/add")
async def module_add(module:str=None, mode:str="gradio"):
    self = GradioModule.get_instance()
    port = self.suggest_port()
    return self.add(port=port, module=module, mode=mode)


@app.get("/rm")
async def module_rm(module:str=None, port:str=None, name:str=None):
    self = GradioModule.get_instance()
    for key, value in graph.items():
        if f"{module}-{port}" in key :
            graph.pop(key, None)
        for link in value:
            if f"{module}-{port}" in link:
                value.remove(link)
    self.rm(port=port)

    for proc in process_iter():
        for conns in proc.connections(kind='inet'):
            if conns.laddr.port == port:
                proc.send_signal(SIGKILL) # or SIGKILL


    return self.port_connected(port)

@app.get("/rm_all")
async def module_rm_all(module:str=None, ):
    self = GradioModule.get_instance()
    return self.rm_all()


@app.get("/getattr")
async def module_getattr(key:str='subprocess_map', ):
    self = GradioModule.get_instance()
    return getattr(self,key)


@app.get("/port2module")
async def port2module(key:str='subprocess_map' ):
    self = GradioModule.get_instance()
    return self.port2module


@app.get("/module2port")
async def module2port( key:str='subprocess_map'):
    self = GradioModule.get_instance()

    return self.module2port

@app.get('/kill_port')
def portopen(port : int):
    module = GradioModule.get_instance()
    if module.port_connected(port):
        for proc in process_iter():
            for conns in proc.connections(kind='inet'):
                if conns.laddr.port == port:
                    proc.send_signal(SIGKILL) # or SIGKILL
        return "Done"
    else:
        return "Port not on"

@app.get('/add_chain')
def add_chain(a : str, b : str):
    if not a in graph.keys(): 
        graph[a] = []
    print(b in graph[a])
    None if b in graph[a] else graph[a].append(b)
    return graph

@app.get('/rm_chain')
def rm_chain(a : str, b : str):
    if not a in graph.keys(): 
        return False
    graph[a].remove(b)
    return True


@app.get('/get_chain')
def get_chain():
    return graph

@app.get("/hugginface_spaces")
def load_hugginface_cache():
    print(f"{os.path.dirname(os.path.realpath(__file__))}/.cache/spaces.json")
    data = ""
    with open(f"{os.path.dirname(os.path.realpath(__file__))}/.cache/spaces.json", 'r') as file:
         data = json.load(file)
         print(file)
     #.read_write_api_data(file="spaces.json", api="https://huggingface.co/api/spaces")
    return data
    



if __name__ == "__main__":
    
    if args.api:
        uvicorn.run(f"module:app", host="0.0.0.0", port=8000, reload=True, workers=2)
    # else:
        # module_proxy = GradioModule()
        # module_proxy.launch(module=args.module, port=args.port)
        