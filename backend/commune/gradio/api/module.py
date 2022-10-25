


import os, sys
sys.path.append(os.environ['PWD'])
import gradio
from commune import Module
from inspect import getfile
import inspect
import socket
from multiprocessing import Process
from commune.utils import SimpleNamespace
from commune.utils import *
import socket
import argparse
from fastapi import FastAPI
import uvicorn
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
    



    

class GradioModule(Module):
    default_config_path =  'gradio.api'


    # without '__reduce__', the instance is unserializable.
    def __reduce__(self):
        deserializer = GradioModule
        serialized_data = (self.config,)
        return deserializer, serialized_data


    def __init__(self, config=None):
        Module.__init__(self, config=config)
        self.subprocess_manager = self.get_object('subprocess.module.SubprocessModule')()

        self.host  = self.config.get('host', '0.0.0.0')
        self.port  = self.config.get('port', 8000)
        self.num_ports = self.config.get('num_ports', 10)
        self.port_range = self.config.get('port_range', [7865, 7870])
        
        # self.thread_manager = PriorityThreadPoolExecutor()
        # self.process_manager = self.get_object('cliProcessManager()

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
            module2port[module ] = port
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

    def find_registered_functions(self, module:str):
        '''
        find the registered functions
        '''
        fn_keys = []
        self.get_module
        for fn_key in self.get_funcs(module):
            try:
                if getattr(getattr(getattr(self,fn_key), '__decorator__', None), '__name__', None) == GradioModule.register.__name__:
                    fn_keys.append(fn_key)
            except:
                continue
        return fn_keys


    @staticmethod
    def get_funcs(self):
        return [func for func in dir(self) if not func.startswith("__") and callable(getattr(self, func, None)) ]


    @staticmethod
    def has_registered_functions(self):
        '''
        find the registered functions
        '''
        for fn_key in GradioModule.get_funcs(self):
            if getattr(getattr(getattr(self,fn_key), '__decorator__', None), '__name__', None) == GradioModule.register.__name__:
                return True


        return False



    def active_port(self, port:int=1):
        return port in self.port2module


    def port_connected(self ,port : int):
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
        subprocess_map = self.subprocess_map

        if str(port) in subprocess_map:
            return False
        else:
            return True
        
    def suggest_port(self, max_trial_count=10):
        for port in range(*self.port_range):
            if self.port_available(port):
                return port
        

        '''
        TODO: kill a port when they are all full
        '''
        raise Exception(f'There does not exist an open port between {self.port_range}')
        
    def compile(self, module:str, live=False, flagging='never', theme='default', **kwargs):
        print("Just putting on the finishing touches... 🔧🧰")
        module_class = self.get_object(module)
        module = module_class()

        gradio_functions_schema = self.get_gradio_function_schemas(module)


        interface_fn_map = {}

        for fn_key, fn_params in gradio_functions_schema.items():                
            interface_fn_map[fn_key] = gradio.Interface(fn=getattr(module, fn_key),
                                        inputs=fn_params['input'],
                                        outputs=fn_params['output'],
                                        theme=theme)
            print(f"{fn_key}....{bcolor.BOLD}{bcolor.OKGREEN} done {bcolor.ENDC}")


        print("\nHappy Visualizing... 🚀")
        demos = list(interface_fn_map.values())
        names = list(interface_fn_map.keys())
        return gradio.TabbedInterface(demos, names)

    @staticmethod
    def expose(**kw):
        "Note that using **kw you can tag the function with any parameters"
        def wrap(func):
            name = func.func_name
            assert not name.startswith('_'), "Only public methods can be exposed"

            meta = func.__meta__ = kw
            meta['exposed'] = True
            return func

    @staticmethod
    def register(inputs, outputs):
        
        def register_gradio(func):
            GradioModule.expose(gradio='True', 
                                inputs=inputs,
                                outputs=outputs)(func=func)
            return func
        return register_gradio


    def get_modules(self, force_update=True):
        modules = []
        failed_modules = []
        for root, dirs, files in self.client.local.walk('/app/commune'):
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

    def add(self,module:str, port:int):
        module_list = self.get_modules()
        assert module in module_list, f'{module} is not in {module_list}'
        command  = f'python {__file__} --module={module} --port={port}'
        process = self.subprocess_manager.add(key=str(port), command=command, add_info= {'module':module })
        return {
            'module': module,
            'port': port,
        }
    submit = add

    def launch(self, interface:gradio.Interface=None, module:str=None, **kwargs):
        """
            @params:
                - name : string
                - interface : gradio.Interface(...)
                - **kwargs
            
            Take any gradio interface object 
            that is created by the gradio 
            package and send it to the backend api
        """
        

        if interface == None:
    
            assert isinstance(module, str)
            module_list = self.get_modules()
            assert module in module_list, f'{args.module} is not in {module_list}'
            interface = self.compile(module=module)
        kwargs["port"] = kwargs.pop('port', self.suggest_port()) 
        if kwargs["port"] == None:
            return {'error': 'Ports might be full'}
        kwargs["server_port"] = kwargs.pop('port')
        kwargs['server_name'] = self.host
        
        default_kwargs = dict(
                    
                        inline= False,
                        share= None,
                        debug=False,
                        enable_queue= None,
                        max_threads=10,
                        auth= None,
                        auth_message= None,
                        prevent_thread_lock= False,
                        show_error= True,
                        show_tips= False,
                        height= 500,
                        width= 900,
                        encrypt= False,
                        favicon_path= None,
                        ssl_keyfile= None,
                        ssl_certfile= None,
                        ssl_keyfile_password= None,
                        quiet= False
        )

        kwargs = {**default_kwargs, **kwargs}
        interface.launch(**kwargs)

        return kwargs


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

    default_uvicorn_kwargs =  dict(path=f"module:app", 
                                 host="0.0.0.0", port=8000, 
                                reload=True, 
                                workers=2)

    @property
    def api_config(self):
        default_api_config = dict(
                app="module:app",
                root= None,
                host="0.0.0.0", port=8000, 
                reload=True, 
                workers=2
                )

        api_config = self.config.get('api', {})
        assert isinstance(api_config, dict)
        api_config = {**default_api_config, **api_config}
        return api_config


    def run_app(self,app=None, **kwargs):

        '''
        expample 
        dict(
            root= None,
            path=f"module:app", 
            host="0.0.0.0", port=8000, 
            reload=True, 
            workers=2
            )
        '''
        kwargs = {**self.api_config, **kwargs}
        root = kwargs.pop('root', None)
        app = self.get_app(app=app, root=root)
        uvicorn.run(**kwargs)

    @classmethod
    def get_app(cls, root=None, app=None, config={}, **kwargs):

        if app == None:
            app = FastAPI(**kwargs)
        assert isinstance(app, FastAPI)

        if root == None:
            root = ''
        elif isinstance(root, str):
            root = root
        else:
            raise NotImplementedError(root)
        
        assert isinstance(root, str), f'"{root}" should be a str but is {type(root)}'
        
        @app.get(f"{root}/")
        async def root_endpoint():
            self = cls.get_instance()
            module = self.get_instance()
            return {"message": "GradioFlow MothaFucka"}
        @app.get(f"{root}/list")
        async def module_list(path_map:bool=False):
            self = cls.get_instance()
            module_list = self.get_modules()
            if path_map:
                module_path_map = {}
                for module in module_list:
                    dict_put(module_path_map ,module.split('.')[:-1], module.split('.')[-1])
                return module_path_map
            else:
                return module_list

        @app.get(f"{root}/schemas")
        async def module_schemas():
            self = cls.get_instance()
            modules = self.get_module_schemas()
            return modules
        @app.get(f"{root}/schema")
        async def module_schema(module:str, gradio:bool=True):

            self = cls.get_instance()  
            if gradio:
                module_schema = self.get_gradio_function_schemas(module, return_type='dict')
            else:
                module_schema = self.get_module_function_schema(module)
            return module_schema
        @app.get(f"{root}/ls_ports")
        async def ls():

            self = cls.get_instance()
            return self.ls_ports()

        @app.get(f"{root}/add")
        async def module_add(module:str=None):

            self = cls.get_instance()
            port = self.suggest_port()
            # self.launch(module=module)
            return self.add(port=port, module=module)
        @app.get(f"{root}/rm")
        async def module_rm(module:str=None, port:str=None ):
            self = cls.get_instance()
            return self.rm(port=port, module=module)
        @app.get(f"{root}/rm_all")
        async def module_rm_all(module:str=None, ):
            self = cls.get_instance()
            return self.rm_all()
        @app.get(f"{root}/getattr")
        async def module_getattr(key:str='subprocess_map', ):
            self = cls.get_instance()
            return getattr(self,key)
        @app.get(f"{root}/port2module")
        async def port2module(key:str='subprocess_map' ):
            self = cls.get_instance()
            return self.port2module
        @app.get(f"{root}/module2port")
        async def module2port( key:str='subprocess_map'):
            self = cls.get_instance()
            return self.module2port  


        return app      

register = GradioModule.register
app = GradioModule.get_app()



if __name__ == "__main__":
    
    args = GradioModule.argparse()
    module = GradioModule()

    if args.api:
        module.run_app()
    else:
        module.launch(module=args.module, port=args.port)
