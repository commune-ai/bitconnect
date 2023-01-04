import ReactFlow, { Background,
                    applyNodeChanges,
                    ReactFlowProvider,
                    addEdge,
                    addNode,
                    updateEdge,
                    applyEdgeChanges,
                    getOutgoers,
                    Controls,
                    MarkerType,
                    } from 'react-flow-renderer';

import React ,{ useState, 
                useCallback,
                useRef } from 'react';

import { v4 as uuidv4 } from 'uuid';


import MessageHub from './Messages/Message';
import Navbar from '../Navagation/navbar';
import CustomEdge from '../Edges/Custom'
import CustomLine from "../Edges/CustomLine.js";

import ModuleFrame from "../Nodes/Module.js";
import InputCompnent from '../Nodes/Input';
import Bundle from '../Nodes/Bundle';
import Process from '../Nodes/Process';
import Space from '../Nodes/space';

import { CgMoreVerticalAlt } from 'react-icons/cg'

import { useThemeDetector } from './utils'

import '../../css/dist/output.css'
import '../../css/index.css'

const NODE = {
  custom      : ModuleFrame,
  customInput : InputCompnent,
  bundle      : Bundle,
  process     : Process,
  space      : Space,
}

const EDGE = {
  custom : CustomEdge
}

export default function Flow() {

    // =======================
    // Initialize State's
    // ======================= 
    const [theme, setTheme] = useState(useThemeDetector)
    const [nodes, setNodes] = useState([]);
    const [edges, setEdges] = useState([]);
    const [reactFlowInstance, setReactFlowInstance] = useState(null);
    const [tool, setTool] = useState(false)


    const reactFlowWrapper = useRef(null);
    const connectingNodeId = useRef(null);
    const ref = useRef(null)

    const notify = useCallback(() => {
      ref.current?.("The Backend is currently setting up the application")
    }, [ref])

    // =======================
    // Changes
    // =======================
    const onNodesChange = useCallback(
      (changes) => setNodes((nds) => applyNodeChanges(changes, nds)),
      [setNodes]
    );
  
    const onEdgesChange = useCallback(
      (changes) => setEdges((eds) => applyEdgeChanges(changes, eds)),
      [setEdges]
    );

    const onEdgeUpdate = useCallback(
      (oldEdge, newConnection) => {
          setEdges((els) => updateEdge(oldEdge, newConnection, els))
        },
      []
    );

    // =======================
    // Node's & Edge's Remove
    // ======================= 
    const deleteEdge = useCallback(async (edge, handle) => {
      var connection = null
      try {
         connection = reactFlowInstance.getNode(handle)
      }catch(err) {
        return 
      }

      console.log("Connection",connection)
      if(connection.type === "bundle") {
        setNodes((nds) => {
          return nds.filter(node => {
            if(node.id === edge.target){
              node.data ={
                ...node.data,
                args : node.data.args.filter((n) => n.id !== edge.source)
              }
              if (node.data.args.length === 0){
                setEdges((eds) => eds.filter(edg => edg.source !== node.id))
                setNodes((nds) => nds.filter(n => n.id !== node.id))
                return false
              }
            }
            return true
          })
        })
      } else if (connection.type === "process") {
        setNodes(nds => nds.filter((node) => {
          if (node.id === connection.id){
            
            node.data = {
              ...node.data,
              args : node.data.args.filter((n) => n !== edge.source)
            }
            if (node.data.args.length === 0){
              setEdges((eds) => eds.filter(edg => edg.source !== node.id))
            }
          }
          return true
        }))
      } else if (connection.type === "custom") {
        setNodes(nds => nds.filter((node) => {
          if (node.id === connection.id){
            node.data = {
              ...node.data,
              args : node.data.args.filter((n) => n !== edge.source)
            }
          }
          return true
        }))
      }
       // else {
      //   fetch(`http://localhost:8000/rm_chain?${new URLSearchParams({a: source, b: target})}`, {method : "GET", mode: 'cors'}).then(res => res.json())
      // }
      setEdges((eds) => eds.filter(e => e.id !== edge.id))

    }, [setEdges, setNodes, reactFlowInstance])


    const deleteEdges = useCallback((source) => {

      const edges = reactFlowInstance.getEdges().filter((eds) => (eds.source === source || eds.target === source))
      edges.forEach((edge) => {
        deleteEdge(edge, source === edge.source ? edge.target : edge.source)
      })
    }, [reactFlowInstance])

    // ==================
    // Delete Node
    // ==================
    const deleteNode = useCallback((nodes) =>{
      setNodes((nds) => nds.filter(n => n.id !== nodes[0].id ))
      if (nodes[0].type === "customInput"){
        // const bundle_connection = nodes.filter(nds => nds.type === "bundle" && nds.data.args.map(c => c.id).includes(_[0].id)).map(n => n.id)
        setNodes((nds)=>{
          return nds.filter( async node => {
            if (node.id === nodes[0].id) {
              await deleteEdges(node.id)
              return false
            }
            return true
          }) })

        return // end function

      } else if(nodes[0].type === "bundle"){
        
        reactFlowInstance.getEdges(nodes[0].id).map((conn) => {
          const node = reactFlowInstance.getNode(conn.target)
          if(node.type === 'custom'){
            setNodes(nds => nds.map((n) => {
              if (n.id === conn.target){
                n.data = {
                  ...n.data,
                  batch : n.data.batch.filter((b)=> b.includes(nodes[0].data.args))
                }
              }
              return n
            }))
          }
        })
        return // end function 
      
      } else {

        fetch(`http://localhost:8000/ls_ports`, { method: "GET", mode: 'cors' }).then(res => res.json()).then((r) => {
          r.forEach((port) => {
            console.log(nodes.map((node) => node))
            if (!nodes.map((node) => node.data.port).includes(port)) {
              fetch(`http://localhost:8000/rm?${new URLSearchParams({ port: port })}`, { method: "GET", mode: 'cors' })
              fetch(`http://localhost:8000/kill_port?${new URLSearchParams({ port: +port })}`, { method: "GET", mode: 'cors' })
            }
          })
        })
        console.log(nodes[0])
        
        setTimeout(() => {
          if (nodes[0].data.port !== undefined) {
            fetch(`http://localhost:8000/rm?${new URLSearchParams({ module: nodes[0].data.module, port: +nodes[0].data.port })}`, { method: "GET", mode: 'cors' })
            fetch(`http://localhost:8000/kill_port?${new URLSearchParams({ port: +nodes[0].data.port })}`, { method: "GET", mode: 'cors' })
          }
        }, 1000)

          return // end function 

      }
    }, [nodes, edges, setEdges, setNodes, reactFlowInstance])

    // =======================
    // Edge's Connection
    // ======================= 
    const onConnect = useCallback(
      (params) => {
       const source = reactFlowInstance.getNode(params.source);
       const target = reactFlowInstance.getNode(params.target);

       if (source.type === "bundle" && target.type === "bundle") return
       if (source.type === "bundle") return
       setEdges((els) => addEdge({...params, type: "custom", animated : true, style : {strokeWidth : "6"}, markerEnd: {type: MarkerType.ArrowClosed}, data : { delete : deleteEdge}, zIndex : 1000}, els))


       // NOTE TO FIX REDUNDANT CODE: create an object/class that handles the dependencies of the node that is being connected
       switch(target.type){

        case "custom":
          setNodes((nodes) => nodes.map((nds) => {
            if (nds.id === target.id){
              const argsBatch = source.type === "bundle" ? source.data.args.map(arg => arg.id) : [source.id]
                nds.data ={
                  ...nds.data,
                  args : [...nds.data.args, ...argsBatch]
              }
            }
            return nds 
          }))
          break
        case "process":
          setNodes((nodes) => nodes.map((nds) => {
            if (nds.id === target.id){
              const argsBatch = source.type === "bundle" ? source.data.args.map(arg => arg.id) : [source.id]
              nds.data ={
                ...nds.data,
                args : [...nds.data.args, ...argsBatch]
              }
            }
            return nds 
          }))
          break
        
        case "bundle":
          setNodes((nodes) => nodes.map((nds) => {
            if (nds.id === params.target){
                nds.data = {
                  ...nds.data,
                  args: [...nds.data.args, source],
                };
              }
              return nds
          }))
          break

        default:
          console.warn("FIX ME LATER")
       }

      },
      [setEdges, setNodes, deleteEdge, reactFlowInstance]
    );

    
    // =======================
    // Drag & Drop
    // ======================= 
    const onDragOver = useCallback((event) => {
      event.preventDefault();
      event.dataTransfer.dropEffect = 'move';
    }, []);

    const onConnectStart = useCallback((_, { nodeId }) => {
      connectingNodeId.current = nodeId;
    }, []);

    const onConnectEnd = useCallback(
      (event) => {
        
        if(!connectingNodeId.current.includes('input-node__custom-')) return

        const targetIsPane = event.target.classList.contains('react-flow__pane');
        const reactFlowBounds = reactFlowWrapper.current.getBoundingClientRect();

        if (targetIsPane){
          const id = `bundle-node-args__custom-${uuidv4()}`
          const position = reactFlowInstance.project({
            x: event.clientX - reactFlowBounds.left,
            y: event.clientY - reactFlowBounds.top,
          });

          const newNode = {
            id,
            type : 'bundle',
            position : position,
            data : { args : [nodes.find((value) => value.id === connectingNodeId.current)] }
          }
          
          setNodes((nds) => nds.concat(newNode));
          setEdges((eds) => eds.concat({ id : `${id}__edge`, type : "custom", source: connectingNodeId.current, target: id, animated : true, style : {strokeWidth : "6"}, markerEnd: {type: MarkerType.ArrowClosed}, data : { delete : deleteEdge} }));
          
        }
      }
    ,[reactFlowInstance, deleteEdge, nodes])

    
    const onDrop = useCallback(
      (event) => {
          event.preventDefault();
          
          // react flow bounds
          // const reactFlowBounds = reactFlowWrapper.current.getBoundingClientRect();
          
          const type = event.dataTransfer.getData('application/reactflow');

          if (typeof type === 'undefined' || !type) {
            return;
          }
          
          // postion of drop on the react flow environment 
          const position = reactFlowInstance.project({
            x: event.clientX,
            y: event.clientY,
          });

          if(type === "custom"){
          
          const item  = event.dataTransfer.getData('application/item');
          const style = JSON.parse(event.dataTransfer.getData('application/style'));
          
          const newNode = {
            id: `${item}__custom__module-${uuidv4()}`,
            type,
            position,
            dragHandle : `#draggable`,
            data: { 
                    label  : `${item}`,
                    colour : `${style.colour}`,
                    emoji  : `${style.emoji}`,
                    module : `${style.stream}`,
                    host   : null,
                    port   : null,
                    batch  : [],
                    args   : [],
                    output   : [],
                    previousOutput : null,
                    delete : deleteNode,
                    notification : notify },};
      setNodes((nds) => nds.concat(newNode));    

      fetch(`http://localhost:8000/add?${new URLSearchParams({module: newNode.data.label, mode : newNode.data.module})}`, {method : "GET", mode: 'cors'}).then(
            (res) => res.json()).then( (data) =>{

              
              setNodes((nds) => nds.map((node) => {
                  if (node.id === newNode.id){
                    node.data = {
                      ...node.data,
                      host : `http://localhost:${data.port}`,
                      port : data.port.toString(),
                    }
                  }
                  return node
                }))
            })
              
          
          } else if (type === "space"){

            const item  = event.dataTransfer.getData('application/item');
            const id = `hugginface_space__custom-node__${item}__-${uuidv4()}`;
            
            console.log(type)
            const newNode = {
              id,
              type,
              position,
              data : {}
            }
            setNodes((nds) => nds.concat(newNode));

          } else if (type === "process"){

            const item  = event.dataTransfer.getData('application/item');
            const style = JSON.parse(event.dataTransfer.getData('application/style'));
            const parentNode = `group_bundle-node-args__custom-process-node__custom__${item}__-${uuidv4()}`;

            const newNode = {
              id: parentNode,
              type,
              position,
              data: { module : item,
                      fn : style.fn,
                      colour : style.colour,
                      emoji : style.emoji,
                      args : [], 
                      kwargs : [], 
                      batch : [], 
                      config : {actor : false, cpus : 0, gpus : 0 , refresh : false, fn : "" }, },
               };

              setNodes((nds) => nds.concat(newNode));  
          } else { 
            const newNode = {
              id: `input-node__custom-${uuidv4()}`,
              type,
              position,
              data: { dtype: "String", value: ""}};
              setNodes((nds) => nds.concat(newNode));                  
          }

      },
      [reactFlowInstance, notify, deleteNode]);

    return (
      <div className={`${theme ? "dark" : ""}`}>          
        
        <div className={` fixed text-center shadow-lg ${tool ? "h-[90px]" : "h-[41px]"} overflow-hidden w-[41px] text-4xl top-4 right-5 z-50 cursor-default select-none bg-white dark:bg-stone-900 rounded-full border border-black dark:border-white duration-500`}  >
          <CgMoreVerticalAlt className={` text-black dark:text-white ${tool ? "-rotate-0 mr-auto ml-auto mt-1" : " rotate-180 mr-auto ml-auto mt-1"}  duration-500`} onClick={() => setTool(!tool)}/>
          <h1 title={theme ? 'Dark Mode' : 'Light Mode'} className={`p-4 px-1 pb-2 ${tool ? "visible" : "invisible"} text-3xl`} onClick={() => setTheme(!theme)} >{theme  ? '🌙' : '☀️'}</h1> 
        </div>

        <div className={`flex h-screen w-screen ${theme ? "dark" : ""} transition-all`}>    
          <ReactFlowProvider>
          <Navbar colour={JSON.parse(localStorage.getItem('colour'))}
                  emoji={JSON.parse(localStorage.getItem('emoji'))}/>

            <div className="h-screen w-screen" ref={reactFlowWrapper}>
              <ReactFlow nodes={nodes}
                         edges={edges}
                         nodeTypes={NODE}
                         edgeTypes={EDGE}
                         onNodesChange={onNodesChange}
                         onEdgesChange={onEdgesChange}
                         onEdgeUpdate={onEdgeUpdate}
                         onConnect={onConnect}
                         onConnectStart={onConnectStart}
                         onConnectStop={onConnectEnd}
                         onNodesDelete={deleteNode}
                         onDragOver={onDragOver}
                         onDrop={onDrop}
                         onInit={setReactFlowInstance}
                         connectionLineComponent={CustomLine}
                         defaultZoom={1.5}
                         fitView>
                <Background variant='dots' size={1} className=" bg-white dark:bg-neutral-800"/>
                <Controls/>
                <MessageHub children={(add) => {ref.current = add}}/>
                
              </ReactFlow>
            </div>
          </ReactFlowProvider>
        </div>
      </div>
    );
  }