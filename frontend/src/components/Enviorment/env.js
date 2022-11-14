import ReactFlow, { Background,
                    applyNodeChanges,
                    ReactFlowProvider,
                    addEdge,
                    updateEdge,
                    applyEdgeChanges,
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

import { CgMoreVerticalAlt } from 'react-icons/cg'

import { useThemeDetector } from './utils'

import '../../css/dist/output.css'
import '../../css/index.css'

const NODE = {
  custom      : ModuleFrame,
  customInput : InputCompnent,
  bundle      : Bundle,
  process     : Process
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

        // if the edges already exist then
        // do not update the edge or if the 
        // newConnection target is not a bundle-node-args__custom (bundle)
        // then if the connection already contains a conection then do not
        // update
        if ((edges.filter((eds) => eds.source === newConnection.source && eds.target === newConnection.target).length === 1)||
            (!newConnection.target.includes('bundle-node-args__custom-') && edges.filter((eds) => eds.target === newConnection.target).length === 1)) return 
        
        // set the connection        
        if (newConnection.target.includes("bundle-node-args__custom") ){
          
          // if the new connection is bundle then
          // update the arguments and add new
          // node.
          
          setNodes((nds) => nds.filter((node) => {
            if (node.id === newConnection.target){
                node.data = {
                  ...node.data,
                  args: [...node.data.args, nds.find((n)=> newConnection.source === n.id)],
                };
                if (node.data.args.length === 0){
                  setEdges((eds) => eds.filter(edge => edge.source !== node.id))
                  setNodes((nds) => nds.filter(n => n.id !== node.id))                
                  return false
                }
              } 
              return true
            }))
          
        }
        
        if (oldEdge.target.includes('bundle-node-args__custom')){
            // if the old connection is a bundle also
            // a bundle then delete the connection from 
            // the arguments
            setNodes((nds) => nds.filter((node) => {
            if (node.id === oldEdge.target){
              node.data = {
                ...node.data,
                args: node.data.args.filter((n) => n.id !== oldEdge.source)
              }
              if (node.data.args.length === 0){
                setEdges((eds) => eds.filter(edge => edge.source !== node.id))
                setNodes((nds) => nds.filter(n => n.id !== node.id))                
                return false
              }
            }
            return true
             }))

          }
          setEdges((els) => updateEdge(oldEdge, newConnection, els))

        },
      [edges]
    );

    // =======================
    // Node's & Edge's Remove
    // ======================= 
    const deleteEdge = useCallback(({id, source, target}) => {
      setEdges((eds) => eds.filter(e => e.id !== id))
      console.log(target)
      if(target.includes("bundle-node-args__custom")) {
        setNodes((nds) => {
          return nds.filter(node => {
            if(node.id === target){
              node.data ={
                ...node.data,
                args : node.data.args.filter((n) => n.id !== source)
              }
              if (node.data.args.length === 0){
                setEdges((eds) => eds.filter(edge => edge.source !== node.id))
                setNodes((nds) => nds.filter(n => n.id !== node.id))
                return false
              }
            }
            return true
          })
        })


      } else {
        fetch(`http://localhost:8000/rm_chain?${new URLSearchParams({a: source, b: target})}`, {method : "GET", mode: 'cors'}).then(res => res.json())
      }

    }, [setEdges, setNodes])


    // ==================
    // Delete Node
    // ==================
    const deleteNode = useCallback((_) =>{
      
      setEdges((eds) => eds.filter(e => e.target !== _[0].id || e.source !== _[0].id))

      if (_[0].type === "customInput"){

        const bundle_connection = nodes.filter(nds => nds.type === "bundle" && nds.data.args.map(c => c.id).includes(_[0].id)).map(n => n.id)
        setNodes((nds)=>{
          return nds.filter(node => {
            if (node.id === _[0].id) return false
            else if (bundle_connection.includes(node.id)){
              
              node.data = {
                ...node.data,
                args : node.data.args.filter(n => n.id !== _[0].id)
              }
              console.log(node.data.args, edges)

              if (node.data.args.length === 0){
                setEdges((eds) => eds.filter(e => e.source !== node.id))
                return false
              }
            }
            return true
          }) })
        return // end function

      } else if(_[0].type === "bundle"){

        setNodes((nds) => nds.filter(n => n.id !== _[0].id ))
        return // end function 
      
      } else{
        
        const metadata = _[0].id.split("-")
        fetch(`http://localhost:8000/rm?${new URLSearchParams({module: metadata[0], port: metadata[1]})}`, {method : "GET", mode: 'cors'}).then(res => res.json())
        setNodes((nds) => nds.filter(n => n.id !== _[0].id ))
        return // end function 
      
      }
    }, [nodes, edges, setEdges, setNodes])

    // =======================
    // Edge's Connection
    // ======================= 
    const onConnect = useCallback(
      (params) => {
        
       if ((params.source.includes('bundle-node-args__custom-') && params.target.includes('bundle-node-args__custom-'))
       || (edges.filter((eds) => eds.source === params.source && eds.target === params.target).length === 1)
       || ((!params.target.includes('bundle-node-args__custom-')
       && edges.filter((eds) => eds.target === params.target).length === 1))) return 
        
        setEdges((els) => addEdge({...params, type: "custom", animated : true, style : {strokeWidth : "6"}, markerEnd: {type: MarkerType.ArrowClosed}, data : { delete : deleteEdge}}, els))
        
        // if (params.target.includes("bundle-node-args__custom-process-node__custom__")){
        //   setNodes(nodes.map((nds) => {
        //     if (nds.id === params.target){
        //         const node = nodes.find((n) => n.id === params.source)
        //         nds.data = {
        //           ...nds.data,
        //           args: [...nds.data.args, node], 
        //         };
        //       }
        //       return nds
        //   }))
        // } else 
        if (params.target.includes("bundle-node-args__custom")){
          
          setNodes(nodes.map((nds) => {
            if (nds.id === params.target){
                const node = nodes.find((n) => n.id === params.source)
                nds.data = {
                  ...nds.data,
                  args: [...nds.data.args, node],
                };
              }
              return nds
          }))
        }else {
          fetch(`http://localhost:8000/add_chain?${new URLSearchParams({a : params.source, b : params.target})}`, {method : "GET", mode : 'cors', }).then( res => {res.json()}).catch(error => {console.error(error)})
        }
        
      },
      [setEdges, setNodes, nodes, edges, deleteEdge]
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
          const reactFlowBounds = reactFlowWrapper.current.getBoundingClientRect();
          
          const type = event.dataTransfer.getData('application/reactflow');

          if (typeof type === 'undefined' || !type) {
            return;
          }
          
          // postion of drop on the react flow environment 
          const position = reactFlowInstance.project({
            x: event.clientX - reactFlowBounds.left,
            y: event.clientY - reactFlowBounds.top,
          });
       
          if(type === "custom"){
          
          const item  = event.dataTransfer.getData('application/item');
          const style = JSON.parse(event.dataTransfer.getData('application/style'));

          fetch(`http://localhost:8000/add?${new URLSearchParams({module: item, mode : style.stream})}`, {method : "GET", mode: 'cors'}).then(
            (res) => res.json()).then( (data) =>{ 
                const newNode = {
                  id: `${item}-${data.port}-${uuidv4()}`,
                  type,
                  position,
                  dragHandle : `#draggable`,
                  data: { 
                          label: `${item}`,
                          host : `http://localhost:${data.port}`,
                          colour : `${style.colour}`,
                          emoji : `${style.emoji}`,
                          delete : deleteNode,
                          notification : notify },};
                  setNodes((nds) => nds.concat(newNode));                  
                })
          
          } else if (type === "process"){
            const item  = event.dataTransfer.getData('application/item');
            const style = JSON.parse(event.dataTransfer.getData('application/style'));

            const newNode = {
              id: `bundle-node-args__custom-process-node__custom__${item}__-${uuidv4()}`,
              type,
              position,
              data: { module : item, fn : style.fn, args : [], kwargs : [], config : {actor : false, cpus : 0, gpus : 0 , refresh : false, fn : "" },  colour : style.colour, emoji : style.emoji}, };
              setNodes((nds) => nds.concat(newNode));  

          } else { 
            const newNode = {
              id: `input-node__custom-${uuidv4()}`,
              type,
              position,
              data: { dtype: "NoneType", value: null },};
              setNodes((nds) => nds.concat(newNode));                  
          }

      },
      [reactFlowInstance, notify, deleteNode]);

    return (
      <div className={`${theme ? "dark" : ""}`}>          
        
        <div className={` absolute text-center ${tool ? "h-[90px]" : "h-[41px]"} overflow-hidden w-[41px] text-4xl top-4 right-5 z-50 cursor-default select-none bg-white dark:bg-stone-900 rounded-full border border-black dark:border-white duration-500`}  >
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
                         onNodesDelete={deleteNode}
                         onEdgesChange={onEdgesChange}
                         onEdgeUpdate={onEdgeUpdate}
                         onConnect={onConnect}
                         onConnectStart={onConnectStart}
                         onConnectStop={onConnectEnd}
                         onDragOver={onDragOver}
                         onDrop={onDrop}
                         onInit={setReactFlowInstance}
                         connectionLineComponent={CustomLine}
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