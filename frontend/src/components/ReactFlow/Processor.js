import ReactFlow, { Background,
                    applyNodeChanges,
                    ReactFlowProvider,
                    addEdge,
                    updateEdge,
                    applyEdgeChanges,
                    Controls,
                    MarkerType
                    } from 'react-flow-renderer';

import React ,{ useState, 
                useMemo,
                useCallback,
                useRef,
                useEffect,
                MouseEvent } from 'react';

import Navbar from '../Navagation/navbar';
import CustomEdge from '../Edges/Custom'
import CustomLine from "../Edges/CustomLine.js";
import CustomNodeIframe from "../Nodes/Custom.js";

import { CgMoreVerticalAlt } from 'react-icons/cg'
import { BsFillEraserFill } from 'react-icons/bs' 
import { FaRegSave } from 'react-icons/fa'

import { useThemeDetector } from './utils'

import { useTransition } from '@react-spring/web'
import { animated } from '@react-spring/web'


import '../../css/dist/output.css'
import '../../css/index.css'
import '../../css/log.css'

const NODE = {
  custom : CustomNodeIframe,
}

const EDGE = {
  custom : CustomEdge
}

let id = 0

export default function Processor() {

    // =======================
    // Initialize State's
    // ======================= 
    const [theme, setTheme] = useState(useThemeDetector)
    const [nodes, setNodes] = useState([]);
    const [edges, setEdges] = useState([]);
    const [reactFlowInstance, setReactFlowInstance] = useState(null);
    const reactFlowWrapper = useRef(null);
    const [tool, setTool] = useState(false)
    const ref = useRef(null)


    const handleClick = () => {
      ref.current?.("The Backend is currently setting up the application")
    }
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
      (oldEdge, newConnection) => setEdges((els) => updateEdge(oldEdge, newConnection, els)),
      []
    );

    // =======================
    // Save, Load & Erase
    // =======================
    useEffect(() => {
      const restore = () => {
      const flow = JSON.parse(localStorage.getItem('flowkey'));
        
        if(flow){
          flow.nodes.map((nds) => nds.data.delete = deleteNode)
          flow.edges.map((eds) => eds.data.delete = deleteEdge)
          setNodes(flow.nodes || [])
          setEdges(flow.edges || [])
          console.log(flow)
        }
      }
      restore()
    },[])

    const onSave = useCallback(() => {
      if (reactFlowInstance) {
        const flow = reactFlowInstance.toObject();
        alert("The current nodes have been saved into the localstorage 💾")
        localStorage.setItem('flowkey', JSON.stringify(flow));
        var labels = [];
        var colour = [];
        var emoji = [];
          for(let i = 0; i < flow.nodes.length; i++){
            if (!labels.includes(flow.nodes[i].data.label))
              colour.push(flow.nodes[i].data.colour)
              emoji.push(flow.nodes[i].data.emoji)
              labels.push(flow.nodes[i].data.label)
          }
        localStorage.setItem('colour',JSON.stringify(colour))
        localStorage.setItem('emoji', JSON.stringify(emoji))
      }
    }, [reactFlowInstance]);

    const onErase = useCallback(() => {
      const flow = localStorage.getItem("flowkey")
      if (reactFlowInstance && flow){
        alert("The current nodes have been erased from the localstorage")
        localStorage.removeItem("flowkey")
        localStorage.removeItem('colour')
        localStorage.removeItem('emoji')
      }
    },[reactFlowInstance])

    // =======================
    // Node's & Edge's Remove
    // ======================= 
    const deleteEdge = (id) => setEdges((eds) => eds.filter(e => e.id !== id))

    const deleteNode = (_) =>{
      const metadata = _[0].id.split("-")
      fetch(`http://localhost:8000/rm?${new URLSearchParams({module: metadata[0], port: metadata[1]})}`, {method : "GET", mode: 'cors'}).then(res => res.json()).then(
      () =>{
        setNodes((nds) => nds.filter(n => n.id !== _[0].id ))
      }
      )
    }

    // =======================
    // Edge's Connection
    // ======================= 
    const onConnect = useCallback(
      (params) => {
        console.log(params)
        setEdges((els) => addEdge({...params, type: "custom", animated : true, style : {stroke : "#00FF4A", strokeWidth : "6"}, markerEnd: {type: MarkerType.ArrowClosed, color : "#00FF4A"}, data : { delete : deleteEdge}}, els))
        // fetch("http://localhost:2000/api/append/connection", {method : "POST", mode : 'cors', headers : { 'Content-Type' : 'application/json' }, body: JSON.stringify({"source": params.source, "target" : params.target})}).then( res => {
        //   console.log(res)
        // }).catch(error => {
        //   console.log(error)
        // })
      },
      [setEdges]
    );

    // =======================
    // Drag & Drop
    // ======================= 
    const onDragOver = useCallback((event) => {
      event.preventDefault();
      event.dataTransfer.dropEffect = 'move';
    }, []);
    
    const onDrop = useCallback(
      (event) => {
        event.preventDefault();
        console.log("dropped")
        if(event.dataTransfer.getData('application/reactflow')  !== ""){
          const reactFlowBounds = reactFlowWrapper.current.getBoundingClientRect();
          const type = event.dataTransfer.getData('application/reactflow');
          const item  = event.dataTransfer.getData('application/item');
          const style = JSON.parse(event.dataTransfer.getData('application/style'));

          if (typeof type === 'undefined' || !type) {
            return;
          }

          fetch(`http://localhost:8000/add?${new URLSearchParams({module: item, mode : style.stream})}`, {method : "GET", mode: 'cors'}).then(
            (res) => res.json()).then( (data) =>{ 
              const position = reactFlowInstance.project({
                x: event.clientX - reactFlowBounds.left,
                y: event.clientY - reactFlowBounds.top,});
                console.log("new Node")
           
                const newNode = {
                  id: `${item}-${data.port}-${nodes.length+1}`,
                  type,
                  position,
                  dragHandle : `#draggable`,
                  data: { 
                          label: `${item}`,
                          host : `http://localhost:${data.port}`,
                          colour : `${style.colour}`,
                          emoji : `${style.emoji}`,
                          delete : deleteNode,
                          notification : handleClick},};
                  setNodes((nds) => nds.concat(newNode));
                  console.log(nodes)
                })    
        }
      },
      [reactFlowInstance, nodes]);


    return (
      <div className={`${theme ? "dark" : ""}`}>          
        
        <div className={` absolute text-center ${tool ? "h-[203.3333px]" : "h-[41px]"} overflow-hidden w-[41px] text-4xl top-4 right-5 z-50 cursor-default select-none bg-white dark:bg-stone-900 rounded-full border border-black dark:border-white duration-500`}  >
          <CgMoreVerticalAlt className={` text-black dark:text-white ${tool ? "-rotate-0 mr-auto ml-auto mt-1" : " rotate-180 mr-auto ml-auto mt-1"} duration-300`} onClick={() => {setTool(!tool);} }/>
          <h1 title={theme ? 'Dark Mode' : 'Light Mode'} className={`p-4 px-1 pb-0 ${tool ? "visible" : "invisible"} text-3xl`} onClick={() => setTheme(!theme)} >{theme  ? '🌙' : '☀️'}</h1> 
          <FaRegSave title="Save" className={`mt-6 text-black dark:text-white ${tool ? "visible" : " invisible"} ml-auto mr-auto `} onClick={() => onSave()}/> 
          <BsFillEraserFill title="Erase" className={`mt-6 text-black dark:text-white ml-auto mr-auto ${tool ? "visible" : " invisible"} `} onClick={() => onErase()}/>
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
                         onDragOver={onDragOver}
                         onDrop={onDrop}
                         onInit={setReactFlowInstance}
                         connectionLineComponent={CustomLine}
                         fitView>
                <Background variant='dots' size={1} className=" bg-white dark:bg-neutral-800"/>
                <Controls/>
              </ReactFlow>
              <MessageHub children={(add) => {ref.current = add}}/>
            </div>
          </ReactFlowProvider>
        </div>
      </div>
    );
  }



function MessageHub({
  config = { tension: 125, friction: 20, precision: 0.1 },
  timeout = 1500,
  children,
}) {
  const refMap = useMemo(() => new WeakMap(), [])
  const cancelMap = useMemo(() => new WeakMap(), [])
  const [items, setItems] = useState([])

  const transitions = useTransition(items, {
    from: { opacity: 0, height: 0, life: '100%' },
    keys: item => item.key,
    enter: item => async (next, cancel) => {
      cancelMap.set(item, cancel)
      await next({ opacity: 1, height: refMap.get(item).offsetHeight })
      await next({ life: '0%' })
    },
    leave: [{ opacity: 0 }, { height: 0 }],
    onRest: (result, ctrl, item) => {
      setItems(state =>
        state.filter(i => {
          return i.key !== item.key
        })
      )
    },
    config: (item, index, phase) => key => phase === 'enter' && key === 'life' ? { duration: timeout } : config,
  })

  useEffect(() => {
    children((msg) => {
      setItems(state => [...state, { key: id++, msg }])
    })
  }, [])

  return (
    <div className='fixed z-[1000] w-auto bottom-7 m-auto right-[30px] flex flex-col '>
      {transitions(({ life, ...style }, item) => (
        
        <animated.div className=" relative overflow-hidden box-border w-[40ch] py-2 rounded-xl" style={style}>
          <div className=' text-black dark:bg-stone-900 dark:text-white bg-slate-100 opacity-[0.9] py-12 px-22 font-sans grid rounded-xl shadow-2xl shadow-blue-600 border-black dark:border-white border-2 ' ref={(ref) => ref && refMap.set(item, ref)}>
            <animated.div className=" absolute bottom-0 left-0 w-auto h-[5px] dark:bg-gradient-to-bl dark:from-Retro-light-pink dark:to-Vapor-Blue bg-gradient-to-bl from-blue-400  to-blue-900 " style={{ right: life }} />
            <div className=' bg-slate-200 dark:bg-stone-800 dark:text-white rounded-t-xl absolute top-0 right-0 h-10 w-full border-3 border-black dark:border-b-white border-2'><h3 className='px-3 py-2 font-bold'>Setting Up Localhost</h3></div>
            <p className='px-3 py-2 font-sans font-semibold'>{item.msg}</p>
            {/* <Button
              onClick={(e) => {
                e.stopPropagation()
                if (cancelMap.has(item) && life.get() !== '0%') cancelMap.get(item)()
              }}>
              <X size={18} /> 
            </Button> */}
          </div>
        </animated.div>
      ))}
    </div>
  )
}