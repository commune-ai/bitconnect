import React, { useCallback, useEffect, useRef, useState } from "react"
import { Handle, Position } from "react-flow-renderer"
import {TbResize} from 'react-icons/tb'
import {BiCube, BiRefresh} from 'react-icons/bi'
import {BsTrash} from 'react-icons/bs'
import {CgLayoutGridSmall} from 'react-icons/cg'
import {useDrag} from '@use-gesture/react'
import { useSpring, animated } from 'react-spring'

import '../../css/counter.css'

const MINIMUM_HEIGHT = 600;
const MINIMUM_WIDTH = 540; 


//create your forceUpdate hook
// function useForceUpdate(){
//     console.log("called")
//     const [value, setValue] = useState(0); // integer state
//     return () => setValue(value => value + 1); // update state to force render
//     // An function that increment 👆🏻 the previous state like here 
//     // is better than directly setting `value + 1`
// }

export default function ModuleFrame({id, data}){
    const [collapsed, setCollapsible] = useState(true)
    const [{width, height}, api] = useSpring(() => ({width: MINIMUM_WIDTH, height: MINIMUM_HEIGHT }))
    const [sizeAdjuster, setSizeAdjuster] = useState(true)
    const [reachable ,setReachable] = useState(false)
    // const forceUpdate = useForceUpdate();
    const dragElement = useRef()

    const bind = useDrag((state) => {
      const isResizing = (state?.event.target === dragElement.current);
      if (isResizing) {
        api.set({
          width: state.offset[0],
          height: state.offset[1],

        });
      } 
    }, {
      from: (event) => {
        const isResizing = (event.target === dragElement.current);
        if (isResizing) {
          return [width.get(), height.get()];
        } 
      },
    });


    const isFetchable = useCallback(async () => {
      return fetch(data.host, {mode: 'no-cors'}).then((res) => {
        return true
      }).catch((err)=>{
        return false
      })
    },[data])

    const handelServerRender = useCallback(async () => {
      if (!collapsed){
        return setCollapsible((clps) => !clps)
      } else {
          const reach = await isFetchable()
          return reach ? setCollapsible((clps) => !clps) : data.notification()
      }
    }, [isFetchable, setCollapsible, collapsed, data] )

    useEffect(() => {
      const fetched = setInterval(
        async () => {
          const fetch = await isFetchable()
          if (fetch){
            setReachable(true)
            clearInterval(fetched)
          }
        },1000) 
    },[isFetchable])


    return (
    <div className="w-auto h-auto">
      
      <div id={'draggable'}className=" flex w-full h-10 top-0 cursor-pointer" onClick={() => {}}>
      <div id={'draggable'} title={collapsed ? "Expand Node" : "Collaspse Node"} className=" flex-none duration-300 cursor-pointer shadow-xl border-2 border-white h-10 w-10 mr-2 -mt-3 bg-Warm-Blue rounded-xl" onClick={ async () => { await handelServerRender()}}><CgLayoutGridSmall className="h-full w-full text-white p-1"/></div>

      <div className={` flex ${!collapsed ? '' : 'w-0 hidden'}`}>
                      <div title="Adjust Node Size" className="duration-300 cursor-pointer shadow-xl border-2 dark:border-white border-white h-10 w-10 mr-2 -mt-3 bg-Warm-Violet rounded-xl" onClick={() => {setSizeAdjuster((size) => !size)}}><TbResize className="h-full w-full text-white p-1"/></div>
                      <a href={data.host} target="_blank" rel="noopener noreferrer"><div title="Gradio Host Site" className="duration-300 cursor-pointer shadow-xl border-2 dark:border-white border-white h-10 w-10 mr-2 -mt-3 bg-Warm-Pink rounded-xl"><BiCube className="h-full w-full text-white p-1"/></div></a>
                      <div title="Delete Node" className="duration-300 cursor-pointer shadow-xl border-2 dark:border-white border-white h-10 w-10 mr-2 -mt-3 bg-Warm-Red rounded-xl" onClick={() => data.delete([{id : id}])}><BsTrash className="h-full w-full text-white p-1"/></div>
                      <div title="Refresh Node" className="duration-300 cursor-pointer shadow-xl border-2 dark:border-white border-white h-10 w-10 mr-2 -mt-3 bg-Warm-Orange rounded-xl"><BiRefresh className="h-full w-full text-white p-1"/></div>
        </div>
      </div>

      { !collapsed && reachable && <>
          <animated.div className={`border-dashed  ${sizeAdjuster ? 'border-4 dark:border-white border-black' : ''} relative top-0 left-0 z-[1000] touch-none shadow-lg rounded-xl`} style={{width, height }} {...bind()}>
            <div id="draggable" className={`absolute h-full w-full ${data.colour} shadow-2xl rounded-xl -z-20`}></div>
            <iframe id="iframe" 
                        src={data.host} 
                        title={data.label}
                        frameBorder="0"
                        className=" p-[0.6rem] -z-10 h-full w-full ml-auto mr-auto overflow-y-scroll"/>
              <div className={` ${sizeAdjuster ? '' : 'hidden'} rounded-full border-2 absolute -bottom-4 -right-4 w-7 h-7 bg-Blue-Royal cursor-nwse-resize touch-none shadow-lg`} ref={dragElement}>
            </div>  
            </animated.div>
            </>
        }

            <Handle type="target"
                    id="input"
                    position={Position.Left}
                    style={{"paddingRight" : "5px" , "marginTop" : "15px", "height" : "25px", "width" : "25px",  "borderRadius" : "3px", "zIndex" : "10000", "background" : "white", "boxShadow" : "3px 3px #888888"}}/>
                  
                  {/*Output*/}
            <Handle type="source"
                    id="output"
                    position={Position.Right}
                    style={{"paddingLeft" : "5px", "marginTop" : "15px" ,"height" : "25px", "width" : "25px",  "borderRadius" : "3px", "zIndex" : "10000", "background" : "white", "boxShadow" : "3px 3px #888888", 'position' : 'absolute'}}/>
                    
        { collapsed  &&
              <div id={`draggable`}
                   className={` w-[340px] h-[140px]  text-white text-md flex flex-col text-center items-center cursor-grab shadow-lg
                                p-5 px-2 rounded-md break-all -z-20 ${data.colour} hover:opacity-70 duration-300`} onClick={async () => await handelServerRender()}>
                  <div  className="absolute text-6xl opacity-60 z-10 pt-8 ">{data.emoji}</div>    
                  <h2 className={`max-w-full font-sans text-blue-50 leading-tight font-bold text-3xl flex-1 z-20 pt-10`} style={{"textShadow" : "0px 1px 2px rgba(0, 0, 0, 0.25)"}} >{data.label}</h2>
              </div > 
        }
      </div>)
}
<<<<<<< HEAD:frontend/src/components/Nodes/Custom.js

=======
>>>>>>> fd926ecf2fa48a23e5fec18c84fb4cf28d09417f:frontend/src/components/Nodes/Module.js
