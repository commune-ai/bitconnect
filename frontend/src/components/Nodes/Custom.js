import React, { useEffect, useMemo, useRef, useState } from "react"
import {TbResize} from 'react-icons/tb'
import {BiCube, BiRefresh} from 'react-icons/bi'
import {BsTrash} from 'react-icons/bs'
import {CgLayoutGridSmall} from 'react-icons/cg'
import {RiDragMove2Fill} from 'react-icons/ri'
import {useDrag} from '@use-gesture/react'
import { useSpring, animated } from 'react-spring'
import { useTransition } from '@react-spring/web'


import '../../css/counter.css'

const MINIMUM_HEIGHT = 600;
const MINIMUM_WIDTH = 540; 

export default function CustomNodeIframe({id, data}){
    const [collapsed, setCollapsible] = useState(true)
    const [{x, y, width, height}, api] = useSpring(() => ({width: MINIMUM_WIDTH, height: MINIMUM_HEIGHT }))
    const [sizeAdjuster, setSizeAdjuster] = useState(true)
    const [reachable ,setReachable] = useState(false)
    const [refresh, setRefresh] = useState(0)
    const dragElement = useRef()
    const ref = useRef(null)

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

    const isFetchable = async () => {
      return fetch(data.host, {mode: 'no-cors'}).then((res) => {
        return true
      }).catch((err)=>{
        return false
      })
    }

    const handelRefresh = () => {
      setRefresh((refresh) => refresh++)
    }

    const handelServerRender = async () => {
      const reach = await isFetchable()
      reach ? setCollapsible((clps) => !clps) : data.notification()
    }
    

    useEffect(() => {
      const fetched = setInterval(
        async () => {
          const fetch = await isFetchable()
          if (fetch){
            setReachable(true)
            clearInterval(fetched)
          }
        },1000) 
    },[])


    return (
    <div className="w-10 h-10">
      
      <div className=" flex w-full h-10 top-0 cursor-pointer" onClick={() => {}}>
      <div title={!collapsed ? "Collaspse Node" : "Expand Node"} className=" flex-none duration-300 cursor-pointer shadow-xl border-2 border-white h-10 w-10 mr-2 -mt-3 bg-Green-Emerald rounded-xl" onClick={ async () => { handelServerRender() }}><CgLayoutGridSmall className="h-full w-full text-white p-1"/></div>
      <div id={'draggable'} title="Drag Node" className={` ${collapsed ? 'hidden' : ''} flex-none duration-300 cursor-pointer shadow-xl border-2 dark:border-white border-white h-10 w-10 mr-2 -mt-3 bg-Warm-Blue  rounded-xl `} onClick={() => {}}><RiDragMove2Fill className="h-full w-full text-white p-1"/></div>

      <div className={`flex ${!collapsed ? '' : 'w-0 hidden'}`}>
                      <div title="Adjust Node Size" className="duration-300 cursor-pointer shadow-xl border-2 dark:border-white border-white h-10 w-10 mr-2 -mt-3 bg-Warm-Violet rounded-xl" onClick={() => {setSizeAdjuster((size) => !size)}}><TbResize className="h-full w-full text-white p-1"/></div>
                      <a href={data.host} target="_blank" rel="noopener noreferrer"><div title="Gradio Host Site" className="duration-300 cursor-pointer shadow-xl border-2 dark:border-white border-white h-10 w-10 mr-2 -mt-3 bg-Warm-Pink rounded-xl"><BiCube className="h-full w-full text-white p-1"/></div></a>
                      <div title="Delete Node" className="duration-300 cursor-pointer shadow-xl border-2 dark:border-white border-white h-10 w-10 mr-2 -mt-3 bg-Warm-Red rounded-xl" onClick={() => data.delete([{id : id}])}><BsTrash className="h-full w-full text-white p-1"/></div>
                      <div title="Refresh Node" className="duration-300 cursor-pointer shadow-xl border-2 dark:border-white border-white h-10 w-10 mr-2 -mt-3 bg-Warm-Orange rounded-xl" onClick={() => {}}><BiRefresh className="h-full w-full text-white p-1"/></div>
        </div>
      </div>

      { !collapsed && reachable && <>
          <animated.div className={`border-dashed ${sizeAdjuster ? 'border-4 border-black dark:border-white' : ''} relative top-0 left-0 z-[1000] touch-none shadow-lg rounded-xl duration-300`} style={{width, height }} {...bind()}>
            <div id="draggable" className={`absolute h-full w-full ${data.colour} shadow-2xl rounded-xl -z-20`}></div>
            <iframe id="iframe" 
                        key={refresh}
                        src={data.host} 
                        title={data.label}
                        frameBorder="0"
                        className=" p-[0.6rem] -z-10 h-full w-full ml-auto mr-auto overflow-y-scroll"/>
              <div className={` ${sizeAdjuster ? '' : 'hidden'} rounded-full border-2 absolute -bottom-4 -right-4 w-7 h-7 bg-Blue-Royal cursor-nwse-resize touch-none shadow-lg`} ref={dragElement}>
            </div>  
            </animated.div>
            </>
        }

        { collapsed  &&
              <div id={`draggable`}
                   className={` w-[340px] h-[140px]  text-white text-md flex flex-col text-center items-center cursor-grab shadow-lg
                                p-5 px-2 rounded-md break-all -z-20 ${data.colour} hover:opacity-70 duration-300`} onClick={() => setCollapsible(collapsed => !collapsed)}>
                  <div  className="absolute text-6xl opacity-60 z-10 pt-8 ">{data.emoji}</div>    
                  <h2 className={`max-w-full font-sans text-blue-50 leading-tight font-bold text-3xl flex-1 z-20 pt-10`} style={{"textShadow" : "0px 1px 2px rgba(0, 0, 0, 0.25)"}} >{data.label}</h2>
              </div > 
        }
      </div>)
}
