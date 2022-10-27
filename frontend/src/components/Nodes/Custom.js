import React, { useEffect, useRef, useState } from "react"
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

export default function CustomNodeIframe({id, data}){
    const [collapsed, setCollapsible] = useState(true)
    const [{x, y, width, height}, api] = useSpring(() => ({width: MINIMUM_WIDTH, height: MINIMUM_HEIGHT }))
    const [sizeAdjuster, setSizeAdjuster] = useState(false)
    const [reachable ,setReachable] = useState(false)
    const [refresh, setRefresh] = useState(0)
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

    const isFetchable = async () => {
      return fetch(data.host, {mode: 'no-cors'}).then((res) => {
        return true
      }).catch((err)=>{
        return false
      })
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
      
      <div id={'draggable'}className=" flex w-full h-10 top-0 cursor-pointer" onClick={() => {}}>
      <div id={'draggable'} title={collapsed ? "Collaspse Node" : "Expand Node"} className=" flex-none duration-300 cursor-pointer shadow-xl border-2 border-white h-10 w-10 mr-2 -mt-3 bg-Warm-Blue rounded-xl" onClick={() => {setCollapsible((clps) => !clps)}}><CgLayoutGridSmall className="h-full w-full text-white p-1"/></div>

      <div className={` flex ${!collapsed ? '' : 'w-0 hidden'}`}>
                      <div title="Adjust Node Size" className="duration-300 cursor-pointer shadow-xl border-2 dark:border-white border-white h-10 w-10 mr-2 -mt-3 bg-Warm-Violet rounded-xl" onClick={() => {setSizeAdjuster((size) => !size)}}><TbResize className="h-full w-full text-white p-1"/></div>
                      <a href={data.host} target="_blank" rel="noopener noreferrer"><div title="Gradio Host Site" className="duration-300 cursor-pointer shadow-xl border-2 dark:border-white border-white h-10 w-10 mr-2 -mt-3 bg-Warm-Pink rounded-xl"><BiCube className="h-full w-full text-white p-1"/></div></a>
                      <div title="Delete Node" className="duration-300 cursor-pointer shadow-xl border-2 dark:border-white border-white h-10 w-10 mr-2 -mt-3 bg-Warm-Red rounded-xl" onClick={() => data.delete([{id : id}])}><BsTrash className="h-full w-full text-white p-1"/></div>
                      <div title="Refresh Node" className="duration-300 cursor-pointer shadow-xl border-2 dark:border-white border-white h-10 w-10 mr-2 -mt-3 bg-Warm-Orange rounded-xl" onClick={() => {setRefresh((old) => old++)}}><BiRefresh className="h-full w-full text-white p-1"/></div>
        </div>
      </div>

      { !collapsed && reachable && <>
          <animated.div className={`border-dashed  ${sizeAdjuster ? 'border-4 border-white' : ''} relative top-0 left-0 z-[1000] touch-none shadow-lg rounded-xl`} style={{width, height }} {...bind()}>
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


// export default class CustomNodeIframe extends React.Component {
//     constructor({id , data}){
//       super()
//       this.myRef = React.createRef()
//       this.original_width = 0;
//       this.original_height = 0;
//       this.original_x = 0;
//       this.original_y = 0;
//       this.original_mouse_x = 0;
//       this.original_mouse_y = 0;  

//       this.state = {
//         id : id,
//         reachable : false,
//         selected : false,
//         data : data,
//         width : 540,
//         height : 600,
//         size : false,
//         iframe : 0,
//         loading : true
//       }
 
//     }
    
//     componentDidMount(){
//       this.fetched = setInterval(
//         async () => {
//           const fetch = await this.isFetchable(this.state.data.host)
//           if (fetch){
//             clearInterval(this.fetched)
//             this.setState({'iframe' : this.state.iframe + 1, reachable : true, loading: false })

//           } else {
//             this.setState({selected : false, loading : true}) 
//           } 
//         },1000) 
//     }

//     componentWillUnmount(){
//       clearInterval(this.fetched)
//     }

//     handelSelected = async () => {
//        const fetchable = await this.isFetchable(this.state.data.host) 
//       if (!fetchable){
//         this.setState({selected : false}) 
//       } else {
//         if(!this.state.reachable){
//           this.setState(prevState => ({'selected' : !prevState.selected, 'size' : false, 'iframe' : prevState.iframe + 1, reachable : true }))
//         }else
//           this.setState(prevState => ({'selected' : !prevState.selected, 'size' : false}))
//       }
//     }

//     handelSizeState = () => {
//         this.setState(prevState => ({'size' : !prevState.size}))
//       }

//     isFetchable = async (host) => {
//       return fetch(host, {mode: 'no-cors'}).then((res) => {
//         return true
//       }).catch((err)=>{
//         return false
//       })

//     }

//     onNodeClick = (id) => {
//       this.state.data.delete({id : id})
//     }

//      onRefresh = async () => {
//       const fetchable = await this.isFetchable(this.state.data.host)
//       if(!fetchable){ 
//         this.onNodeClick(this.state.id)
//       } else{
//         this.setState(prevState => ({'iframe' : prevState.iframe + 1}))
//       }
//     }

//     handelOnChange(evt, type){
//       this.setState({[`${type}`] : parseInt(evt.target.value) })
//       type === "width" ? this.myRef.current.style.width = `${parseInt(evt.target.value)}px` : this.myRef.current.style.height = `${parseInt(evt.target.value)}px` 
//     }

//     handelSize(evt, increment, change){
//       if (evt === "increment") {
//         this.setState({[`${change}`] :  change === "width" ? this.state.width + increment : this.state.height + increment })
//         change === "width" ? this.myRef.current.style.width = `${this.state.width + increment}px` : this.myRef.current.style.height = `${this.state.height + increment}px` 
//       }

//     }
    
//     //resize nodes by dragging
//     initial = (e) => {
//       this.original_width = this.myRef.current.offsetWidth
//       this.original_height = this.myRef.current.offsetHeight

//       this.original_x = this.myRef.current.getBoundingClientRect().left;
//       this.original_y = this.myRef.current.getBoundingClientRect().top;

//       this.original_mouse_x = e.clientX
//       this.original_mouse_y = e.clientY
//     }

//     resize = (e, point) => {
//       var height = 0;
//       var width = 0;
//       // e.dataTransfer.setDragImage(new Image(), 0, 0)
//       if (point === 'bottom-right'){
//         width = this.original_width + (e.clientX - this.original_mouse_x);
//         height = this.original_height + (e.clientY - this.original_mouse_y)
//         if (width > MINIMUM_WIDTH) {
//           this.myRef.current.style.width = `${width}px`
//           this.setState({'width' :  parseInt(width) , 'height' : parseInt(height)})

//         }
//         if (height > MINIMUM_HEIGHT) {
//           this.myRef.current.style.height = `${height}px`
//           this.setState({'width' :  parseInt(width) , 'height' : parseInt(height)})

//         }
//       } 
//     }

//     OnDragEnd = () => {
//       this.setState({'width' : parseInt(this.myRef.current.offsetWidth), 'height' : parseInt(this.myRef.current.offsetHeight)})
//     }

//     Counter(focus, size){
//       return (<div className="custom-number-input h-10 w-32 dark:text-white text-black ">
//                 <div className="flex flex-row h-10 w-full rounded-lg relative bg-transparent">
//                   <button data-action="decrement" className=" border-2 border-dotted border-Retro-dark-blue hover:border-rose-700 rounded-l-xl hover:animate-pulse h-full w-20 cursor-pointer outline-none " onClick={(e)=> {this.handelSize("increment", -5, focus)}}>
//                     <span className="m-auto text-2xl font-bold">−</span>
//                   </button>
//                   <input type="number" className="focus:outline-none border-Retro-dark-blue border-y-2 border-dotted text-center w-full font-semibold text-md focus:from-fuchsia-200 md:text-basecursor-default flex items-cente outline-none bg-transparent" name="input-number" value={size} onChange={(e) => this.handelOnChange(e, focus)} onKeyDown={(e) => {this.handelSize(e.key, size, focus)}}></input>
//                   <button data-action="increment" className="border-2 border-dotted border-Retro-dark-blue hover:border-green-400 rounded-r-xl hover:animate-pulse h-full w-20  cursor-pointer" onClick={(e)=> {this.handelSize("increment", 5, focus)}} >
//                     <span className="m-auto text-2xl font-bold">+</span>
//                   </button>
//                 </div>
//               </div>)
//     }
        
//     render(){
//       // if (!this.state.reachable) {this.onNodeClick(this.state.id) }
//       return (<>
//                   <div className=" flex w-full h-10 top-0 cursor-pointer" onClick={this.handelEvent} onKeyDown={(e)=>{console.log(e)}}>
//                   <div title={this.state.selected ? "Collaspse Node" : "Expand Node"} className=" duration-300 cursor-pointer shadow-xl border-2 border-white h-10 w-10 mr-2 -mt-3 bg-Warm-Blue rounded-xl" onClick={this.handelSelected}><CgLayoutGridSmall className="h-full w-full text-white p-1"/></div>

    
//                     <div className={` flex ${this.state.selected ? '' : 'w-0 hidden'}`}>
//                       <div title="Adjust Node Size" className="duration-300 cursor-pointer shadow-xl border-2 dark:border-white border-white h-10 w-10 mr-2 -mt-3 bg-Warm-Violet rounded-xl" onClick={this.handelSizeState}><TbResize className="h-full w-full text-white p-1"/></div>
//                       <a href={this.state.data.host} target="_blank" rel="noopener noreferrer"><div title="Gradio Host Site" className="duration-300 cursor-pointer shadow-xl border-2 dark:border-white border-white h-10 w-10 mr-2 -mt-3 bg-Warm-Pink rounded-xl"><BiCube className="h-full w-full text-white p-1"/></div></a>
//                       <div title="Delete Node" className="duration-300 cursor-pointer shadow-xl border-2 dark:border-white border-white h-10 w-10 mr-2 -mt-3 bg-Warm-Red rounded-xl" onClick={() => this.state.data.delete([{id : this.state.id}])}><BsTrash className="h-full w-full text-white p-1"/></div>
//                       <div title="Refresh Node" className="duration-300 cursor-pointer shadow-xl border-2 dark:border-white border-white h-10 w-10 mr-2 -mt-3 bg-Warm-Orange rounded-xl" onClick={() => this.onRefresh()}><BiRefresh className="h-full w-full text-white p-1"/></div>

//                       { this.state.size && <div className="duration-300 flex w-auto h-full  mr-2 -mt-3 space-x-4">
//                         {this.Counter("width", this.state.width)}
//                         {this.Counter("height", this.state.height)}
//                       </div>}
//                     </div>
                  
//                   </div>               
                
//                   <div id={`draggable`} className={`relative overflow-hidden m-0 p-0 shadow-2xl ${this.state.selected ? "w-[540px] h-[600px]" : "hidden"} duration-200`} ref={this.myRef}>

//                     <div className={`absolute p-5 h-full w-full ${this.state.data.colour} shadow-2xl rounded-xl -z-20`}></div>
//                       <iframe 
//                         id="iframe" 
//                         key={this.state.iframe}
//                         src={this.state.data.host} 
//                         title={this.state.data.label}
//                         frameBorder="0"
//                         className="p-2 -z-10 h-full w-full ml-auto mr-auto overflow-y-scroll" 
//                         sandbox="allow-forms allow-modals allow-popups allow-popups-to-escape-sandbox allow-same-origin allow-scripts"
//                         ></iframe>
//                   </div>
                  
                  // <Handle type="target"
                  //         id="input"
                  //         position={Position.Left}
                  //         style={!this.state.selected ? 
                  //               {"paddingRight" : "5px" , "marginTop" : "15px", "height" : "25px", "width" : "25px",  "borderRadius" : "3px", "zIndex" : "10000", "background" : "white", "boxShadow" : "3px 3px #888888"}
                  //               :{"paddingRight" : "5px" ,"height" : "25px", "width" : "25px",  "borderRadius" : "3px", "zIndex" : "10000", "background" : "white", "boxShadow" : "3px 3px #888888"}}
                  //         />
                  
                  // {/*Output*/}
                  // <Handle type="source" id="output" position={Position.Right} style={ !this.state.selected ?
                  //     {"paddingLeft" : "5px", "marginTop" : "15px" ,"height" : "25px", "width" : "25px",  "borderRadius" : "3px", "zIndex" : "10000", "background" : "white", "boxShadow" : "3px 3px #888888"}
                  //     : {"paddingLeft" : "5px", "marginTop" : "0px" ,"height" : "25px", "width" : "25px",  "borderRadius" : "3px", "zIndex" : "10000", "background" : "white", "boxShadow" : "3px 3px #888888"}}/>

//                   {
//                     !this.state.selected &&
                    // <div 
                    // id={`draggable`}
                    // className={` w-[340px] h-[140px]  text-white text-md flex flex-col text-center items-center cursor-grab shadow-lg
                    //              p-5 px-2 rounded-md break-all -z-20 ${this.state.data.colour} ${this.state.loading ? "animate-pulse" : "hover:opacity-70 duration-300"}`}>
                    // {this.state.loading && <>
                    //     {/* <Loader active/> */}
                    //   </>}
                    // <div  className="absolute text-6xl opacity-60 z-10 pt-8 ">{this.state.data.emoji}</div>    
                    // <h2 className={`max-w-full font-sans text-blue-50 leading-tight font-bold text-3xl flex-1 z-20 pt-10`} style={{"textShadow" : "0px 1px 2px rgba(0, 0, 0, 0.25)"}} >{this.state.data.label}</h2>
                    // </div > 
              
//                   }
//                   { this.state.size && !navigator.userAgent.match(/firefox|fxios/i)  && <>
                  
//                   <div id="remove-ghost" className={`absolute select-none -bottom-0 right-0 w-5 h-5border-2 shadow-2xl rounded-xl z-10 cursor-nwse-resize hover:opacity-50  `}
//                        style={{"userDrag": "none"}} 
//                          draggable
                         
//                          >
//                           <BsArrowDownRightSquare className=" text-selected-text text-2xl bg-white"/>
//                           </div> 
  
//                       </>
//                   }
                

//         </>)
//     }
// }