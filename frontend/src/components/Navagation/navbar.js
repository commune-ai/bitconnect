import React, { Component } from "react";
import { Loader } from 'semantic-ui-react'
import { random_colour, random_emoji } from "./utils";

import "../../css/dist/output.css"
import '../../css/index.css'
import {BsArrowLeftShort} from 'react-icons/bs';
import {ReactComponent as Gradio} from '../../images/gradio.svg'
import {ReactComponent as Streamlit} from '../../images/streamlit.svg'


export default class Navbar extends Component{
    constructor(props){
        super(props) 
        this.state = {
            open : true,
            stream : [],
            menu : [],
            style : { colour : {}, emoji : {}},
            mode : false,
            loading : false,
            toggle : 'gradio',
            search : ""
           }
       
    }

    componentDidMount(){
        this.fetch_classes()   
    }

    /**
     *  Asynchronously call the Flask api server every second to check if there exist a gradio application info
     *  @return null
     */
    fetch_classes = async () => {
        this.setState({loading : true})
        await fetch(`http://localhost:8000/list?${new URLSearchParams({ mode : "streamable" })}`, { method: 'GET', mode : 'cors',})
            .then(response => response.json())
            .then(data => {
                    this.handelModule(this.state.menu, Object.keys(data))
                    this.setState({loading : false})
                    this.setState({menu : Object.keys(data).sort(function(x, y) {return (x === y)? 0 : x? -1 : 1;}), stream : data})
                }).catch(error => {console.log(error)}) 
    }


    /**
     * when dragged get all the information needed
     * @param {*} event 
     * @param {*} nodeType string 'custom' node type
     * @param {*} item object information returned from the api
     * @param {*} index current index
     */
    onDragStart = (event, nodeType, item, index) => {
        event.dataTransfer.setData('application/reactflow', nodeType);
        event.dataTransfer.setData('application/style', JSON.stringify({colour : this.state.style.colour[item], emoji : this.state.style.emoji[item], stream : this.state.toggle }))
        event.dataTransfer.setData('application/item',  item)
        event.dataTransfer.effectAllowed = 'move';
      };


    // /**
    //  * update the tabs within the navbar
    //  * @param {*} e current menu 
    //  * @param {*} d integer variable of the diffence between the current menu and new menu updated ment
    //  */
    
    async handelModule(currentMenu, newMenu){
        var style = {colour : {}, emoji : {}};
        var prevState = null;
        if (currentMenu.length === newMenu.length) return 
        else if ( newMenu.length - currentMenu.length < 0){
            /** FIX LATER */
            for(var i = 0; i < newMenu.length; i++){
                style["colour"][newMenu[i]] = random_colour(prevState === null ? null : prevState["colour"])
                style["emoji"][newMenu[i]] = random_emoji(prevState === null ? null : prevState["emoji"])
                prevState = {colour : style["colour"][newMenu[i]], emoji : style["emoji"][newMenu[i]]}
            }
        }
        else {  
            for(var i = 0; i < newMenu.length; i++){
                style["colour"][newMenu[i]] = random_colour(prevState === null ? null : prevState["colour"])
                style["emoji"][newMenu[i]] = random_emoji(prevState === null ? null : prevState["emoji"])
                prevState = {colour : style["colour"][newMenu[i]], emoji : style["emoji"][newMenu[i]]}
            }
        }
        this.setState({style : style})
    }

    /**
     * handel navagation open and close function
     */
    handelNavbar = () => {
        this.setState({'open' : !this.state.open})
    }

    handelToggle(){
        console.log(this.state.toggle)
        this.setState({'toggle' : this.state.toggle === "gradio" ? "streamlit" : "gradio"})
    }

    /**
     * 
     * @param {*} e : event type to get the target value of the current input
     * @param {*} type : text | name string that set the changed value of the input to the current value 
     */
    updateText = (e, {name, value}) => this.setState({[`${name}`] : value }) 

    /**
     * 
     * @param {*} item : object infomation from the flask api  
     * @param {*} index : current index with in the list
     * @returns div component that contians infomation of gradio 
     */
    subComponents(item, index){
        console.log(this.state.style.colour)
        return(<>
                <li key={`${index}-li`} onDragStart={(event) => this.onDragStart(event, 'custom', item, index)} 
                    className={` text-white text-md flex flex-col text-center items-center cursor-grab shadow-lg
                                 p-5 px-2 mt-4 rounded-md ${ this.state.open ? `hover:animate-pulse ${this.state.style.colour[item] === null ? "" : this.state.style.colour[item]} ` : `hidden`}  break-all -z-20`} draggable={true}>

                    <div key={`${index}-div`}  className=" absolute -mt-2 text-4xl opacity-60 z-10 ">{`${this.state.style.emoji[item] === null ? "" : this.state.style.emoji[item]}`}</div>    
                    <h4 key={`${index}-h4`}  className={`  max-w-full font-sans text-blue-50 leading-tight font-bold text-xl flex-1 z-20  ${this.state.open ? "" : "hidden"}`} style={{"textShadow" : "0px 1px 2px rgba(0, 0, 0, 0.25)"}} >{`${item}`} </h4>

                </li >      

        </>)
    }

    render(){
        
        return (<div>
        
            <div className={`z-10 flex-1 float-left bg-white dark:bg-stone-900 h-screen p-5 pt-8 ${this.state.open ? "w-80" : "w-10"} duration-300 absolute shadow-2xl border-black border-r-[1px] dark:border-white dark:text-white`} >

            <BsArrowLeftShort onClick={this.handelNavbar} className={`  bg-white text-Retro-darl-blue text-3xl rounded-full absolute -right-3 top-9 border border-black cursor-pointer ${!this.state.open && 'rotate-180'} dark:border-white duration-300 dark:text-white dark:bg-stone-900 `}/>

                <div className="inline-flex w-full pb-3">
                    <h1 className={`font-sans font-bold text-lg ${this.state.open ? "" : "hidden"} duration-500 ml-auto mr-auto`}> {/*<ReactLogo className="w-9 h-9 ml-auto mr-auto"/>*/}Modular Flow 🌊 </h1>
                </div>

                <div className={`${this.state.open ? 'mb-5' : 'hidden'} flex`}>
                    <div className={` w-14 h-7 flex items-center border-2 ${this.state.toggle === "gradio" ? 'bg-white border-orange-400' : ' bg-slate-800'}  shadow-xl rounded-full p-1 cursor-pointer float-left duration-300 `} onClick={() => {this.handelToggle()}}>
                        <Streamlit className=" absolute w-5 h-5"/>
                        <Gradio className=" absolute w-5 h-5 translate-x-6"/>
                    <div className={`border-2 h-[1.42rem] w-[1.42rem] rounded-full shadow-md transform duration-300 ease-in-out  ${this.state.toggle === "gradio" ? ' bg-orange-400 transform -translate-x-[0.2rem]' : " bg-red-700 transform translate-x-[1.45rem] "}`}></div>
                    </div>
                    
                    <form>
                        <div className="relative ml-2">
                            <div className="flex absolute inset-y-0 left-0 items-center pl-3 pointer-events-none">
                                <svg aria-hidden="true" className="w-4 h-4 text-gray-500 dark:text-gray-200" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path></svg>
                            </div>
                            <input type="search" name="search" id="default-search" value={this.state.search} className="block p-1 pl-10 w-full text-sm text-gray-900 bg-gray-50 rounded-lg border border-gray-300 focus:ring-blue-500 focus:border-blue-500 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-300 focus:placeholder-gray-100 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500 focus:shadow-xl" onChange={(e) => this.updateText(e , {name : 'search', value : e.target.value})} placeholder="Search Module..." required/>
                        </div>
                    </form>



                </div>
  
                <div id="module-list" className={` relative z-10 w-full h-[92%] overflow-auto ${this.state.loading ? " animate-pulse duration-300 bg-neutral-900 rounded-lg bottom-0" : ""} `}>
                    <ul className="overflow-hidden rounded-lg">
                    {this.state.loading &&<Loader active/>}
                    {this.state.menu.filter((value) => {
                        if (this.state.stream[value][this.state.toggle]){
                            if (this.state.search.replace(/\s+/g, '') === "" || value.toLocaleLowerCase().includes(this.state.search.replace(/\s+/g, '').toLocaleLowerCase())) 
                            return value 
                        }
                    }).map((item, index) => {return this.subComponents(item, index)})}
                    </ul>
                </div>

            </div>
            
        </div>)
    }
}