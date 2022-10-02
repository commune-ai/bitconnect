import React, { Component } from "react";
import { Icon, Loader } from 'semantic-ui-react'
import Import from '../Modal/importer'
import { random_colour, random_emoji } from "./utils";

import "../../css/dist/output.css"
import '../../css/index.css'
import {BsArrowLeftShort} from 'react-icons/bs';
// import {ReactComponent as ReactLogo} from '../../images/logo.svg'

export default class Navbar extends Component{
    constructor(props){
        super(props) 
        this.state = {
            open : true,
            menu : [],
            colour : props.colour || [],
            text : "",
            name : "",
            emoji : props.emoji || [],
            mode : false,
            modal : false,
            error : false,
            loading : false
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
        await fetch(`http://localhost:8000/list?${new URLSearchParams({ mode : "gradio" })}`, { method: 'GET', mode : 'cors',})
            .then(response => response.json())
            .then(data => {
                    this.handelTabs(this.state.menu, data)
                    this.setState({loading : false})
                    this.setState({menu : data.sort(function(x, y) {return (x.read === y.read)? 0 : x.read? -1 : 1;})})
                }).catch(error => {console.log(error)}) 
    }

    /**
     * Append new node from the user 
     */
    appendStreamNode = async (type) => {
        // const pattern = {
        //     local : /^https?:\/\/(localhost)*(:[0-9]+)?(\/)?$/,
        //     share : /^https?:\/\/*([0-9]{5})*(-gradio)*(.app)?(\/)?$/,
        //     hugginFace : /^https?:\/\/*(hf.space)\/*(embed)\/*([a-zA-Z0-9+_-]+)\/*([a-zA-Z0-9+_-]+)\/*([+])?(\/)?$/
        // } 
        return
    }

    /**
     * error check the user input
     * @param {*} bool boolean of the current state of the modal  
     */
    handelModal = (bool) => {
        this.setState({'error' : !bool ? false : this.state.error ,
                       'modal' : bool})
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
        event.dataTransfer.setData('application/style', JSON.stringify({colour : this.state.colour[index], emoji : this.state.emoji[index] }))
        event.dataTransfer.setData('application/item',  item)
        event.dataTransfer.effectAllowed = 'move';
      };


    /**
     * update the tabs within the navbar
     * @param {*} e current menu 
     * @param {*} d integer variable of the diffence between the current menu and new menu updated ment
     */
    handelTabs = async (e, d) => {
        // if less then 0 we must remove colour's and emoji's
        // get index of the object
        // remove
        var c = []
        var j = []
        if (d.length - e.length === 0) return
        else if(d.length - e.length < 0){
            var a = this.state.menu.filter(item => e.includes(item)) // get the items not in menu anymore
            c = this.state.colour
            j = this.state.emoji
            for(var k=0; k < d.length; k++){
                c.splice(this.state.menu.indexOf(a[k]), 1)
                j.splice(this.state.menu.indexOf(a[k]), 1)
            }
            this.setState({'colour' : c, 'emoji' : j})
        }else{
            //append new colours
            for(var i =0; i < d.length; i++){
                    c.push(random_colour(i === 0 ? null : c[i-1]));
                    j.push(random_emoji(i === 0 ? null : c[i-1]));
                
            }
            const colour = [...this.state.colour]
            const emoji  = [...this.state.emoji]
            this.setState({'colour' : [...colour, ...c], 'emoji' : [...emoji, ...j],})
        }
    }

    handelError = (boolean) => {
        this.setState({'error' : boolean})
    }

    /**
     * handel navagation open and close function
     */
    handelNavbar = () => {
        this.setState({'open' : !this.state.open})
    }

    /**
     * 
     * @param {*} e : event type to get the target value of the current input
     * @param {*} type : text | name string that set the changed value of the input to the current value 
     */
    updateText = (e, type) => {
        this.setState({[`${type}`] : e.target.value })
    }

    /**
     * 
     * @param {*} item : object infomation from the flask api  
     * @param {*} index : current index with in the list
     * @returns div component that contians infomation of gradio 
     */
    subComponents(item, index){
        return(<>
                <li key={`${index}-li`} onDragStart={(event) => this.onDragStart(event, 'custom', item.path, index)} 
                    className={` text-white text-md flex flex-col text-center items-center cursor-grab shadow-lg
                                 p-5 px-2 mt-4 rounded-md ${ this.state.open ? `${item.read ? "hover:animate-pulse" : "opacity-50 select-none"}  ${this.state.colour[index] === null ? "" : this.state.colour[index]} ` : `hidden`}  break-all -z-20`} draggable={item.read}>

                    <div key={`${index}-div`}  className=" absolute -mt-2 text-4xl opacity-60 z-10 ">{`${this.state.emoji[index] === null ? "" : this.state.emoji[index]}`}</div>    
                    <h4 key={`${index}-h4`}  className={`  max-w-full font-sans text-blue-50 leading-tight font-bold text-xl flex-1 z-20  ${this.state.open ? "" : "hidden"}`} style={{"textShadow" : "0px 1px 2px rgba(0, 0, 0, 0.25)"}} >{`${item.path}`} </h4>

                </li >      

        </>)
    }


    render(){
        
        return (<div>
        
            <div className={`z-10 flex-1 float-left bg-white dark:bg-stone-900 h-screen p-5 pt-8 ${this.state.open ? "lg:w-72 md:64 sm:w-60" : "w-10"} duration-300 absolute shadow-2xl border-black border-r-[1px] dark:border-white dark:text-white`} >

            <BsArrowLeftShort onClick={this.handelNavbar} className={`  bg-white text-Retro-darl-blue text-3xl rounded-full absolute -right-3 top-9 border border-black cursor-pointer ${!this.state.open && 'rotate-180'} dark:border-white duration-300 dark:text-white dark:bg-stone-900 `}/>

                <div className="inline-flex w-full">
                    <h1 className={`font-sans font-bold text-lg ${this.state.open ? "" : "hidden"} duration-500 ml-auto mr-auto`}> {/*<ReactLogo className="w-9 h-9 ml-auto mr-auto"/>*/}Modular Flow 🌊 </h1>
                </div>

                <div className={`rounded-md text-center ${this.state.open ? "" : "px-0"} py-3`} onClick={() => {}}>
                    <div className={` text-center bg-transparent w-full h-10 border border-slate-300 hover:border-Retro-purple hover:animate-pulse border-dashed rounded-md py-2 pl-5 ${this.state.open ? "pr-3" : "hidden"} shadow-sm sm:text-sm`}>
                        <Icon className=" block mr-auto ml-auto" name="plus"/>
                    </div>
                </div>
                <Import open={this.state.modal} 
                        quitHandeler={this.handelModal}
                        textHandler={this.updateText}
                        appendHandler={this.appendStreamNode}
                        handelError={this.handelError}
                        catch={this.state.error}/>
                <div id="module-list" className={`relative z-10 w-full h-[90%] overflow-auto ${this.state.loading ? " animate-pulse duration-300 bg-neutral-900 rounded-lg" : ""} `}>
                    <ul className="overflow-hidden">
                    {this.state.loading &&<Loader active/>}
                    {this.state.menu.map((item, index) => {return this.subComponents(item, index)})}
                    </ul>
                </div>

            </div>
            
        </div>)
    }
}
