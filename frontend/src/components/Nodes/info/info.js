import React, { useEffect } from "react";
import {ImInfo} from 'react-icons/im'
import { useReactFlow } from "react-flow-renderer";
import '../../../css/dist/output.css';

export default function Info({display, data}){
    const { getNode } = useReactFlow();

    let dataCopy = JSON.parse(JSON.stringify(data));
    dataCopy.args = dataCopy.args.map(arg => getNode(arg)?.data.dtype)
    
    if (!display) return (<></>)
    return ( <div tabIndex='-1' className="  absolute top-0 left-0 right-0 z-50 w-full h-full p-5 px-10 h-modal">
    <div className="relative w-full h-full break-words">
        <div className="relative rounded-lg shadow-2xl bg-gray-800 border-[1.5px] border-gray-700">

            <div className=" relative p-6 break-all">
                <ImInfo className="ml-auto mr-auto w-10 h-10 text-white"/>
                <h2 className="mt-3 text-xl font-semibold text-white text-center">Config</h2>
                <div className="mt-6 p-5 bg-slate-700 rounded-md border-slate-600 border-[1.5px] shadow-2xl">
                <pre className=" text-white whitespace-pre-wrap"><code>{JSON.stringify(dataCopy, null, 3)}</code></pre>
                
                </div>
            </div>
        </div>
    </div>
</div>)
}