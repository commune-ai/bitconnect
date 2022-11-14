import React, { useCallback, useEffect, useState } from "react"
import { Handle, Position, useReactFlow, useStoreApi } from "react-flow-renderer"

import '../../css/dist/output.css'

const itemList = ["String", "Integer", "Float", "Json", "Image"];

export default function InputCompnent({ id, data }) {
  const [open, setOpen] = useState(false)
  const [item, setItem] = useState("String")

  const { setNodes, getEdges, getNode } = useReactFlow();
  const store = useStoreApi();


  useEffect(() => {
    const { nodeInternals } = store.getState();
    const children = getEdges(id).map(edg => edg.target) // n
    setNodes(
      Array.from(nodeInternals.values()).map((node) => {
        if (node.id === id) {
          node.data = {
            ...node.data,
            dtype: item,
          };
        } else if (children.includes(node.id) && node.id.includes('bundle-node-args__custom-')) {
          const index = node.data.args.findIndex(args => args.id === id)
          const newList = node.data.args
          newList[index] = getNode(id)
          node.data = {
            ...node.data,
            args: newList
          }
        }
        return node;
      }))
  }, [item, id, setNodes, store, getEdges, getNode])


  const onChange = useCallback((value) => {
    const { nodeInternals } = store.getState();
    setNodes(
      Array.from(nodeInternals.values()).map((node) => {
        if (node.id === id) {
          node.data = {
            ...node.data,
            dtype: item,
            value: value
          };
        }
        return node;
      })
    );
  }, [item, id, setNodes, store])

  return (<div className="w-[300px] border-2 rounded-lg shadow-lg border-black bg-white dark:bg-stone-800 dark:border-white p-2 duration-300">

    <button id="dropdownDefault" onClick={(e) => { setOpen((open) => !open); }} data-dropdown-toggle="dropdown" className={`w-full text-white bg-[#7b3fe4] focus:ring-4 focus:outline-none focus:ring-[#F48CF4] font-medium ${open ? 'rounded-t-lg' : 'rounded-lg'} text-sm px-4 py-2.5 text-center inline-flex items-center dark:bg-[#7b3fe4] dark:focus:ring-blue-800 hover:shadow-lg duration-300`} type="button">{item}
      <svg className={`ml-2 w-4 h-4 right-5 absolute ${open ? '' : 'rotate-180'} duration-150`} aria-hidden="true" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"></path></svg></button>

    <div id="dropdown" className={`w-[282px] fixed ${open ? '' : 'invisible'} z- bg-white rounded-b-lg divide-y divide-gray-100 shadow-lg dark:bg-stone-700`}>
      <ul className="py-0 text-sm text-gray-700 dark:text-gray-200 z-[1000]" aria-labelledby="dropdownDefault">
        {itemList.filter((v) => v !== item).map((value) => {
          return (<li onClick={() => { setItem(value); onChange(null); setOpen((open) => !open) }}>
            <span className="block py-2 px-4 hover:bg-stone-100 dark:hover:bg-stone-600 dark:hover:text-white">{value}</span>
          </li>)
        })}
      </ul>
    </div>


    {item === "String" &&
      <div className="pt-2 rounded-lg">
        <textarea id="message" rows="10" onChange={(e) => onChange(e.target.value)} className="resize-none block p-2.5 w-full text-sm text-gray-900 bg-gray-50 rounded-lg border border-gray-600 focus:ring-violet-600/100 focus:border-purple-400 dark:bg-stone-800 dark:border-gray-300 focus:ring-1 dark:placeholder-gray-400 dark:text-white dark:focus:ring-[#8a52ea] dark:focus:border-[#8a52ea] focus:shadow-lg duration-300" placeholder="Enter Input..."></textarea>
      </div>}

    {["Integer", "Float"].includes(item) &&
      <div className="flex pt-2 rounded-lg">
        <span className="inline-flex items-center px-3 text-sm text-gray-900 bg-gray-200 rounded-l-md border border-r-0 border-gray-300 dark:bg-stone-700 dark:text-gray-200 dark:border-gray-300">{item === "Integer" ? "ℤ" : "ℕ"}</span>
        <input type="number"
          id="number-input"
          onChange={(e) => { onChange(item === "Integer" ? Math.round(+e.target.value) : +e.target.value) }}
          className="rounded-none rounded-r-lg bg-gray-50 border text-gray-900 focus:ring-1 focus:shadow-lg focus:ring-[#7b3fe4] focus:border-[#7b3fe4] block flex-1 min-w-0 w-full text-sm border-gray-300 p-2.5 dark:bg-stone-800 dark:border-gray-300 dark:placeholder-gray-400 dark:text-white dark:focus:ring-[#8b5bde] dark:focus:border-[#7b3fe4]"
          placeholder={item === "Integer" ? "Integers" : "Float"}></input>
      </div>}

    <Handle type="source"
      id="output"
      position={Position.Right}
      style={{ "marginRight": "-12px", "marginTop": "0px", "height": "25px", "width": "25px", "borderRadius": "3px", "zIndex": "10000", "background": "white", "boxShadow": "3px 3px #888888", 'position': 'absolute' }} />

    {["Image", "Json"].includes(item) &&
      <div className="flex justify-center items-center w-full hover:shadow-lg mt-2 rounded-lg duration-300">
        <label htmlFor="dropzone-file" className="flex flex-col justify-center items-center w-full h-64 bg-gray-50 rounded-lg border-2 border-gray-300 border-dashed cursor-pointer  dark:bg-stone-800 hover:bg-gray-100 dark:border-gray-300 dark:hover:border-[#7b3fe4] duration-300">
          <div className="flex flex-col justify-center items-center pt-5 pb-6 px-2">
            <svg aria-hidden="true" className="mb-3 w-10 h-10 text-gray-500 dark:text-blue-50" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path></svg>
            <p className="mb-2 text-sm text-gray-500 dark:text-blue-50"><span className="font-semibold">Click to upload</span> or drag and drop</p>
            <p className="text-xs text-gray-500 dark:text-blue-50">{item === "Json" ? "JSON" : 'SVG, PNG, JPG or GIF'}</p>
          </div>
          <input id="dropzone-file" type="file" accept={item === "Json" ? "application/JSON" : ""} className="hidden" />
        </label>
      </div>}
  </div>)
}