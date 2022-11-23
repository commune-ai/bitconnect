import React from 'react';
import {BiX} from 'react-icons/bi'
import { getBezierPath, getEdgeCenter, useReactFlow } from 'react-flow-renderer';
import '../../css/dist/output.css';
const foreignObjectSize = 40;

export default function CustomEdge({
  id,
  source,
  target,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style = {},
  markerEnd,
  data
}) {
  const edgePath = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });
  const [edgeCenterX, edgeCenterY] = getEdgeCenter({
    sourceX,
    sourceY,
    targetX,
    targetY,
  });
  const { getEdge } = useReactFlow();

  return (
    <>
      <path
        id={id}
        style={style}
<<<<<<< HEAD
        className="react-flow__edge-path dark:stroke-white stroke-Deep-Space-Black bg-white"
=======
<<<<<<< HEAD
        className="react-flow__edge-path dark:stroke-white stroke-Deep-Space-Black bg-white"
=======
        className="react-flow__edge-path dark:stroke-white stroke-Deep-Space-Black bg-white "
>>>>>>> fd926ecf2fa48a23e5fec18c84fb4cf28d09417f
>>>>>>> 80464945cc7e284bdc13b838cf648407f2b98c7c
        d={edgePath}
      />
      <foreignObject
        width={foreignObjectSize}
        height={200}
        x={edgeCenterX - foreignObjectSize / 2}
        y={edgeCenterY - foreignObjectSize / 2}
        className="edgebutton-foreignobject"
        requiredExtensions="http://www.w3.org/1999/xhtml"
      >
<<<<<<< HEAD
          <div className="flex hover:border-purple-400 dark:hover:border-purple-400 w-10 h-10 dark:bg-black bg-white dark:border-white border-2 rounded-xl hover:shadow-lg text-center duration-200 " onClick={() => data.delete(getEdge(id), target)}>
=======
<<<<<<< HEAD
          <div className="flex hover:border-purple-400 dark:hover:border-purple-400 w-10 h-10 dark:bg-black bg-white dark:border-white border-2 rounded-xl hover:shadow-lg text-center duration-200 " onClick={() => onEdgeClick(data.delete, id)}>
=======
          <div className="flex hover:border-purple-400 dark:hover:border-purple-400 w-10 h-10 dark:bg-black bg-white dark:border-white border-2 rounded-xl hover:shadow-lg text-center duration-200 " onClick={() => onEdgeClick(data.delete, {id, source, target})}>
>>>>>>> fd926ecf2fa48a23e5fec18c84fb4cf28d09417f
>>>>>>> 80464945cc7e284bdc13b838cf648407f2b98c7c
            <BiX className=' flex-1 w-9 h-9 text-black dark:text-white'/>
          </div>
      </foreignObject>
    </>
  );
}