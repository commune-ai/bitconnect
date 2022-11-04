import dagre from 'dagre'

// import styled from 'styled-components'
// import { animated } from '@react-spring/web'


const dagreGraph = new dagre.graphlib.Graph();
dagreGraph.setDefaultEdgeLabel(() => ({}));

const nodeWidth = 172;
const nodeHeight = 36;

export const getLayoutedElements = (nodes, edges, direction = 'LR') => {
  const isHorizontal = direction === 'LR';
  dagreGraph.setGraph({ rankdir: direction });

  nodes.forEach((node) => {
    dagreGraph.setNode(node.id, { width: nodeWidth, height: nodeHeight });
  });

  edges.forEach((edge) => {
    dagreGraph.setEdge(edge.source, edge.target);
  });

  dagre.layout(dagreGraph);

  nodes.forEach((node) => {
    const nodeWithPosition = dagreGraph.node(node.id);
    node.targetPosition = isHorizontal ? 'left' : 'top';
    node.sourcePosition = isHorizontal ? 'right' : 'bottom';

    // We are shifting the dagre node position (anchor=center center) to the top left
    // so it matches the React Flow node anchor point (top left).
    node.position = {
      x: nodeWithPosition.x - nodeWidth / 2,
      y: nodeWithPosition.y - nodeHeight / 2,
    };

    return node;
  });

  return { nodes, edges };
};

export const root = {
  id: '1',
  type: 'input',
  sourcePosition: 'right',
  data: { label: 'input', depth : 0 },
  position: { x: 0, y: 0 },
}

const createNode = (label, index) => {
  return {
    id : `${label}-${index}`,
    data: { label: label, },
    sourcePosition: 'right',
    targetPosition: 'left',
    position : { x : index, y : index},
  }
}

const createEdge = (start, end) => {
  return {
    id: `${start}-${end}`,
    source : start, 
    target : end,
    type: 'smoothstep', 
  }
}

export const bfs = (roots, nodes=[],edges=[] ) => {

  // ========================
  // Initialize variables
  // ======================== 
  let Q = Object.keys(roots).map((key) => { return {[`${key}`] : roots[key]} }) // Initialize queue nodes
  var children = null // Initialize children nodes
  let v = null; // Initialize children nodes

  // attach the edges from the true root
  Q.forEach((item, index) => {
    if (Object.keys(item).includes("module")) edges.push(createEdge(root.id, `${item.module}-${index}`))
    else edges.push(createEdge(root.id, `${Object.keys(item)[0]}-${index}`))
  })

  while (Q.length){ // while Queue is not 0
    v = Q.shift() // deqeue from queue
    if (Object.keys(v).includes("module")) { // 
      nodes.push(createNode(v.module, nodes.length));
      continue;
    } else {
      nodes.push(createNode(Object.keys(v)[0], nodes.length ))
    }
    for (const value of Object.values(v)) { // value is always length of 1 so this loop runs O(1)
      children = Object.keys(value).map((key) => { return {[`${key}`] : value[key]} })
      Q = [...Q, ...children]
      children.forEach((item, index)=>{
        if (Object.keys(item).includes("module")) edges.push(createEdge(`${Object.keys(v)[0]}-${nodes.length - 1}`, `${item.module}-${nodes.length + Q.length - children.length + index }`))
        else edges.push(createEdge(`${Object.keys(v)[0]}-${nodes.length - 1}`, `${Object.keys(item)[0]}-${nodes.length + Q.length - children.length + index}`))
      })

    }
  }
  return {nodes, edges}
}

/**
 * 
 * @returns 
 */
 export const useThemeDetector = () => {
  const getCurrentTheme = () => window.matchMedia("(prefers-color-scheme: dark)").matches;
  return getCurrentTheme();
}

// import styled from 'styled-components'
// import { animated } from '@react-spring/web'

// export const Main = styled.div`
//   cursor: pointer;
//   color: #676767;
//   -webkit-user-select: none;
//   user-select: none;
//   display: flex;
//   align-items: center;
//   height: 100%;
//   justify-content: center;
// `

// export const Container = styled.div`
//   position: fixed;
//   z-index: 1000;
//   width: 0 auto;
//   bottom: 30px;
//   margin: 0 auto;
//   left: 30px;
//   right: 30px;
//   display: flex;
//   flex-direction: column;
//   pointer-events: none;
//   align-items: flex-end;
//   @media (max-width: 680px) {
//     align-items: center;
//   }
// `

// export const Message = styled(animated.div)`
//   box-sizing: border-box;
//   position: relative;
//   overflow: hidden;
//   width: 40ch;
//   @media (max-width: 680px) {
//     width: 100%;
//   }
// `

// export const Content = styled.div`
//   color: white;
//   background: #445159;
//   opacity: 0.9;
//   padding: 12px 22px;
//   font-size: 1em;
//   display: grid;
//   grid-template-columns: 1fr auto;
//   grid-gap: 10px;
//   overflow: hidden;
//   height: auto;
//   border-radius: 3px;
//   margin-top: 10px;
// `

// export const Button = styled.button`
//   cursor: pointer;
//   pointer-events: all;
//   outline: 0;
//   border: none;
//   background: transparent;
//   display: flex;
//   align-self: flex-end;
//   overflow: hidden;
//   margin: 0;
//   padding: 0;
//   padding-bottom: 14px;
//   color: rgba(255, 255, 255, 0.5);
//   :hover {
//     color: rgba(255, 255, 255, 0.6);
//   }
// `

// export const Life = styled(animated.div)`
//   position: absolute;
//   bottom: 0;
//   left: 0px;
//   width: auto;
//   background-image: linear-gradient(130deg, #00b4e6, #00f0e0);
//   height: 5px;
// `
