import "./App.css";
import React from "react";
import Sketch from "react-p5";
import Matrix  from './components/Matrix.js'
import { NeuralNetwork } from "./components/NeuralNetwork";

function App() {

  let learning_data = [
    {
      inputs: [0,0],
      targets: [0]
    },
    {
      inputs: [1,1],
      targets: [0]
    },
    {
      inputs: [0,1],
      targets: [1]
    },
    {
      inputs: [1,0],
      targets: [1]
    }
  ]
  
  const setup = (p5, canvasParentRef) => {
    // use parent to render the canvas in this ref
    // (without that p5 will render the canvas outside of your component)
    p5.createCanvas(600, 600).parent(canvasParentRef);
    /* let nn = new NeuralNetwork(2,2,1) */
  };

  const nn: NeuralNetwork =new NeuralNetwork(2,10,1)

  const draw = (p5) => {
    
    
    for(let i = 0; i < 5000; i++ ){
      let o = Math.floor(Math.random() * 4);
      nn.train(learning_data[o].inputs, learning_data[o].targets)
    }
    p5.background(0)
    let resolution = 10

    let clos = Math.floor(p5.width / resolution );
    let rows = Math.floor(p5.height / resolution);

    for (let i = 0; i< clos ; i++){
      for (let j = 0; j< rows; j++){
        let x = i * resolution 
        let y = j * resolution
        let input_1 = i / (clos -1);
        let input_2 = j / (rows -1);
        let output = nn.feedforward([input_1, input_2]);
        let col = output[0] * 255;
        p5.fill(col);
        p5.noStroke();
        p5.rect(x, y, resolution, resolution);
      }
    }
  };

  return <Sketch setup={setup} draw={draw} className="prova" />;
}

export default App;
