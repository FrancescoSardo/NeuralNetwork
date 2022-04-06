import Matrix from "./Matrix"

function sigmoid(x){
  return 1 / (1+ Math.exp(-x))
}
function dsigmoit(y){
  //! dentro la matrice ho già i valori di sigmoid, quidni non devo ricalcolarli
  //return sigmoid(x) * (1 - sigmoid(x)) // derivata
  return y * (1 - y)
}

export class NeuralNetwork {

  constructor(input_nodes, hiddn_nodes, output_nodes){
    
    this.input_nodes = input_nodes
    this.hiddn_nodes = hiddn_nodes
    this.output_nodes = output_nodes

    this.weights_ih = new Matrix(this.hiddn_nodes,this.input_nodes );
    this.weights_ho = new Matrix(this.output_nodes,this.hiddn_nodes );

    this.weights_ih.randomize()
    this.weights_ho.randomize()
    /* console.table(this.weights_ih.matrix) */
    //!                                       1 sarebbe l'input che è sempre 1
    this.bias_h = new Matrix(this.hiddn_nodes, 1)
    this.bias_o = new Matrix(this.output_nodes, 1)
    this.bias_h.randomize()
    this.bias_o.randomize()

    this.lr = 0.01

  }

  feedforward(input_array){
    let input = Matrix.fromArray(input_array)
    let hidden = Matrix.multiply(this.weights_ih, input)
    hidden.add(this.bias_h)
    hidden.map(sigmoid)
    
    //! ACTIOVATINO FUNCTION . Applay function for every element in matrix
    // exp function in  Math.
    /* hidden.map(sigmoid) */
    // h------> o
    let output = Matrix.multiply(this.weights_ho, hidden)
    output.add(this.bias_o)
    output.map(sigmoid)


    return output.toArray()

  }
  
  train(input_array, targets_array){
    //TODO: rifaccio il feed forward
    let input = Matrix.fromArray(input_array)
    let hidden = Matrix.multiply(this.weights_ih, input)
    hidden.add(this.bias_h)
    hidden.map(sigmoid)
     
    let outputs = Matrix.multiply(this.weights_ho, hidden)
    outputs.add(this.bias_o)
    outputs.map(sigmoid)

    
    let targets = Matrix.fromArray(targets_array)

    /* console.table(ouputs.data)
    console.table(targets.data) */
    //! error = targets - ouputs
    let outputs_errors = Matrix.subtract(targets, outputs)
    /* let gradiente = outputs * ( 1 * outputs ) */
    let gradiente = Matrix.map(outputs, dsigmoit)
    gradiente.multiply(outputs_errors)
    gradiente.multiply(this.lr)
    
    // calculate deltas
    let hiden_transpose = Matrix.transpose(hidden)
    let weight_ho_deltas = Matrix.multiply(gradiente, hiden_transpose)
    
    this.weights_ho.add(weight_ho_deltas)
    this.bias_o.add(gradiente)

    //! HIDDN ERROR 
    let weights_ho_transpose = Matrix.transpose(this.weights_ho)
    let hidden_erorrs = Matrix.multiply(weights_ho_transpose, outputs_errors)

    // calciulate hidden gradiente
    let hidden_gradiente = Matrix.map(hidden, dsigmoit)
    hidden_gradiente.multiply(hidden_erorrs)
    hidden_gradiente.multiply(this.lr)
    

    // calculate input--> hidden deltas
    let input_transpose = Matrix.transpose(input)
    let weight_ih_deltas = Matrix.multiply(hidden_gradiente, input_transpose)

    this.weights_ih.add(weight_ih_deltas)
    this.bias_h.add(hidden_gradiente)

  }

}

