using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.AccessControl;
using System.Text;
using System.Threading.Tasks;

namespace NeuronNet
{
    public class Network
    {
        public int InputSize { get; }
        public int OutputSize { get; }

        private Layer[] _layers;

        /// <summary>
        /// Generates a nework with the given parameters
        /// </summary>
        /// <param name="inputSize">The size of the input</param>
        /// <param name="layers">The layers in this network</param>
        private Network(int inputSize, Layer[] layers)
        {
            //Set Layers
            _layers = layers;

            //Set Input Size
            InputSize = inputSize;

            //Set OutputSize to size of last Layer
            OutputSize = layers.Last().NeuronCount;
        }

        /// <summary>
        /// Calculates the output for the given input
        /// </summary>
        /// <param name="input">The input</param>
        /// <returns>The output</returns>
        public double[] Calulate(double[] input)
        {
            //Check input size
            if(input.Length != InputSize)
                throw new ArgumentException("Input has to be the same size as InputSize!");

            //Calculate output
            return _layers.Aggregate(input, (current, t) => t.ForwardPass(current));
        }

        /// <summary>
        /// Calculates the error of the netowrk with the given inout, output
        /// </summary>
        /// <param name="input">The input</param>
        /// <param name="expectedOutput">The expected output</param>
        /// <returns>The error of the network</returns>
        public double GetError(double[] input, double[] expectedOutput)
        {
            //Check input size
            if (input.Length != InputSize)
                throw new ArgumentException("input has to be the same size as InputSize!");

            //Check outputSize size
            if (expectedOutput.Length != OutputSize)
                throw new ArgumentException("expectedOutput has to be the same size as OutputSize!");

            //Calculate result
            var result = Calulate(input);
            double error = 0;

            //Calculate error
            for (int i = 0; i < result.Length; i++)
                error += Math.Pow(result[i] - expectedOutput[i], 2);

            return error;
        }


        /// <summary>
        /// Trains the network to the given input, output pair
        /// </summary>
        /// <param name="input">The input</param>
        /// <param name="expectedOutput">The expected output</param>
        /// <param name="learnParameter">The resolution to train the network at</param>
        public void Train(double[] input, double[] expectedOutput, double learnParameter)
        {
            //Check input size
            if (input.Length != InputSize)
                throw new ArgumentException("input has to be the same size as InputSize!");

            //Check outputSize size
            if (expectedOutput.Length != OutputSize)
                throw new ArgumentException("expectedOutput has to be the same size as OutputSize!");

            //Create array for the layer results
            var layerResults = new double[_layers.Length][];

            //Calulate layer results
            for (int i = 0; i < _layers.Length; i++)
            {
                if (i == 0)
                    layerResults[i] = _layers[i].ForwardPass(input);
                else
                    layerResults[i] = _layers[i].ForwardPass(layerResults[i - 1]);
            }

            //Create array for the layer deltas
            var layerDeltas = new double[_layers.Length][];

            //Calculate layer deltas
            for (int i = layerDeltas.Length - 1; i >= 0; i--)
            {
                if (i == layerDeltas.Length - 1)
                    layerDeltas[i] = _layers[i].GetDeltaOutputLayer(layerResults[i], expectedOutput);
                else
                    layerDeltas[i] = _layers[i]
                        .GetDeltaHiddenLayer(layerResults[i],
                            _layers[i + 1].GetPreviousLayerCalcValues(layerDeltas[i + 1]));
            }

            //Update synapse weights
            for (int i = layerDeltas.Length - 1; i >= 0; i--)
                _layers[i].UpdateValues(layerDeltas[i], i == 0 ? input : layerResults[i - 1], learnParameter);
        }


        /// <summary>
        /// Creates neural networks
        /// </summary>
        public class Factory
        {
            /// <summary>
            /// The size of the input of the neural network
            /// </summary>
            public int InputSize { get; set; }

            /// <summary>
            /// The layers in the neuronal network
            /// </summary>
            public List<KeyValuePair<LayerType, int>> Layers { get; set; } = new List<KeyValuePair<LayerType, int>>();

            /// <summary>
            /// Appends an layer to the neuroal network
            /// </summary>
            /// <param name="type">The type of layer to append</param>
            /// <param name="size">The count of neurons in this layer</param>
            public void AppendLayer(LayerType type, int size)
            {
                //Check size
                if(size < 1)
                    throw new ArgumentException("Size has to be > 0 !");

                //Add to Layers
                Layers.Add(new KeyValuePair<LayerType, int>(type, size));
            }

            /// <summary>
            /// Creates the neural network
            /// </summary>
            /// <returns>The neural network</returns>
            public Network Build()
            {
                //Check InputSize
                if(InputSize < 1)
                    throw new InvalidOperationException("InputSize has to be > 0 !");

                //Check Layers
                if (Layers.Count < 1)
                    throw new InvalidOperationException("At least 1 Layers it necessary!");

                var layers = new Layer[Layers.Count];

                //Create Layers
                for (int i = 0; i < layers.Length; i++)
                {
                    //2nd Layer
                    if(i == 0)
                        layers[i] = new Layer(InputSize, Layers[i].Value, Layers[i].Key);

                    //Everything else
                    else
                        layers[i] = new Layer(layers[i - 1].NeuronCount, Layers[i].Value, Layers[i].Key);
                }

                //Create and return Netowork
                return new Network(InputSize, layers);
            }
        }
    }
}
