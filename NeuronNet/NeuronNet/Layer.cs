using System;
using System.Linq;

namespace NeuronNet
{
    internal class Layer
    {
        public double Bias { get; set; }
        public int NeuronCount => _neurons.Length;
        public LayerType Type { get; }

        private static readonly Random Rnd = new Random();
        private readonly double[][] _neurons;

        /// <summary>
        /// Creates a layer with the given parameters
        /// </summary>
        /// <param name="previousLayerNeuronCount">The count of neurons in the previous layer</param>
        /// <param name="neuronCount">The count of neurons in this layer</param>
        /// <param name="layerType">The type of neurons in this layer</param>
        public Layer(int previousLayerNeuronCount, int neuronCount, LayerType layerType)
        {
            //Set Bias to 1
            Bias = 1;

            //Set own type
            Type = layerType;

            //Create neurons with a synapse to all neurons of the previous Layer.
            _neurons = new double[neuronCount][];

            //Initalize synapses with random values.
            for (int i = 0; i < _neurons.GetLength(0); i++)
            {
                _neurons[i] = new double[previousLayerNeuronCount + 1];

                for (int j = 0; j < _neurons[i].Length; j++)
                    _neurons[i][j] = Rnd.NextDouble() * 2 - 1;
            }
        }

        /// <summary>
        /// Claculates the neuron values for a forward pass in this layer
        /// </summary>
        /// <param name="input">The values of the previous layer</param>
        /// <returns>The processed values</returns>
        public double[] ForwardPass(double[] input)
        {
            //Check input size
            if(input.Count() != _neurons[0].Length - 1)
                throw new ArgumentException("Invalid input array size !");

            //Add bias to input
            Array.Resize(ref input, input.Length + 1);
            input[input.Length - 1] = Bias;

            var result = new double[NeuronCount];

            //Calculate neuron values
            for (var i = 0; i < NeuronCount; i++)
                result[i] = ActivationFunction(NeuronMath.MatrixMultiply(_neurons[i], input));

            return result;
        }

        /// <summary>
        /// Returns the neuron delta values for an output layer
        /// </summary>
        /// <param name="isValues">The values the neurons in this layer have</param>
        /// <param name="wantValues">The values the neurons in this layer should have</param>
        /// <returns>The delta values for the neurons</returns>
        public double[] GetDeltaOutputLayer(double[] isValues, double[] wantValues)
        {
            //Create result array
            var result = new double[NeuronCount];

            //Calculate delta values
            for (int i = 0; i < result.Length; i++)
                 result[i] = NeuronMath.GetDeltaOutputLayer(Derivative(isValues[i]), isValues[i], wantValues[i]);

            return result;
        }

        /// <summary>
        /// Generates the calc values for the previous layer
        /// </summary>
        /// <param name="ownDelta">The delta values of this layer</param>
        /// <returns>The calc values for the previous layer</returns>
        public double[] GetPreviousLayerCalcValues(double[] ownDelta)
        {
            //Create result array
            var result = new double[_neurons[0].Length - 1];

            //Calculate calc values
            for (int i = 0; i < NeuronCount; i++)
                for (int j = 0; j < result.Length; j++)
                    result[i] += ownDelta[i] * _neurons[i][j];

            return result;
        }

        /// <summary>
        /// Updates own synapse weights
        /// </summary>
        /// <param name="ownDelta">The delta values of this network</param>
        /// <param name="previousLayerValues">The values of the previous layer</param>
        /// <param name="learnParameter">The resolution to learn at</param>
        public void UpdateValues(double[] ownDelta, double[] previousLayerValues, double learnParameter)
        {
            //Calculate and add weight deltas
            for (int i = 0; i < ownDelta.Length; i++)
                for (int j = 0; j < previousLayerValues.Length; j++)
                    _neurons[i][j] += NeuronMath.GetWeightDelta(learnParameter, ownDelta[i], previousLayerValues[j]);

            double biasDelta = 0;

            //Calculate bias delta
            for (int i = 0; i < NeuronCount; i++)
                biasDelta += ownDelta[i] * _neurons[i][_neurons[i].Length - 1];

            //Update connection weights to bias
            for (int i = 0; i < NeuronCount; i++)
                _neurons[i][_neurons[i].Length - 1] *= NeuronMath.GetWeightDelta(learnParameter, biasDelta, Bias);
        }

        /// <summary>
        /// Returns the neuron delta values for an output layer
        /// </summary>
        /// <param name="isValues">The values the neurons in this layer have</param>
        /// <param name="calcValues">The calc values of the next layer</param>
        /// <returns>The delta values for the neurons</returns>
        public double[] GetDeltaHiddenLayer(double[] isValues, double[] calcValues)
        {
            //Create result array
            var result = new double[calcValues.Length];

            //Calculate delta values
            for (int i = 0; i < calcValues.Length; i++)
                result[i] = NeuronMath.GetDeltaHiddenLayer(isValues[i], calcValues[i]);

            return result;
        }

        /// <summary>
        /// Returns the result of the activation function in the layer
        /// </summary>
        /// <param name="value">The value to calculate</param>
        /// <returns>The result</returns>
        private double ActivationFunction(double value)
        {
            //Check own type
            switch (Type)
            {
                //Case Sigmoid:
                case LayerType.Sigmoid:
                    //Return Sigmoid of value
                    return NeuronMath.Sigmoid(value);

                //This should not happen
                default:
                    throw new Exception("This should not happen !");
            }
        }

        /// <summary>
        /// Returns the derivative of the activation function in the layer
        /// </summary>
        /// <param name="value">The value to calculate</param>
        /// <returns>The result</returns>
        private double Derivative(double value)
        {
            //Check own type
            switch (Type)
            {
                //Case Sigmoid:
                case LayerType.Sigmoid:
                    //Return SigmoidDerivative of value
                    return NeuronMath.SigmoidDerivative(value);

                //This should not happen
                default:
                    throw new Exception("This should not happen !");
            }
        }
    }
}
