using System;
using NeuronNet;

namespace SampleProject
{
    class Program
    {
        /// <summary>
        /// Creates a neural network and trains it.
        /// </summary>
        static void Main(string[] args)
        {
            //Create a new NetworkFactory
            var networkFactory = new Network.Factory();

            //Set input size to 2
            networkFactory.InputSize = 2;

            //Add 2 layers
            networkFactory.AppendLayer(LayerType.Sigmoid, 2);
            networkFactory.AppendLayer(LayerType.Sigmoid, 2);

            //Build the network
            var network = networkFactory.Build();

            //Generate training data
            var input = new [] { 0.3435435, 0.3454357 };
            var output = new [] { 0.45345345, 0.3454354351 };

            for (int i = 1; i <= 100000; i++)
            {
                //Train the network
                network.Train(input, output, 0.07);

                //Print out the error of the network
                Console.WriteLine($"Ideration {i}, Error: {network.GetError(input, output)}");
            }

            //Do a test
            Console.WriteLine("\nTraining finished!\n\n" +
                              "Test:\n" +
                              $"Input: {string.Join(" | ", input)}\n" +
                              $"Output: {string.Join(" | ", network.Calulate(input))}");

            //Prevent instant exit
            Console.ReadLine();
        }
    }
}
