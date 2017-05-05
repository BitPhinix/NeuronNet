# What is NeuronNet ?

A fast, lightweight and easy to use neural network library written in C#.

# Usage
```C#
//Create a new NetworkFactory
var networkFactory = new Network.Factory();

//Set input size to 2
networkFactory.InputSize = 2;

//Add 2 layers
networkFactory.AppendLayer(LayerType.Sigmoid, 2);
networkFactory.AppendLayer(LayerType.Sigmoid, 1);

//Build the network
var network = networkFactory.Build();

//Generate training data
var input = new double[] { 0.123, 0.3466 };
var output = new double[] { 0.723 };

while (true)
{
      //Train the network
      network.Train(input, output, 0.01);

      //Print out the error of the network
      Console.WriteLine("Error: " + network.GetError(input, output));
}
```

Or take a look at the sample project [here](https://github.com/BitPhinix/NeuronNet/blob/master/NeuronNet/SampleProject/Program.cs).

# How it works

The neural network is trained using [backpropagation](https://en.wikipedia.org/wiki/Backpropagation).

If you are interested, you can watch this youtube video (It's really awesome):

<a href="http://www.youtube.com/watch?feature=player_embedded&v=q555kfIFUCM
" target="_blank"><img src="http://img.youtube.com/vi/q555kfIFUCM/0.jpg" 
alt="Backpropagation in 5 Minutes" width="240" height="180" border="10" /></a>
