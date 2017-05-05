using System;
using System.Linq;

namespace NeuronNet
{
    internal class NeuronMath
    {
        public static double MatrixMultiply(double[] trans1, double[] trans2)
        {
            return trans1.Sum(t1 => trans2.Sum(t => t1 * t));
        }

        public static double Sigmoid(double d)
        {
            return 1 / (1 + Math.Exp(-d));
        }

        public static double SigmoidDerivative(double d)
        {
            return d * (1 - d);
        }

        public static double GetWeightDelta(double learnParameter, double delta, double activationLevel)
        {
            return learnParameter * delta * activationLevel;
        }

        public static double GetDeltaOutputLayer(double derivative, double value, double want)
        {
            return derivative * (want - value);
        }

        public static double GetDeltaHiddenLayer(double derivative, double calcValue)
        {
            return derivative * calcValue;
        }
    }
}
