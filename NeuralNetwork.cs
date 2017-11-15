using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_networks
{
    public class NeuralNetwork
    {
        #region Data Members

        private NodeTable nodeTable_;
        private LinkTable linkTable_;

        #endregion

        #region Constructors

        public NeuralNetwork(Func<double, double> nonLinearity, params int[] layerSizes)
        {
            nodeTable_ = new NodeTable(nonLinearity, layerSizes);
            linkTable_ = new LinkTable(nodeTable_);            
        }

        public NeuralNetwork(Func<double, double> nonLinearity, List<Matrix> weights)
        {
            if (weights.Count < 1)
                throw new Exception("Too few link layers!");

            var layerSizes = new List<int>();
            layerSizes.Add(weights[0].Height);

            for (int i = 0; i < weights.Count - 1; i++)
            {
                if (weights[i].Width == weights[i + 1].Height)
                {
                    layerSizes.Add(weights[i + 1].Height);
                }
                else
                    throw new Exception("Dimension mismatch");
            }
            layerSizes.Add(weights[weights.Count - 1].Width);

            nodeTable_ = new NodeTable(nonLinearity, layerSizes.ToArray());
            linkTable_ = new LinkTable(nodeTable_);
            linkTable_.LoadLinks(weights);
        }

        #endregion

        #region Operations

        public void LoadWeights(List<Matrix> L)
        {
            this.linkTable_.LoadLinks(L);
        }

        public void LoadWeights(int baseLayerIndex, Matrix W)
        {
            this.linkTable_.LoadLinks(baseLayerIndex, W);
        }

        public void ApplyInputs(List<double> inputValues)
        {
            if (inputValues.Count != Inputs.Count)
            {
                throw new Exception("Size Mismatch!!!!!");
            }

            for (int i = 0; i < inputValues.Count; i++)
            {
                Inputs[i].Value = inputValues[i];
            }
        }

        public static NeuralNetwork CreateRandom()
        {
            // Generate random sizes for input and output layer sizes
            Random r = new Random();
            int inputSize = r.Next(10) + 1;
            int outputSize = r.Next(10) + 1;
            return CreateRandom(inputSize, outputSize);
        }

        public static NeuralNetwork CreateRandom(int inputSize, int outputSize)
        {
            // Generate random number of hidden layers
            Random r = new Random();

            int numLayers = r.Next(5) + 2;
            int[] layerSizes = new int[numLayers + 2];

            layerSizes[0] = inputSize;

            // Generate random sizes for each layer
            for (int i = 1; i < numLayers - 1; i++)
            {
                layerSizes[i] = r.Next(10) + 1;
            }

            layerSizes[numLayers - 1] = outputSize;

            // Generate random weights between each layer
            List<Matrix> weights = new List<Matrix>(numLayers - 1);
            for (int i = 0; i < numLayers - 1; i++)
            {
                double[,] matrix = new double[layerSizes[i], layerSizes[i + 1]];
                for (int j = 0; j < layerSizes[i]; j++)
                    for (int k = 0; k < layerSizes[i + 1]; k++)
                        matrix[j, k] = r.NextDouble();
                weights.Add(matrix);
            }

            return new NeuralNetwork(Math.Tanh, weights);
        }

        public double CalculateExpectedLoss(List<double> expectedValues)
        {
            double acc = 0.0;
            if (expectedValues.Count != Outputs.Count)
                throw new Exception("Size mismatch");

            for (int i = 0; i < expectedValues.Count; i++)
            {
                double e = (Outputs[i].Value - expectedValues[i]);
                acc += e * e;                   
            }

            return 0.5 * acc;
        }

        #endregion

        #region Fields

        public List<InputNode> Inputs
        {
            get
            {
                return nodeTable_.Inputs;
            }
        }

        public List<OutputNode> Outputs
        {
            get
            {
                return nodeTable_.Outputs;
            }
        }

        public double this[int i, int j, int k]
        {
            get
            {
                return this.linkTable_[i, j, k];
            }
            set
            {
                this.linkTable_[i, j, k] = value;
            }
        }

        #endregion
    }
}
