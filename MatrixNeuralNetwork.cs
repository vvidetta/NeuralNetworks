using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_networks
{
    public delegate double RealFunc(double x);

    public class MatrixNeuralNetwork
    {
        #region Data Members

        private List<Matrix> weights_;
        private RealFunc activationFunc_;
        private List<Matrix> activations_ = new List<Matrix>();
        private List<Matrix> sums_ = new List<Matrix>();

        #endregion

        #region Constructors

        public MatrixNeuralNetwork(RealFunc activation, params int[] layerSizes)
        {
            int neuralLayers = layerSizes.Length - 1;
            weights_ = new List<Matrix>(neuralLayers);

            for (int i = 0; i < neuralLayers; i++)
            {
                weights_.Add(new Matrix(layerSizes[i], layerSizes[i + 1]));
            }

            activationFunc_ = activation;
        }

        public MatrixNeuralNetwork(RealFunc activation, List<Matrix> weights)
        {
            weights_ = weights;
            activationFunc_ = activation;
        }

        #endregion

        #region Operations

        public void LoadWeights(List<Matrix> L)
        {
            weights_ = L;
        }

        public void LoadWeights(int baseLayerIndex, Matrix W)
        {
            weights_[baseLayerIndex] = W;
        }

        public void Evaluate(Matrix Input)
        {
            sums_.Clear();
            activations_.Clear();

            activations_.Add(Input);
            for (int i = 0; i < weights_.Count; i++)
            {
                EvaluateLayer(this.activations_[i], this.weights_[i]);
            }
        }

        public Matrix CalculateSum(Matrix layerInput, Matrix layerWeights)
        {
            return layerInput * layerWeights;
        }

        public Matrix CalculateActivation(Matrix sumMatrix)
        {
            return Matrix.ApplyFunction(this.activationFunc_, sumMatrix);
        }

        public void EvaluateLayer(Matrix layerInput, Matrix layerWeights)
        {
            Matrix sum = CalculateSum(layerInput, layerWeights);
            sums_.Add(sum);
            activations_.Add(CalculateActivation(sum));
        }

        public static MatrixNeuralNetwork CreateRandom()
        {
            // Generate random sizes for input and output layer sizes
            Random r = new Random();
            int inputSize = r.Next(10) + 1;
            int outputSize = r.Next(10) + 1;
            return CreateRandom(inputSize, outputSize, Program.sigmoid);
        }

        public static MatrixNeuralNetwork CreateRandom(int inputSize, int outputSize, RealFunc activation)
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
                weights.Add(Matrix.CreateRandom(layerSizes[i], layerSizes[i + 1]));

            return new MatrixNeuralNetwork(activation, weights);
        }

        #endregion

        #region Fields

        public RealFunc ActivationFunction
        {
            get { return this.activationFunc_; }
        }

        public List<Matrix> Activations
        {
            get { return this.activations_; }
        }

        public Matrix Input
        {
            get { return this.activations_.First(); }
        }

        public Matrix Output
        {
            get { return this.activations_.Last(); }
        }

        public List<Matrix> Sums
        {
            get { return this.sums_; }
        }

        public double this[int i, int j, int k]
        {
            get
            {
                return weights_[i][j, k];
            }
            set
            {
                weights_[i][j,k] = value;
            }
        }

        public List<Matrix> Weights
        {
            get { return this.weights_; }
        }

        #endregion
    }
}
