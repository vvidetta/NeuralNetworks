using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_networks
{
    public delegate double MatrixLossFunction(MatrixNeuralNetwork net, TrainingSet trainingSet);

    public struct TrainingSet
    {
        public Matrix Input;
        public Matrix Output;
    }

    public struct MNNOptimiserOptions
    {
        public RealFunc actFuncPrime;
        public double errorBound;
        public double learningRate;
    }

    public class MNNOptimiser
    {
        #region Data Members

        private List<Matrix> sumDerivatives_;
        private List<Matrix> activationDerivatives_;
        private List<Matrix> weightDerivatives_;
        private MatrixLossFunction lossFunction_;
        private MatrixNeuralNetwork net_;
        private MNNOptimiserOptions options_;
        private TrainingSet trainingSet_;

        #endregion

        #region Constructors
        
        public MNNOptimiser(MatrixLossFunction lossFunction)
        {
            this.lossFunction_ = lossFunction;
        }

        #endregion

        #region Operations

        public void Optimise(MatrixNeuralNetwork net, TrainingSet trainingData, MNNOptimiserOptions options)
        {
            this.net_ = net;
            this.trainingSet_ = trainingData;
            this.options_ = options;

            IOptimiserMethod method = new MomentumOptimiserMethod(net, trainingData, this.lossFunction_, options); //new GradientDescentOptimiserMethod(net, trainingData, this.lossFunction_, options);
            double currentError = EvaluateLossFunction();
            int iterNum = 0;
            while (currentError > options_.errorBound)
            {
                //if (iterNum % 3000 == 0)
                    Console.WriteLine("Error at iteration {0} = {1}", iterNum, currentError);

                iterNum++;
                Backpropagate(this.options_.actFuncPrime);
                method.Step(currentError, this.weightDerivatives_);
                if (method.NewError >= currentError)
                {
                    // Minimum found
                    break;
                }

                this.net_.LoadWeights(method.NewWeights);
                currentError = method.NewError;
            }
            Console.WriteLine();
            Console.WriteLine("Network Trained!");
            Console.WriteLine("Error after {0} iterations = {1}", iterNum, currentError);
        }

        private double Descend(double currentError, out List<Matrix> newWeights)
        {
            newWeights = new List<Matrix>(this.net_.Weights.Count);
            for (int i = 0; i < this.net_.Weights.Count; i++)
            {
                newWeights.Add(this.net_.Weights[i] - this.options_.learningRate * this.weightDerivatives_[i]);
            }

            MatrixNeuralNetwork newNet = new MatrixNeuralNetwork(this.net_.ActivationFunction, newWeights);
            double newError = this.lossFunction_(newNet, this.trainingSet_);
            if (newError >= currentError)
            {
                newWeights = this.net_.Weights;
                return currentError;
            }

            return newError;
        }

        private double EvaluateLossFunction()
        {
            return this.lossFunction_(this.net_, this.trainingSet_);
        }

        public void Backpropagate(RealFunc activationPrime)
        {
            InitialiseBackpropagation(activationPrime);
            for (int i = activationDerivatives_.Count - 2; i >= 0; i--)
            {
                BackpropagateLayer(i);
            }
        }

        public void InitialiseBackpropagation (RealFunc activationPrime)
        {
            sumDerivatives_ = InitialiseMatrixList(this.net_.Sums.Count);
            activationDerivatives_ = InitialiseMatrixList(this.net_.Activations.Count);
            weightDerivatives_ = InitialiseMatrixList(this.net_.Weights.Count);

            int lastLayer = this.net_.Activations.Count - 1;
            activationDerivatives_[lastLayer] = this.net_.Activations[lastLayer] - trainingSet_.Output;
        }

        public List<Matrix> InitialiseMatrixList(int numberOfMatrices)
        {
            List<Matrix> matrixList = new List<Matrix>(numberOfMatrices);
            for (int i = 0; i < numberOfMatrices; i++)
            {
                matrixList.Add(null);
            }
            return matrixList;
        }

        public void BackpropagateLayer(int targetLayer)
        {
            Matrix sumLayer = this.net_.Sums[targetLayer];
            Matrix actPrimeSum = Matrix.ApplyFunction(this.options_.actFuncPrime, sumLayer);
            sumDerivatives_[targetLayer] = Matrix.HadamardProduct(actPrimeSum, this.activationDerivatives_[targetLayer + 1]);

            Matrix targetWeightsTranspose = Matrix.Transpose(this.net_.Weights[targetLayer]);
            activationDerivatives_[targetLayer] = sumDerivatives_[targetLayer] * targetWeightsTranspose;

            Matrix targetActivationTranspose = Matrix.Transpose(this.net_.Activations[targetLayer]);
            weightDerivatives_[targetLayer] = targetActivationTranspose * sumDerivatives_[targetLayer];
        }

        #endregion

        #region Fields
        #endregion
    }
}
