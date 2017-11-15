using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_networks
{
    public class GradientDescentOptimiserMethod : IOptimiserMethod
    {
        #region Data Members

        private MatrixLossFunction lossFunction_;
        private MatrixNeuralNetwork net_;
        private double newError_;
        private List<Matrix> newWeights_;
        private MNNOptimiserOptions options_;
        private TrainingSet trainingSet_;

        #endregion

        #region Constructors

        public GradientDescentOptimiserMethod(MatrixNeuralNetwork net, TrainingSet trainingSet, MatrixLossFunction lossFunc, MNNOptimiserOptions options)
        {
            this.net_ = net;
            this.trainingSet_ = trainingSet;
            this.lossFunction_ = lossFunc;
            this.options_ = options;
        }

        #endregion

        #region Operations

        public void Step(double currentError, List<Matrix> WeightDerivatives)
        {
            this.newWeights_ = new List<Matrix>(this.net_.Weights.Count);
            for (int i = 0; i < this.net_.Weights.Count; i++)
            {
                this.newWeights_.Add(this.net_.Weights[i] - this.options_.learningRate * WeightDerivatives[i]);
            }

            MatrixNeuralNetwork newNet = new MatrixNeuralNetwork(this.net_.ActivationFunction, this.newWeights_);
            this.newError_ = this.lossFunction_(newNet, this.trainingSet_);
            if (this.newError_ >= currentError)
            {
                this.newWeights_ = this.net_.Weights;
                this.newError_ = currentError;
            }
        }

        #endregion

        #region Fields

        public double NewError
        {
            get { return this.newError_; }
        }

        public List<Matrix> NewWeights
        {
            get { return this.newWeights_; }
        }

        #endregion
    }
}
