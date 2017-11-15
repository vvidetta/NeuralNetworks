using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_networks
{
    public class MomentumOptimiserMethod : IOptimiserMethod
    {
        #region Data Members

        private MatrixLossFunction lossFunction_;
        private double momentum_ = 0.9;
        private MatrixNeuralNetwork net_;
        private double newError_;
        private List<Matrix> newWeights_;
        private MNNOptimiserOptions options_;
        private TrainingSet trainingSet_;
        private List<Matrix> velocity_;

        #endregion

        #region Constructors

        public MomentumOptimiserMethod(MatrixNeuralNetwork net, TrainingSet trainingSet, MatrixLossFunction lossFunc, MNNOptimiserOptions options)
        {
            this.net_ = net;
            this.trainingSet_ = trainingSet;
            this.lossFunction_ = lossFunc;
            this.options_ = options;
            this.velocity_ = new List<Matrix>(net.Weights.Count);
            for (int i = 0; i < net.Weights.Count; i++)
            {
                this.velocity_.Add(new Matrix(net.Weights[i].Height, net.Weights[i].Width));
            }
        }

        #endregion

        #region Operations

        public void Step(double currentError, List<Matrix> WeightDerivatives)
        {
            this.newWeights_ = new List<Matrix>(this.net_.Weights.Count);
            for (int i = 0; i < this.net_.Weights.Count; i++)
            {
                this.velocity_[i] = this.momentum_ * this.velocity_[i] - this.options_.learningRate * WeightDerivatives[i]; 
                this.newWeights_.Add(this.net_.Weights[i] + this.velocity_[i]);
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
