using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_networks
{
    public interface IOptimiserMethod
    {
        void Step(double currentError, List<Matrix> WeightDerivatives);

        double NewError { get; }

        List<Matrix> NewWeights { get; }
    }
}
