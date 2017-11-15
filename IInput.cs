using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_networks
{
    public interface IInput : IEvaluable
    {
        event HasChangedHandler HasChanged;

        event IsChangingHandler IsChanging;

        List<IOutput> Outputs { get; }

        void AddOutput(IOutput output);
    }
}
