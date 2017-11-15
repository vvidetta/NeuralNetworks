using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_networks
{
    public interface IOutput : IEvaluable
    {
        void AddInput(IInput input);

        List<IInput> Inputs { get; }

        void OnInputHasChanged(object sender, EventArgs e);

        void OnInputIsChanging(object sender, EventArgs e);
    }
}
