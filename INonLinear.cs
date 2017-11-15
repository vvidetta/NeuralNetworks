using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_networks
{
    public interface INonLinear
    {
        Func<double, double> NonLinearity { get; set; }
    }
}
