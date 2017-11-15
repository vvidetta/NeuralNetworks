using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_networks
{
    public abstract class Node : IEvaluable
    {
        #region Data Members
        #endregion

        #region Constructors
        #endregion

        #region Operations
        #endregion

        #region Fields

        public virtual double Value
        {
            get { return 0.0; }
            set
            {
                // No-op
            }
        }

        public string Name
        {
            get;
            set;
        }

        #endregion
    }
}
