using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_networks
{
    public class InputNode : Node, IInput
    {
        #region Data Members

        private double value_;
        private List<IOutput> outputs_ = new List<IOutput>();

        #endregion

        #region Constructors

        public InputNode() { }

        public InputNode(double initialValue)
        {
            this.value_ = initialValue;
        }

        #endregion

        #region Operations
        
        public void AddOutput(IOutput output)
        {
            if (!this.outputs_.Contains(output))
            {
                this.outputs_.Add(output);
            }
        }

        public event HasChangedHandler HasChanged;

        public event IsChangingHandler IsChanging;

        #endregion

        #region Fields

        public List<IOutput> Outputs
        {
            get { return outputs_; }
        }

        public override double Value
        {
            get { return this.value_; }
            set
            {
                if (this.value_ != value)
                {
                    double oldValue = this.value_;
                    if (IsChanging != null)
                    {
                        IsChanging(this, null);
                    }
                    this.value_ = value;
                    if (HasChanged != null)
                    {
                        HasChanged(this, null);
                    }
                }
            }
        }

        #endregion
    }
}
