using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_networks
{
    public class HiddenNode: Node, IInput, IOutput, INonLinear
    {
        #region Data Members

        private CachedValue<double> cache_;
        private Func<double, double> nonlinearity_ = null;
        private List<IInput> inputs_ = new List<IInput>();
        private List<IOutput> outputs_ = new List<IOutput>();

        #endregion

        #region Constructors

        public HiddenNode()
        {
            cache_ = new CachedValue<double>(0.0, this.ValueUpdater);
            cache_.HasChanged += OnCacheHasChanged;
            cache_.IsChanging += OnCacheIsChanging;
        }

        public HiddenNode(Func<double, double> nonLinearity)
            : this()
        {
            this.nonlinearity_ = nonLinearity;
        }

        #endregion

        #region Operations

        public event HasChangedHandler HasChanged;

        public event IsChangingHandler IsChanging;

        public void OnCacheHasChanged(object sender, EventArgs e)
        {
            if (HasChanged != null)
            {
                HasChanged(this, e);
            }
        }

        public void OnCacheIsChanging(object sender, EventArgs e)
        {
            if (IsChanging != null)
            {
                IsChanging(this, e);
            }
        }

        public void OnInputHasChanged(object sender, EventArgs e)
        {
            
        }

        public void OnInputIsChanging(object sender, EventArgs e)
        {
            cache_.RequiresUpdate = true;
        }

        public void AddInput(IInput input)
        {
            if (!inputs_.Contains(input))
            {
                inputs_.Add(input);
                input.HasChanged += OnInputHasChanged;
                input.IsChanging += OnInputIsChanging;
                cache_.RequiresUpdate = true;
                input.AddOutput(this);
            }
        }

        public void AddOutput(IOutput output)
        {
            if (!this.outputs_.Contains(output))
            {
                outputs_.Add(output);
            }
        }

        public double ValueUpdater()
        {
            double acc = 0.0;
            foreach (IInput input in inputs_)
            {
                acc += input.Value;
            }
            if (nonlinearity_ != null)
            {
                acc = nonlinearity_(acc);
            }
            return acc;
        }

        #endregion

        #region Fields

        public List<IInput> Inputs
        {
            get { return inputs_; }
        }

        public Func<double, double> NonLinearity
        {
            get { return this.nonlinearity_; }
            set { this.nonlinearity_ = value; }
        }

        public List<IOutput> Outputs
        {
            get { return outputs_; }
        }

        public override double Value
        {
            get { return this.cache_.Value; }
        }

        #endregion
    }
}
