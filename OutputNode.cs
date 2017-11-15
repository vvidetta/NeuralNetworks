using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_networks
{
    public class OutputNode : Node, IOutput, INonLinear
    {
        #region Data Members

        private CachedValue<double> cache_ = new CachedValue<double>();
        private List<IInput> inputs_ = new List<IInput>();
        private Func<double, double> nonLinearity_ = null;

        #endregion

        #region Constructors

        public OutputNode()
        {
            cache_ = new CachedValue<double>(0.0, this.ValueUpdater);
            cache_.HasChanged += OnCacheHasChanged;
            cache_.IsChanging += OnCacheIsChanging;
        }

        public OutputNode(Func<double, double> nonLinearity)
            : this()
        {
            this.nonLinearity_ = nonLinearity;
        }

        #endregion

        #region Operations

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
 	        // No-op
        }

        public void OnInputIsChanging(object sender, EventArgs e)
        {
            this.cache_.RequiresUpdate = true;
        }

        public double ValueUpdater()
        {
            double acc = 0.0;
            foreach (IInput input in inputs_)
            {
                acc += input.Value;
            }
            if (nonLinearity_ != null)
            {
                acc = nonLinearity_(acc);
            }
            return acc;
        }

        #endregion

        #region Fields

        public List<IInput> Inputs
        {
            get { return this.inputs_; }
        }

        public override double Value
        {
            get
            {
                return this.cache_.Value;
            }
        }

        public Func<double, double> NonLinearity
        {
            get { return this.nonLinearity_; }
            set { this.nonLinearity_ = value; }
        }

        #endregion
    }
}
