using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_networks
{
    public class Link : IInput, IOutput
    {
        #region Data Members

        private CachedValue<double> cache_ = new CachedValue<double>();
        private double weight_ = 0.0;
        private IInput inputNode;
        private IOutput outputNode;

        #endregion

        #region Constructors

        public Link(IInput input, IOutput output)
        {
            this.AddInput(input);
            output.AddInput(this);
            cache_ = new CachedValue<double>(0.0, this.ValueUpdater);
            cache_.IsChanging += OnCacheIsChanging;
            cache_.HasChanged += OnCacheHasChanged;
        }

        #endregion

        #region Operations

        public static void AddLayer(Dictionary<Tuple<IInput, IOutput>, Link> dict, IEnumerable<IEvaluable> inputs, IEnumerable<IEvaluable> outputs)
        {
            foreach (var input in inputs)
            {
                var inRef = input as IInput;
                foreach (var output in outputs)
                {
                    var outRef = output as IOutput;
                    dict.Add(new Tuple<IInput,IOutput>(inRef, outRef), new Link(inRef, outRef));
                }
            }
        }

        public event HasChangedHandler HasChanged;

        public event IsChangingHandler IsChanging;

        private void OnCacheHasChanged(object sender, EventArgs e)
        {
            if (HasChanged != null)
            {
                HasChanged(this, e);
            }
        }

        private void OnCacheIsChanging(object sender, EventArgs e)
        {
            if (IsChanging != null)
            {
                IsChanging(this, e);
            }
        }
        
        public void OnInputHasChanged(object sender, EventArgs e) { }

        public void OnInputIsChanging(object sender, EventArgs e)
        {
            cache_.RequiresUpdate = true;
        }

        public double ValueUpdater()
        {
            return weight_ * this.inputNode.Value;
        }

        public void AddInput(IInput input)
        {
            this.inputNode = input;
            input.HasChanged += OnInputHasChanged;
            input.IsChanging += OnInputIsChanging;
            cache_.RequiresUpdate = true;
            input.AddOutput(this);
        }

        public void AddOutput(IOutput output)
        {
            this.outputNode = output;
        }

        #endregion

        #region Fields

        public List<IInput> Inputs
        {
	        get { return new List<IInput>(new IInput[1] {inputNode}); }
        }

        public List<IOutput> Outputs
        {
	        get { return new List<IOutput>(new IOutput[1] { outputNode }); }
        }

        public double Value
        {
            get { return cache_.Value; }
        }

        public double Weight
        {
            get { return this.weight_; }
            set
            {
                if (this.weight_ != value)
                {
                    if (IsChanging != null)
                    {
                        IsChanging(this, null);
                    }
                    this.weight_ = value;
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
