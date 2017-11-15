using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_networks
{
    public delegate void IsChangingHandler(object sender, EventArgs e);

    public delegate void HasChangedHandler(object sender, EventArgs e);

    public class CachedValue<T>
    {
        #region Data Members

        protected T cachedValue_;
        protected bool requiresUpdate_ = true;
        protected Func<T> valueUpdater_ = null;

        #endregion

        #region Constructors

        public CachedValue() { }

        public CachedValue(T initialValue)
        {
            cachedValue_ = initialValue;
            requiresUpdate_ = false;
        }

        public CachedValue(T initialValue, Func<T> valueUpdater)
        {
            this.cachedValue_ = initialValue;
            this.valueUpdater_ = valueUpdater;
        }

        #endregion

        #region Operations

        public event HasChangedHandler HasChanged;

        public event IsChangingHandler IsChanging;

        #endregion

        #region Fields
        
        public bool RequiresUpdate
        {
            get
            {
                return this.requiresUpdate_;
            }
            set
            {
                bool oldValue = this.requiresUpdate_;
                this.requiresUpdate_ = value;

                if (value)
                {
                    if (IsChanging != null)
                    {
                        IsChanging(this, null);
                    }
                }
                else
                {
                    if (oldValue)
                    {
                        if (HasChanged != null)
                        {
                            HasChanged(this, null);
                        }
                    }                   
                }
            }
        }

        protected void UpdateCache()
        {
            cachedValue_ = valueUpdater_();
            RequiresUpdate = false;
        }

        public Func<T> UpdateValue
        {
            get { return this.valueUpdater_; }
            set { this.valueUpdater_ = value; }
        }

        public T Value
        {
            get
            {
                if (RequiresUpdate)
                {
                    UpdateCache();
                }

                return cachedValue_;
            }
        }

        #endregion
    }
}
