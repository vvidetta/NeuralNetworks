using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_networks
{
    public class NodeTable
    {
        #region Data Members

        private Dictionary<Tuple<int, int>, Node> nodeTable_ = new Dictionary<Tuple<int,int>,Node>();
        private int[] layerSizes_;
        private List<InputNode> inputLayer;
        private List<OutputNode> outputLayer;

        #endregion

        #region Constructors

        public NodeTable()
        {

        }

        public NodeTable(Func<double, double> nonLinearity, params int[] layerSizes)
        {
            if (layerSizes.Length < 2)
                throw new Exception("Need at least 2 layers!!!!!");

            this.layerSizes_ = layerSizes;

            for (int i = 0; i < layerSizes.Length; i++)
            {
                if (i == 0)
                {
                    inputLayer = AddLayer<InputNode>(i, layerSizes[i]);
                }
                else
                {
                    IEnumerable<IEvaluable> nodeLayer;
                    if (i == layerSizes.Length - 1)
                    {
                        nodeLayer = outputLayer = AddLayer<OutputNode>(i, layerSizes[i]);
                    }
                    else
                    {
                        nodeLayer = AddLayer<HiddenNode>(i, layerSizes[i]);
                    }
                    foreach (var node in nodeLayer)
                    {
                        var nonLinear = node as INonLinear;
                        nonLinear.NonLinearity = nonLinearity;
                    }
                }
            }
        }

        #endregion

        #region Operations

        private List<T> AddLayer<T>(int layerIndex, int layerSize) where T : Node, new()
        {
            var nodeLayer = new List<T>();
            T nodeRef;
            for (int i = 0; i < layerSize; i++)
            {
                nodeRef = new T();
                nodeRef.Name = String.Format("{0},{1}", layerIndex, i);
                nodeLayer.Add(nodeRef);
                nodeTable_.Add(Tuple.Create(layerIndex, i), nodeRef);
            }
            return nodeLayer;
        }

        public List<Node> GetLayer(int layerIndex)
        {
            var keyedList = new List<KeyValuePair<Tuple<int, int>, Node>>();
            foreach (var pair in nodeTable_)
            {
                if (pair.Key.Item1 == layerIndex)
                {
                    keyedList.Add(pair);
                }
            }

            Comparison<KeyValuePair<Tuple<int, int>, Node>> comparison = (x, y) => 
            {
                int i = x.Key.Item2;
                int j = y.Key.Item2;
                return (i < j) ? -1 : ( (i == j ) ? 0 : 1);
            };

            keyedList.Sort(comparison);

            Func<KeyValuePair<Tuple<int, int>, Node>, Node> selector = (x) => { return x.Value; };

            return keyedList.Select(selector).ToList();
        }

        #endregion

        #region Fields

        public List<InputNode> Inputs
        {
            get
            {
                return this.inputLayer;
            }
        }

        public int[] LayerSizes
        {
            get
            {
                return this.layerSizes_;
            }
        }

        public List<OutputNode> Outputs
        {
            get 
            {
                return this.outputLayer;
            }
        }

        public IEvaluable this[int i, int j]
        {
            get
            {
                return this.nodeTable_[new Tuple<int, int>(i, j)];
            }
        }

        #endregion
    }
}
