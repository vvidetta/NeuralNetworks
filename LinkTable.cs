using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neural_networks
{
    public class LinkTable
    {

        #region Data Members

        private Dictionary<Tuple<IInput, IOutput>, Link> linkTable_ = new Dictionary<Tuple<IInput, IOutput>, Link>();
        private NodeTable nodeTable_;

        #endregion

        #region Constructors

        public LinkTable()
        {

        }

        public LinkTable(NodeTable nodeTable)
        {
            this.nodeTable_ = nodeTable;

            for (int i = 0; i < nodeTable.LayerSizes.Length - 1; i++)
            {
                IEnumerable<IEvaluable> inputLayer = nodeTable.GetLayer(i);
                IEnumerable<IEvaluable> outputLayer = nodeTable.GetLayer(i + 1);
                Link.AddLayer(linkTable_, inputLayer, outputLayer);
            }
        }

        #endregion

        #region Operations

        public Link FindLink(int layer, int inputIndex, int outputIndex)
        {
            IInput inRef = nodeTable_[layer, inputIndex] as IInput;
            IOutput outRef = nodeTable_[layer + 1, outputIndex] as IOutput;
            return this[inRef, outRef];
        }
        
        public void LoadLinks(List<Matrix> L)
        {
            for (int i = 0; i < L.Count; i++)
            {
                LoadLinks(i, L[i]);
            }
        }

        public void LoadLinks(int baseLayerIndex, Matrix W)
        {
            if (nodeTable_.LayerSizes[baseLayerIndex] != W.Height)
                throw new Exception("Matrix has wrong height!");

            if (nodeTable_.LayerSizes[baseLayerIndex + 1] != W.Width)
            {
                throw new Exception ("Matrix has wrong width");
            }

            for (int i = 0; i< W.Height; i++)
            {
                for (int j = 0; j < W.Width; j++)
                {
                    this[baseLayerIndex, i, j] = W[i, j];
                }
            }
        }

        #endregion

        #region Fields

        private Link this[IInput inRef, IOutput outRef]
        {
            get
            {
                return this.linkTable_[Tuple.Create(inRef, outRef)];
            }
        }

        public double this[int i, int j, int k]
        {
            get
            {
                Link link = FindLink(i, j, k);
                return link.Weight;
            }
            set
            {
                Link link = FindLink(i, j, k);
                link.Weight = value;
            }
        }

        #endregion

    }
}
