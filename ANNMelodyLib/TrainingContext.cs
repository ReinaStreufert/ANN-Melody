using ILGPU;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ANNMelodyLib
{
    public struct TrainingContext
    {
        public GenerativeMelodyNetwork Network;
        public ArrayView<float> InputVecs;
        public ArrayView<float> SampleOutputVecs;

        public void AdjustNodeWeights()
        {
            var paramsPerNode = Network.ParametersPerNode;
        }
    }
}
