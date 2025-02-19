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
            var sampleVec = SampleOutputVecs[Grid.IdxX];
            var outputVec = Network.FindOutputVec(InputVecs);
            var errorVec = sampleVec - outputVec;
            for (int i = 0; i < Network.ParameterCount; i++)
            {
                var parameter = Network.GetParameter(i);
                parameter.AdjustWeight(errorVec, Network.RMSPropBaseLearningRate, Network.RMSPropForgetRate);
            }

        }
    }
}
