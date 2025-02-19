using ILGPU;

namespace ANNMelodyLib
{
    // this will be a 1-layer neural network. the network has 9 inputs for each 32nd-note in the context window.
    // each position in the context window must have either all 9 inputs at zero, or a singular one set to 1 with
    // with all the others being zero. this represents either silence, a specific scale note, or the 9th input=a continuation
    // of the last playing note. the network has 9 outputs, each one is a probability value predicting the likelihood of the
    // next position in the context window being the corresponding note. backpropogation for learning, RMSProp for convergence, 
    // and a logistic activation function

    public struct GenerativeMelodyNetwork
    {
        public float RMSPropBaseLearningRate { get; }
        public float RMSPropForgetRate { get; }
        public ArrayView<float> WeightVecs { get; }
        public ArrayView<float> SparsityVecs { get; }
        public int ContextWindowLen { get; }
        public int ParameterCount => ContextWindowLen * 9;
        public int OutputCount => 9;

        public RMSParameter GetParameter(int index)
        {
            var vecIdx = Grid.IdxX * ParameterCount + index;
            return new RMSParameter(ref WeightVecs[vecIdx], ref SparsityVecs[vecIdx]);
        }

        public float FindOutputVec(ArrayView<float> inputVecs)
        {
            var weightedSum = 0f;
            for (int i = 0; i < ParameterCount; i++)
            {
                var parameter = GetParameter(i);
                weightedSum += inputVecs[i] * parameter.WeightVec;
            }
            return ANN.ActivateLogistic(weightedSum);
        }
    }

    public ref struct RMSParameter
    {
        public ref float WeightVec;
        public ref float SlidingSparsityVec;

        public RMSParameter(ref float weightVec, ref float slidingSparsityVec)
        {
            WeightVec = weightVec;
            SlidingSparsityVec = slidingSparsityVec;
        }

        public void AdjustWeight(float outputError, float baseLearnRate, float adaptRate)
        {
            var correctionVec = outputError * WeightVec;
            ApplyDeltaRMS(correctionVec, baseLearnRate, adaptRate);
            // backpropogation feels like magic, like why does it work even?
        }

        private void ApplyDeltaRMS(float correctionVal, float baseLearnRate, float adaptRate)
        {
            SlidingSparsityVec = RMSProp.SlidingSparsityAdjust(SlidingSparsityVec, adaptRate, correctionVal);
            var adjustedDelta = RMSProp.ParamDeltaRMSAdjust(WeightVec, correctionVal, baseLearnRate, SlidingSparsityVec);
            WeightVec += adjustedDelta;
        }
    }
}