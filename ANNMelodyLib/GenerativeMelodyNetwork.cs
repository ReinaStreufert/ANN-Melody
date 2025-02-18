using ILGPU;

namespace ANNMelodyLib
{
    // this will be a 2-layer neural network. the network has 9 inputs for each 32nd-note in the context window.
    // each position in the context window must have either all 9 inputs at zero, or a singular one set to 1 with
    // with all the others being zero. this represents either silence, a specific scale note, or the 9th input=a continuation
    // of the last playing note. backpropogation for learning, RMSProp for convergence, and a logistic activation function

    public struct GenerativeMelodyNetwork
    {
        public float RMSPropBaseLearningRate { get; }
        public float RMSPropBaseForgetRate { get; }
        public ArrayView<float> WeightVecs { get; }
        public ArrayView<float> SparsityVecs { get; }
    }

    public ref struct RMSParameter
    {
        public ref float WeightVec;
        public ref float SlidingSparsityVec;

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