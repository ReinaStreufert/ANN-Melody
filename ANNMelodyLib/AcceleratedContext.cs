using ILGPU;

namespace ANNMelodyLib
{
    public struct AcceleratedContext
    {
        public float RMSPropBaseLearningRate { get; }
        public float RMSPropBaseForgetRate { get; }
        public ArrayView<float> HiddenLayer { get; }
        public ArrayView<float> OutputLayer { get; }
        public bool IsOutputKernel { get; }
    }

    public ref struct RMSParameter
    {
        public ref float WeightVec;
        public ref float SlidingSparsityVec;

        public void AdjustWeightRMS(float deltaWeight, float baseLearnRate, float adaptRate)
        {
            SlidingSparsityVec = RMSProp.SlidingSparsityAdjust(SlidingSparsityVec, adaptRate, deltaWeight);
            var adjustedDelta = RMSProp.ParamDeltaRMSAdjust(baseLearnRate, deltaWeight, SlidingSparsityVec);
            WeightVec += adjustedDelta;
        }
    }
}