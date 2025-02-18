﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ANNMelodyLib
{
    public static class RMSProp
    {
        public static float SlidingSparsityAdjust(float slidingSparsity, float adaptRate, float deltaWeight)
        {
            return (1 - adaptRate) * slidingSparsity +
                adaptRate * MathF.Pow(deltaWeight, 2);
        }

        public static float ParamDeltaRMSAdjust(float baseLearnRate, float deltaWeight, float slidingSparsity)
        {
            return (baseLearnRate / MathF.Sqrt(slidingSparsity)) * deltaWeight;
        }
    }
}
