using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ANNMelodyLib
{
    public static class ANN
    {
        public static float ActivateLogistic(float weightedSum)
        {
            return 1F / (1 + MathF.Pow(MathF.E, -weightedSum));
        }
    }
}
