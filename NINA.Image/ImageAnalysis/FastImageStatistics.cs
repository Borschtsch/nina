#region "copyright"

/*
    Copyright © 2016 - 2024 Stefan Berg <isbeorn86+NINA@googlemail.com> and the N.I.N.A. contributors

    This file is part of N.I.N.A. - Nighttime Imaging 'N' Astronomy.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#endregion "copyright"

using NINA.Core.Utility;
using NINA.Image.Interfaces;
using OxyPlot;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;

namespace NINA.Image.ImageData {

        /// <summary>
        /// Optimized statistics calculator to avoid per-call allocations and heavy LINQ.
        /// Returns the same values as ImageStatistics.Create.
        /// </summary>
        internal static class FastImageStatistics {
            // Bit-precise path: same math as ImageStatistics, but uses pooled buffers and unrolled pointer loops.
        public static IImageStatistics Create(IImageData imageData) =>
            Create(imageData.Properties, imageData.Data.FlatArray);

        public static (IImageStatistics Red, IImageStatistics Green, IImageStatistics Blue) CreateRgb(ImageProperties imageProperties, LRGBArrays arrays) {
            using (MyStopWatch.Measure()) {
            if (arrays == null || arrays.Red == null || arrays.Green == null || arrays.Blue == null) {
                throw new ArgumentException("RGB arrays are required for CreateRgb", nameof(arrays));
            }

            int length = arrays.Red.Length;
            int maxValue = (1 << imageProperties.BitDepth) - 1;
            int histLen = maxValue + 1;

            long sumR = 0, sumG = 0, sumB = 0;
            long squareSumR = 0, squareSumG = 0, squareSumB = 0;
            ushort minR = ushort.MaxValue, minG = ushort.MaxValue, minB = ushort.MaxValue;
            ushort maxR = 0, maxG = 0, maxB = 0;
            long minOccR = 0, minOccG = 0, minOccB = 0;
            long maxOccR = 0, maxOccG = 0, maxOccB = 0;

            int proc = Math.Max(1, Environment.ProcessorCount);
            int chunk = (length + proc - 1) / proc;

            int[][] threadHistR = new int[proc][];
            int[][] threadHistG = new int[proc][];
            int[][] threadHistB = new int[proc][];

            long[] sumRArr = new long[proc];
            long[] sumGArr = new long[proc];
            long[] sumBArr = new long[proc];
            long[] sqR = new long[proc];
            long[] sqG = new long[proc];
            long[] sqB = new long[proc];
            ushort[] minRArr = new ushort[proc];
            ushort[] minGArr = new ushort[proc];
            ushort[] minBArr = new ushort[proc];
            ushort[] maxRArr = new ushort[proc];
            ushort[] maxGArr = new ushort[proc];
            ushort[] maxBArr = new ushort[proc];
            long[] minOccRArr = new long[proc];
            long[] minOccGArr = new long[proc];
            long[] minOccBArr = new long[proc];
            long[] maxOccRArr = new long[proc];
            long[] maxOccGArr = new long[proc];
            long[] maxOccBArr = new long[proc];

            for (int t = 0; t < proc; t++) {
                threadHistR[t] = ArrayPool<int>.Shared.Rent(histLen);
                threadHistG[t] = ArrayPool<int>.Shared.Rent(histLen);
                threadHistB[t] = ArrayPool<int>.Shared.Rent(histLen);
                Array.Clear(threadHistR[t], 0, histLen);
                Array.Clear(threadHistG[t], 0, histLen);
                Array.Clear(threadHistB[t], 0, histLen);
                minRArr[t] = minGArr[t] = minBArr[t] = ushort.MaxValue;
            }

            // Process RGB channels in parallel; each worker writes to its own histograms.
            Parallel.For(0, proc, t => {
                int start = t * chunk;
                if (start >= length) return;
                int end = Math.Min(length, start + chunk);

                var histR = threadHistR[t];
                var histG = threadHistG[t];
                var histB = threadHistB[t];

                long sR = 0, sG = 0, sB = 0;
                long ssR = 0, ssG = 0, ssB = 0;
                ushort lminR = ushort.MaxValue, lminG = ushort.MaxValue, lminB = ushort.MaxValue;
                ushort lmaxR = 0, lmaxG = 0, lmaxB = 0;
                long lminOccR = 0, lminOccG = 0, lminOccB = 0;
                long lmaxOccR = 0, lmaxOccG = 0, lmaxOccB = 0;

                int count = end - start;

                unsafe {
                    // Pin channel arrays for pointer-based iteration.
                    fixed (ushort* rPtr = arrays.Red, gPtr = arrays.Green, bPtr = arrays.Blue)
                    fixed (int* hR = histR, hG = histG, hB = histB) {
                        ushort* rBase = rPtr + start;
                        ushort* gBase = gPtr + start;
                        ushort* bBase = bPtr + start;

                        int i = 0;
                        int unroll = count & ~3; // process 4 pixels per iteration (bit-identical to scalar loop)
                        for (; i < unroll; i += 4) {
                            ushort r0 = rBase[i + 0]; ushort g0 = gBase[i + 0]; ushort b0 = bBase[i + 0];
                            ushort r1 = rBase[i + 1]; ushort g1 = gBase[i + 1]; ushort b1 = bBase[i + 1];
                            ushort r2 = rBase[i + 2]; ushort g2 = gBase[i + 2]; ushort b2 = bBase[i + 2];
                            ushort r3 = rBase[i + 3]; ushort g3 = gBase[i + 3]; ushort b3 = bBase[i + 3];

                            sR += r0 + r1 + r2 + r3;
                            sG += g0 + g1 + g2 + g3;
                            sB += b0 + b1 + b2 + b3;

                            ssR += (long)r0 * r0 + (long)r1 * r1 + (long)r2 * r2 + (long)r3 * r3;
                            ssG += (long)g0 * g0 + (long)g1 * g1 + (long)g2 * g2 + (long)g3 * g3;
                            ssB += (long)b0 * b0 + (long)b1 * b1 + (long)b2 * b2 + (long)b3 * b3;

                            hR[r0]++; hR[r1]++; hR[r2]++; hR[r3]++;
                            hG[g0]++; hG[g1]++; hG[g2]++; hG[g3]++;
                            hB[b0]++; hB[b1]++; hB[b2]++; hB[b3]++;

                            if (r0 < lminR) { lminR = r0; lminOccR = 1; } else if (r0 == lminR) { lminOccR++; }
                            if (g0 < lminG) { lminG = g0; lminOccG = 1; } else if (g0 == lminG) { lminOccG++; }
                            if (b0 < lminB) { lminB = b0; lminOccB = 1; } else if (b0 == lminB) { lminOccB++; }

                            if (r0 > lmaxR) { lmaxR = r0; lmaxOccR = 1; } else if (r0 == lmaxR) { lmaxOccR++; }
                            if (g0 > lmaxG) { lmaxG = g0; lmaxOccG = 1; } else if (g0 == lmaxG) { lmaxOccG++; }
                            if (b0 > lmaxB) { lmaxB = b0; lmaxOccB = 1; } else if (b0 == lmaxB) { lmaxOccB++; }

                            if (r1 < lminR) { lminR = r1; lminOccR = 1; } else if (r1 == lminR) { lminOccR++; }
                            if (g1 < lminG) { lminG = g1; lminOccG = 1; } else if (g1 == lminG) { lminOccG++; }
                            if (b1 < lminB) { lminB = b1; lminOccB = 1; } else if (b1 == lminB) { lminOccB++; }

                            if (r1 > lmaxR) { lmaxR = r1; lmaxOccR = 1; } else if (r1 == lmaxR) { lmaxOccR++; }
                            if (g1 > lmaxG) { lmaxG = g1; lmaxOccG = 1; } else if (g1 == lmaxG) { lmaxOccG++; }
                            if (b1 > lmaxB) { lmaxB = b1; lmaxOccB = 1; } else if (b1 == lmaxB) { lmaxOccB++; }

                            if (r2 < lminR) { lminR = r2; lminOccR = 1; } else if (r2 == lminR) { lminOccR++; }
                            if (g2 < lminG) { lminG = g2; lminOccG = 1; } else if (g2 == lminG) { lminOccG++; }
                            if (b2 < lminB) { lminB = b2; lminOccB = 1; } else if (b2 == lminB) { lminOccB++; }

                            if (r2 > lmaxR) { lmaxR = r2; lmaxOccR = 1; } else if (r2 == lmaxR) { lmaxOccR++; }
                            if (g2 > lmaxG) { lmaxG = g2; lmaxOccG = 1; } else if (g2 == lmaxG) { lmaxOccG++; }
                            if (b2 > lmaxB) { lmaxB = b2; lmaxOccB = 1; } else if (b2 == lmaxB) { lmaxOccB++; }

                            if (r3 < lminR) { lminR = r3; lminOccR = 1; } else if (r3 == lminR) { lminOccR++; }
                            if (g3 < lminG) { lminG = g3; lminOccG = 1; } else if (g3 == lminG) { lminOccG++; }
                            if (b3 < lminB) { lminB = b3; lminOccB = 1; } else if (b3 == lminB) { lminOccB++; }

                            if (r3 > lmaxR) { lmaxR = r3; lmaxOccR = 1; } else if (r3 == lmaxR) { lmaxOccR++; }
                            if (g3 > lmaxG) { lmaxG = g3; lmaxOccG = 1; } else if (g3 == lmaxG) { lmaxOccG++; }
                            if (b3 > lmaxB) { lmaxB = b3; lmaxOccB = 1; } else if (b3 == lmaxB) { lmaxOccB++; }
                        }

                        for (; i < count; i++) {
                            ushort r = rBase[i];
                            ushort g = gBase[i];
                            ushort b = bBase[i];

                            sR += r; ssR += (long)r * r; histR[r]++;
                            sG += g; ssG += (long)g * g; histG[g]++;
                            sB += b; ssB += (long)b * b; histB[b]++;

                            if (r < lminR) { lminR = r; lminOccR = 1; } else if (r == lminR) { lminOccR++; }
                            if (g < lminG) { lminG = g; lminOccG = 1; } else if (g == lminG) { lminOccG++; }
                            if (b < lminB) { lminB = b; lminOccB = 1; } else if (b == lminB) { lminOccB++; }

                            if (r > lmaxR) { lmaxR = r; lmaxOccR = 1; } else if (r == lmaxR) { lmaxOccR++; }
                            if (g > lmaxG) { lmaxG = g; lmaxOccG = 1; } else if (g == lmaxG) { lmaxOccG++; }
                            if (b > lmaxB) { lmaxB = b; lmaxOccB = 1; } else if (b == lmaxB) { lmaxOccB++; }
                        }
                    }
                }

                sumRArr[t] = sR; sumGArr[t] = sG; sumBArr[t] = sB;
                sqR[t] = ssR; sqG[t] = ssG; sqB[t] = ssB;
                minRArr[t] = lminR; minOccRArr[t] = lminOccR;
                minGArr[t] = lminG; minOccGArr[t] = lminOccG;
                minBArr[t] = lminB; minOccBArr[t] = lminOccB;
                maxRArr[t] = lmaxR; maxOccRArr[t] = lmaxOccR;
                maxGArr[t] = lmaxG; maxOccGArr[t] = lmaxOccG;
                maxBArr[t] = lmaxB; maxOccBArr[t] = lmaxOccB;
            });

            int[] histR = ArrayPool<int>.Shared.Rent(histLen);
            int[] histG = ArrayPool<int>.Shared.Rent(histLen);
            int[] histB = ArrayPool<int>.Shared.Rent(histLen);
            Array.Clear(histR, 0, histLen);
            Array.Clear(histG, 0, histLen);
            Array.Clear(histB, 0, histLen);

            for (int t = 0; t < proc; t++) {
                sumR += sumRArr[t]; sumG += sumGArr[t]; sumB += sumBArr[t];
                squareSumR += sqR[t]; squareSumG += sqG[t]; squareSumB += sqB[t];

                MergeMinMax(ref minR, ref minOccR, minRArr[t], minOccRArr[t]);
                MergeMinMax(ref minG, ref minOccG, minGArr[t], minOccGArr[t]);
                MergeMinMax(ref minB, ref minOccB, minBArr[t], minOccBArr[t]);

                MergeMaxMax(ref maxR, ref maxOccR, maxRArr[t], maxOccRArr[t]);
                MergeMaxMax(ref maxG, ref maxOccG, maxGArr[t], maxOccGArr[t]);
                MergeMaxMax(ref maxB, ref maxOccB, maxBArr[t], maxOccBArr[t]);
            }

            Parallel.For(0, histLen, i => {
                int r = 0, g = 0, b = 0;
                for (int t = 0; t < proc; t++) {
                    r += threadHistR[t][i];
                    g += threadHistG[t][i];
                    b += threadHistB[t][i];
                }
                histR[i] = r;
                histG[i] = g;
                histB[i] = b;
            });

            // Return rented histograms to the pool after merge; all math is preserved.
            for (int t = 0; t < proc; t++) {
                ArrayPool<int>.Shared.Return(threadHistR[t]);
                ArrayPool<int>.Shared.Return(threadHistG[t]);
                ArrayPool<int>.Shared.Return(threadHistB[t]);
            }

            try {
                var statsR = BuildStatistics(imageProperties, length, sumR, squareSumR, minR, minOccR, maxR, maxOccR, histR, maxValue);
                var statsG = BuildStatistics(imageProperties, length, sumG, squareSumG, minG, minOccG, maxG, maxOccG, histG, maxValue);
                var statsB = BuildStatistics(imageProperties, length, sumB, squareSumB, minB, minOccB, maxB, maxOccB, histB, maxValue);
                return (statsR, statsG, statsB);
            } finally {
                ArrayPool<int>.Shared.Return(histR);
                ArrayPool<int>.Shared.Return(histG);
                ArrayPool<int>.Shared.Return(histB);
            }
            }
        }

        public static IImageStatistics Create(ImageProperties imageProperties, ushort[] array) {
            using (MyStopWatch.Measure()) {
                int length = array.Length;
                int maxValue = (1 << imageProperties.BitDepth) - 1;
                int histLen = maxValue + 1;
                long sum = 0;
                long squareSum = 0;
                ushort min = ushort.MaxValue;
                ushort max = 0;
                long minOccurrences = 0;
                long maxOccurrences = 0;

                int[] pixelValueCounts = ArrayPool<int>.Shared.Rent(histLen);
                Array.Clear(pixelValueCounts, 0, histLen);

                try {
                    for (int i = 0; i < length; i++) {
                        ushort val = array[i];
                        sum += val;
                        squareSum += (long)val * val;

                        _ = ++pixelValueCounts[val];

                        if (val < min) {
                            min = val;
                            minOccurrences = 1;
                        } else if (val == min) {
                            minOccurrences++;
                        }

                        if (val > max) {
                            max = val;
                            maxOccurrences = 1;
                        } else if (val == max) {
                            maxOccurrences++;
                        }
                    }

                    return BuildStatistics(imageProperties, length, sum, squareSum, min, minOccurrences, max, maxOccurrences, pixelValueCounts, maxValue);
                } finally {
                    ArrayPool<int>.Shared.Return(pixelValueCounts);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ProcessPixel(
            ushort v,
            ref long sum,
            ref long squareSum,
            ref ushort min,
            ref long minOcc,
            ref ushort max,
            ref long maxOcc,
            int[] hist) {

            sum += v;
            squareSum += (long)v * v;
            _ = ++hist[v];

            if (v < min) { min = v; minOcc = 1; }
            else if (v == min) { minOcc++; }

            if (v > max) { max = v; maxOcc = 1; }
            else if (v == max) { maxOcc++; }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void MergeMinMax(ref ushort globalMin, ref long globalOcc, ushort localMin, long localOcc) {
            if (localOcc == 0) return;
            if (localMin < globalMin) {
                globalMin = localMin;
                globalOcc = localOcc;
            } else if (localMin == globalMin) {
                globalOcc += localOcc;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void MergeMaxMax(ref ushort globalMax, ref long globalOcc, ushort localMax, long localOcc) {
            if (localOcc == 0) return;
            if (localMax > globalMax) {
                globalMax = localMax;
                globalOcc = localOcc;
            } else if (localMax == globalMax) {
                globalOcc += localOcc;
            }
        }

        private struct LocalRgbStats {
            public long SumR, SumG, SumB;
            public long SquareSumR, SquareSumG, SquareSumB;
            public ushort MinR, MinG, MinB;
            public ushort MaxR, MaxG, MaxB;
            public long MinOccR, MinOccG, MinOccB;
            public long MaxOccR, MaxOccG, MaxOccB;
            public int[] HistR, HistG, HistB;

            public LocalRgbStats(int histLen) {
                SumR = SumG = SumB = 0;
                SquareSumR = SquareSumG = SquareSumB = 0;
                MinR = MinG = MinB = ushort.MaxValue;
                MaxR = MaxG = MaxB = 0;
                MinOccR = MinOccG = MinOccB = 0;
                MaxOccR = MaxOccG = MaxOccB = 0;
                HistR = ArrayPool<int>.Shared.Rent(histLen);
                HistG = ArrayPool<int>.Shared.Rent(histLen);
                HistB = ArrayPool<int>.Shared.Rent(histLen);
                Array.Clear(HistR, 0, histLen);
                Array.Clear(HistG, 0, histLen);
                Array.Clear(HistB, 0, histLen);
            }

            public void Dispose() {
                ArrayPool<int>.Shared.Return(HistR);
                ArrayPool<int>.Shared.Return(HistG);
                ArrayPool<int>.Shared.Return(HistB);
            }
        }

        private static ImageStatistics BuildStatistics(
            ImageProperties imageProperties,
            int length,
            long sum,
            long squareSum,
            int min,
            long minOccurrences,
            int max,
            long maxOccurrences,
            int[] pixelValueCounts,
            int maxPossibleValueOverride = ushort.MaxValue) {

            double mean = sum / (double)length;
            double variance = (squareSum - length * mean * mean) / length;
            double stdev = Math.Sqrt(variance);

            double median = 0d;
            int median1 = 0, median2 = 0;
            int occurrences = 0;
            double medianlength = length / 2.0;
            for (int i = 0; i <= maxPossibleValueOverride; i++) {
                occurrences += pixelValueCounts[i];
                if (occurrences > medianlength) {
                    median1 = median2 = i;
                    break;
                } else if (occurrences == medianlength) {
                    median1 = i;
                    for (int j = i + 1; j <= maxPossibleValueOverride; j++) {
                        if (pixelValueCounts[j] > 0) {
                            median2 = j;
                            break;
                        }
                    }
                    break;
                }
            }
            median = (median1 + median2) / 2.0;

            double medianAbsoluteDeviation = 0.0d;
            occurrences = 0;
            int idxDown = median1;
            int idxUp = median2;
            while (true) {
                if (idxDown >= 0 && idxDown != idxUp) {
                    occurrences += pixelValueCounts[idxDown] + pixelValueCounts[idxUp];
                } else {
                    occurrences += pixelValueCounts[idxUp];
                }

                if (occurrences > medianlength) {
                    medianAbsoluteDeviation = Math.Abs(idxUp - median);
                    break;
                }

                idxUp++;
                idxDown--;
                if (idxUp > maxPossibleValueOverride) {
                    break;
                }
            }

            int maxPossibleValue = maxPossibleValueOverride;
            double factor = (double)ImageStatistics.HISTOGRAMRESOLUTION / maxPossibleValue;
            int bucketCount = ImageStatistics.HISTOGRAMRESOLUTION + 1;
            int[] buckets = new int[bucketCount];
            for (int i = 0; i <= maxPossibleValue; i++) {
                int bucket = (int)Math.Floor(Math.Min(maxPossibleValue, i) * factor);
                buckets[bucket] += pixelValueCounts[i];
            }

            var points = new List<DataPoint>(bucketCount);
            for (int i = 0; i < bucketCount; i++) {
                points.Add(new DataPoint(i, buckets[i]));
            }

            var statistics = new ImageStatistics();
            statistics.BitDepth = imageProperties.BitDepth;
            statistics.StDev = stdev;
            statistics.Mean = mean;
            statistics.Median = median;
            statistics.MedianAbsoluteDeviation = medianAbsoluteDeviation;
            statistics.Max = max;
            statistics.MaxOccurrences = maxOccurrences;
            statistics.Min = min;
            statistics.MinOccurrences = minOccurrences;
            statistics.Histogram = points.ToImmutableList();
            return statistics;
        }
    }
}
