#region "copyright"

/*
    Copyright c 2016 - 2024 Stefan Berg <isbeorn86+NINA@googlemail.com> and the N.I.N.A. contributors

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

namespace NINA.Image.ImageData {

    public class ImageStatistics : BaseINPC, IImageStatistics {
        public static ImageStatistics EmptyImageStatistics = new ImageStatistics();

        public const int HISTOGRAMRESOLUTION = 100;

        public ImageStatistics() {
        }

        public int BitDepth { get; set; }
        public double StDev { get; set; }
        public double Mean { get; set; }
        public double Median { get; set; }
        public double MedianAbsoluteDeviation { get; set; }
        public int Max { get; set; }
        public long MaxOccurrences { get; set; }
        public int Min { get; set; }
        public long MinOccurrences { get; set; }
        public ImmutableList<DataPoint> Histogram { get; set; }

        public static IImageStatistics Create(IImageData imageData) {
            return Create(imageData.Properties, imageData.Data.FlatArray);
        }

        public static IImageStatistics Create(ImageProperties imageProperties, ushort[] array) {
            using (MyStopWatch.Measure()) {
                int length = array.Length;
                long sum = 0;
                long squareSum = 0;
                ushort min = ushort.MaxValue;
                ushort max = 0;
                long minOccurrences = 0;
                long maxOccurrences = 0;

                int[] pixelValueCounts = ArrayPool<int>.Shared.Rent(ushort.MaxValue + 1);
                Array.Clear(pixelValueCounts, 0, ushort.MaxValue + 1);

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

                    double mean = sum / (double)length;
                    double variance = (squareSum - length * mean * mean) / length;
                    double stdev = Math.Sqrt(variance);

                    double median = 0d;
                    int median1 = 0, median2 = 0;
                    int occurrences = 0;
                    double medianlength = length / 2.0;
                    for (int i = 0; i <= ushort.MaxValue; i++) {
                        occurrences += pixelValueCounts[i];
                        if (occurrences > medianlength) {
                            median1 = median2 = i;
                            break;
                        } else if (occurrences == medianlength) {
                            median1 = i;
                            for (int j = i + 1; j <= ushort.MaxValue; j++) {
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
                        if (idxUp > ushort.MaxValue) {
                            break;
                        }
                    }

                    ushort maxPossibleValue = (ushort)((1 << imageProperties.BitDepth) - 1);
                    double factor = (double)HISTOGRAMRESOLUTION / maxPossibleValue;
                    int bucketCount = HISTOGRAMRESOLUTION + 1;
                    int[] buckets = new int[bucketCount];
                    for (int i = 0; i <= maxPossibleValue; i++) {
                        int bucket = (int)Math.Floor(Math.Min(maxPossibleValue, i) * factor);
                        buckets[bucket] += pixelValueCounts[i];
                    }

                    var points = new List<DataPoint>(bucketCount);
                    for (int i = 0; i < bucketCount; i++) {
                        points.Add(new DataPoint(i, buckets[i]));
                    }

                    var statistics = new ImageStatistics {
                        BitDepth = imageProperties.BitDepth,
                        StDev = stdev,
                        Mean = mean,
                        Median = median,
                        MedianAbsoluteDeviation = medianAbsoluteDeviation,
                        Max = max,
                        MaxOccurrences = maxOccurrences,
                        Min = min,
                        MinOccurrences = minOccurrences,
                        Histogram = points.ToImmutableList()
                    };
                    return statistics;
                } finally {
                    ArrayPool<int>.Shared.Return(pixelValueCounts);
                }
            }
        }
    }
}
