#region "copyright"

/*
    Copyright © 2016 - 2024 Stefan Berg <isbeorn86+NINA@googlemail.com> and the N.I.N.A. contributors

    This file is part of N.I.N.A. - Nighttime Imaging 'N' Astronomy.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#endregion "copyright"

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Threading;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using Accord.Imaging;
using Accord.Imaging.Filters;

namespace NINA.Image.ImageAnalysis {

    /// <summary>
    /// Optimized in-place Canny edge detector with built-in 3x3 separable Gaussian blur.
    /// Mirrors the behavior of Accord's CannyEdgeDetector but reduces allocations
    /// and expensive math on the hot path.
    /// </summary>
    public class CannyEdgeDetector : BaseUsingCopyPartialFilter {
        private byte lowThreshold = 20;
        private byte highThreshold = 100;

        // format translation dictionary
        private Dictionary<PixelFormat, PixelFormat> formatTranslations = new Dictionary<PixelFormat, PixelFormat>();

        public override Dictionary<PixelFormat, PixelFormat> FormatTranslations => formatTranslations;

        public byte LowThreshold {
            get => lowThreshold;
            set => lowThreshold = value;
        }

        public byte HighThreshold {
            get => highThreshold;
            set => highThreshold = value;
        }

        // Kept for compatibility; fixed 3x3 kernel.
        public int GaussianSize {
            get => 3;
            set { /* ignore to preserve fixed kernel */ }
        }

        public CannyEdgeDetector() {
            formatTranslations[PixelFormat.Format8bppIndexed] = PixelFormat.Format8bppIndexed;
        }

        public CannyEdgeDetector(byte lowThreshold, byte highThreshold) : this() {
            this.lowThreshold = lowThreshold;
            this.highThreshold = highThreshold;
        }

        protected override unsafe void ProcessFilter(UnmanagedImage source, UnmanagedImage destination, Rectangle rect) {
            // processing start and stop X,Y positions
            int startX = rect.Left + 1;
            int startY = rect.Top + 1;
            int stopX = startX + rect.Width - 2;
            int stopY = startY + rect.Height - 2;

            int dstStride = destination.Stride;
            int srcStride = source.Stride;
            int dstOffset = dstStride - rect.Width + 2;

            int widthFull = source.Width;
            int heightFull = source.Height;
            int totalPixels = widthFull * heightFull;

            byte[] srcBuffer = ArrayPool<byte>.Shared.Rent(totalPixels);
            int[] tempBlur = ArrayPool<int>.Shared.Rent(totalPixels);
            byte[] blurBuffer = ArrayPool<byte>.Shared.Rent(totalPixels);
            byte[] orientBuffer = ArrayPool<byte>.Shared.Rent(totalPixels);
            int[] magBuffer = ArrayPool<int>.Shared.Rent(totalPixels);

            int maxGradient = 1;

            try {
                // copy source into contiguous buffer (ignore stride padding)
                byte* srcBase = (byte*)source.ImageData.ToPointer();
                for (int y = 0; y < heightFull; y++) {
                    Marshal.Copy(new IntPtr(srcBase + y * srcStride), srcBuffer, y * widthFull, widthFull);
                }

                // STEP 0: 3x3 Gaussian blur (separable: horizontal then vertical)
                for (int y = 0; y < heightFull; y++) {
                    int row = y * widthFull;
                    int left = srcBuffer[row];
                    int center = srcBuffer[row];
                    int right = widthFull > 1 ? srcBuffer[row + 1] : center;
                    tempBlur[row] = left + (center << 1) + right;
                    for (int x = 1; x < widthFull - 1; x++) {
                        left = srcBuffer[row + x - 1];
                        center = srcBuffer[row + x];
                        right = srcBuffer[row + x + 1];
                        tempBlur[row + x] = left + (center << 1) + right;
                    }
                    if (widthFull > 1) {
                        left = srcBuffer[row + widthFull - 2];
                        center = srcBuffer[row + widthFull - 1];
                        right = center;
                        tempBlur[row + widthFull - 1] = left + (center << 1) + right;
                    }
                }

                for (int x = 0; x < widthFull; x++) {
                    int top = tempBlur[x];
                    int center = tempBlur[x];
                    int bottom = heightFull > 1 ? tempBlur[widthFull + x] : center;
                    blurBuffer[x] = (byte)((top + (center << 1) + bottom + 8) >> 4);
                    for (int y = 1; y < heightFull - 1; y++) {
                        int idx = y * widthFull + x;
                        top = tempBlur[idx - widthFull];
                        center = tempBlur[idx];
                        bottom = tempBlur[idx + widthFull];
                        blurBuffer[idx] = (byte)((top + (center << 1) + bottom + 8) >> 4);
                    }
                    if (heightFull > 1) {
                        int idx = (heightFull - 1) * widthFull + x;
                        top = tempBlur[idx - widthFull];
                        center = tempBlur[idx];
                        bottom = center;
                        blurBuffer[idx] = (byte)((top + (center << 1) + bottom + 8) >> 4);
                    }
                }

                // STEP 1 - calculate magnitude (L1) and quantized orientation (parallel per row)
                Parallel.For(startY, stopY, () => 0, (y, state, localMax) => {
                    int rowBase = y * widthFull;
                    for (int x = startX; x < stopX; x++) {
                        int idx = rowBase + x;

                        int a = blurBuffer[idx - widthFull - 1];
                        int b = blurBuffer[idx - widthFull];
                        int c = blurBuffer[idx - widthFull + 1];
                        int d = blurBuffer[idx - 1];
                        int f = blurBuffer[idx + 1];
                        int g = blurBuffer[idx + widthFull - 1];
                        int h = blurBuffer[idx + widthFull];
                        int i = blurBuffer[idx + widthFull + 1];

                        int gx = (c + (f << 1) + i) - (a + (d << 1) + g);
                        int gy = (g + (h << 1) + i) - (a + (b << 1) + c);

                        int ax = Math.Abs(gx);
                        int ay = Math.Abs(gy);
                        int m = ax + ay;
                        magBuffer[idx] = m;
                        if (m > localMax) {
                            localMax = m;
                        }

                        // Fast orientation quantization: 0,45,90,135
                        const int slope = 414; // ~tan(22.5deg) * 1000
                        byte orientation;
                        if (ay * 1000 <= slope * ax) {
                            orientation = 0; // horizontal
                        } else if (ax * 1000 <= slope * ay) {
                            orientation = 90; // vertical
                        } else {
                            orientation = (byte)((gx ^ gy) >= 0 ? 45 : 135);
                        }
                        orientBuffer[idx] = orientation;
                    }

                    return localMax;
                }, localMax => {
                    int initial;
                    while (true) {
                        initial = maxGradient;
                        if (localMax <= initial)
                            break;
                        if (Interlocked.CompareExchange(ref maxGradient, localMax, initial) == initial)
                            break;
                    }
                });

                if (maxGradient <= 0)
                    maxGradient = 1;

                // STEP 2 - suppress non maximums (parallel per row)
                byte* dstBase = (byte*)destination.ImageData.ToPointer();
                byte* dst = dstBase + dstStride * startY + startX;

                Parallel.For(startY, stopY, y => {
                    byte* rowDst = dst + (y - startY) * dstStride;
                    int row = y * widthFull;

                    for (int x = startX; x < stopX; x++, rowDst++) {
                        int idx = row + x;
                        int m = magBuffer[idx];
                        if (m == 0) {
                            *rowDst = 0;
                            continue;
                        }

                        int m1, m2;
                        switch (orientBuffer[idx]) {
                            case 0:
                                m1 = magBuffer[idx - 1];
                                m2 = magBuffer[idx + 1];
                                break;
                            case 45:
                                m1 = magBuffer[idx - widthFull + 1];
                                m2 = magBuffer[idx + widthFull - 1];
                                break;
                            case 90:
                                m1 = magBuffer[idx - widthFull];
                                m2 = magBuffer[idx + widthFull];
                                break;
                            default: // 135
                                m1 = magBuffer[idx - widthFull - 1];
                                m2 = magBuffer[idx + widthFull + 1];
                                break;
                        }

                        if (m < m1 || m < m2) {
                            *rowDst = 0;
                        } else {
                            int scaled = (m * 255) / maxGradient;
                            if (scaled > 255) scaled = 255;
                            *rowDst = (byte)scaled;
                        }
                    }
                });

                // STEP 3 - hysteresis
                byte* dstHyst = dstBase + dstStride * startY + startX;

                for (int y = startY; y < stopY; y++) {
                    for (int x = startX; x < stopX; x++, dstHyst++) {
                        if (*dstHyst < highThreshold) {
                            if (*dstHyst < lowThreshold) {
                                *dstHyst = 0;
                            } else {
                                if ((dstHyst[-1] < highThreshold) &&
                                    (dstHyst[1] < highThreshold) &&
                                    (dstHyst[-dstStride - 1] < highThreshold) &&
                                    (dstHyst[-dstStride] < highThreshold) &&
                                    (dstHyst[-dstStride + 1] < highThreshold) &&
                                    (dstHyst[dstStride - 1] < highThreshold) &&
                                    (dstHyst[dstStride] < highThreshold) &&
                                    (dstHyst[dstStride + 1] < highThreshold)) {
                                    *dstHyst = 0;
                                }
                            }
                        }
                    }
                    dstHyst += dstOffset;
                }

                // STEP 4 - draw black rectangle to remove those pixels, which were not processed
                Drawing.Rectangle(destination, rect, Color.Black);
            } finally {
                ArrayPool<byte>.Shared.Return(srcBuffer);
                ArrayPool<int>.Shared.Return(tempBlur);
                ArrayPool<byte>.Shared.Return(blurBuffer);
                ArrayPool<byte>.Shared.Return(orientBuffer);
                ArrayPool<int>.Shared.Return(magBuffer);
            }
        }
    }
}
