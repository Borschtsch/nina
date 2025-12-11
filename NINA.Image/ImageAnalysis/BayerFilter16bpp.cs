#region "copyright"

/*
    Copyright © 2016 - 2024 Stefan Berg <isbeorn86+NINA@googlemail.com> and the N.I.N.A. contributors

    This file is part of N.I.N.A. - Nighttime Imaging 'N' Astronomy.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#endregion "copyright"

using Accord.Imaging;
using Accord.Imaging.Filters;
using NINA.Core.Utility;
using NINA.Image.ImageData;
using System;
using System.Data.Entity.Core.Common.CommandTrees.ExpressionBuilder;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using static System.Windows.Forms.VisualStyles.VisualStyleElement.TrackBar;

namespace NINA.Image.ImageAnalysis {
    public sealed class BayerFilter16bpp : BayerFilter {

        public BayerFilter16bpp() {
            FormatTranslations[
                System.Drawing.Imaging.PixelFormat.Format16bppGrayScale] = System.Drawing.Imaging.PixelFormat.Format48bppRgb;
        }

        public bool SaveColorChannels { get; set; }
        public bool SaveLumChannel { get; set; }

        public LRGBArrays LRGBArrays { get; private set; }

        protected override unsafe void ProcessFilter(UnmanagedImage sourceData, UnmanagedImage destinationData) {
            int width = sourceData.Width;
            int height = sourceData.Height;

            int pixelCount = width * height;
            InitLRGBArrays(pixelCount);

            ushort* srcPtr = (ushort*)sourceData.ImageData.ToPointer();
            ushort* dstPtr = (ushort*)destinationData.ImageData.ToPointer();
            int srcStride = sourceData.Stride / 2; // in ushorts

            if (!PerformDemosaicing) {
                CopyPatternToRgb(srcPtr, dstPtr, width, height, srcStride);
            } else {
                DemosaicHybrid(srcPtr, dstPtr, width, height, srcStride);
            }

            if (SaveColorChannels || SaveLumChannel) {
                ExtractLrgba(dstPtr, width, height);
            }

#if DEBUG
            // 2) Verify result against reference implementation
            VerifyReference(sourceData, destinationData);
#endif
        }

        private void InitLRGBArrays(int pixelCount) {
            if (SaveColorChannels && SaveLumChannel) {
                LRGBArrays = new LRGBArrays(
                    new ushort[pixelCount],
                    new ushort[pixelCount],
                    new ushort[pixelCount],
                    new ushort[pixelCount]);
            } else if (SaveLumChannel) {
                LRGBArrays = new LRGBArrays(
                    new ushort[pixelCount],
                    Array.Empty<ushort>(),
                    Array.Empty<ushort>(),
                    Array.Empty<ushort>());
            } else if (SaveColorChannels) {
                LRGBArrays = new LRGBArrays(
                    Array.Empty<ushort>(),
                    new ushort[pixelCount],
                    new ushort[pixelCount],
                    new ushort[pixelCount]);
            } else {
                LRGBArrays = null;
            }
        }

        private unsafe void CopyPatternToRgb(
            ushort* srcPtr,
            ushort* dstPtr,
            int width,
            int height,
            int srcStride) {

            int dstStride = width * 3; // in ushorts, tightly packed 48bpp RGB

            ushort* src = srcPtr;
            ushort* dst = dstPtr;

            int srcOffset = srcStride - width;
            int dstOffset = dstStride - width * 3; // should be 0 if tightly packed

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++, src++, dst += 3) {
                    dst[RGB.R] = dst[RGB.G] = dst[RGB.B] = 0;
                    dst[BayerPattern[y & 1, x & 1]] = *src;
                }
                src += srcOffset;
                dst += dstOffset;
            }
        }

        private unsafe void DemosaicHybrid(
            ushort* srcPtr,
            ushort* dstPtr,
            int width,
            int height,
            int srcStride) {

            using (MyStopWatch.Measure()) {
                int innerStartY = 1;
                int innerEndY = height - 1;

                int p00 = BayerPattern[0, 0];
                int p01 = BayerPattern[0, 1];
                int p10 = BayerPattern[1, 0];
                int p11 = BayerPattern[1, 1];
                int[] flatPattern = { p00, p01, p10, p11 };

                // Process the interior rows in parallel; the hot path assumes valid neighbors
                // so we peel borders out to a separate pass to avoid per-pixel edge checks.
                Parallel.For(innerStartY, innerEndY, y => ProcessRowCpu(y, width, srcPtr, dstPtr, srcStride, flatPattern));

                ProcessBorders(width, height, srcPtr, dstPtr, srcStride);
            }
        }

        private struct ColumnAccum {
            public int s0, s1, s2;   // sums for channels 0,1,2
            public int c0, c1, c2;   // counts for channels 0,1,2
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Acc(ref ColumnAccum col, int ch, ushort v) {
            // ch is pat[...] value: 0,1,2
            switch (ch) {
                case 0:
                    col.s0 += v; col.c0++; break;
                case 1:
                    col.s1 += v; col.c1++; break;
                default:
                    col.s2 += v; col.c2++; break;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private unsafe static void InitColumn(
            ref ColumnAccum col,
            ushort* rowTop,
            ushort* rowMid,
            ushort* rowBot,
            int x,
            int[] pat,
            int baseTop,
            int baseMid,
            int baseBot) {
            // reset accumulators
            col.s0 = col.s1 = col.s2 = 0;
            col.c0 = col.c1 = col.c2 = 0;

            int xp = x & 1;

            // top
            ushort v = rowTop[x];
            int ch = pat[baseTop + xp];
            Acc(ref col, ch, v);

            // middle
            v = rowMid[x];
            ch = pat[baseMid + xp];
            Acc(ref col, ch, v);

            // bottom
            v = rowBot[x];
            ch = pat[baseBot + xp];
            Acc(ref col, ch, v);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private unsafe void ProcessRowCpu(
            int y,
            int width,
            ushort* srcBase,
            ushort* dstBase,
            int srcStride,
            int[] pat) {
            // Pointers to the three rows around y
            ushort* rowTop = srcBase + (y - 1) * srcStride;
            ushort* rowMid = srcBase + y * srcStride;
            ushort* rowBot = srcBase + (y + 1) * srcStride;

            // Row parity bases for Bayer pattern
            int baseTop = ((y - 1) & 1) << 1;   // (yTop & 1) * 2
            int baseMid = (y & 1) << 1;
            int baseBot = ((y + 1) & 1) << 1;

            // Destination pointer: first inner pixel (x = 1)
            ushort* dst = dstBase + (y * width + 1) * 3;

            // Column accumulators for x-1, x, x+1
            ColumnAccum colL = default, colC = default, colR = default, tmp = default;

            // Seed the sliding 3x3 window centered at x = 1 using columns 0, 1, 2
            InitColumn(ref colL, rowTop, rowMid, rowBot, 0, pat, baseTop, baseMid, baseBot);
            InitColumn(ref colC, rowTop, rowMid, rowBot, 1, pat, baseTop, baseMid, baseBot);
            InitColumn(ref colR, rowTop, rowMid, rowBot, 2, pat, baseTop, baseMid, baseBot);

            // Temporary arrays for final sums/counts per channel (stack) to avoid heap allocations
            int* sums = stackalloc int[3];
            int* counts = stackalloc int[3];

            int lastInnerX = width - 2;

            for (int x = 1; x < lastInnerX; x++, dst += 3) {
                // Sum the 3x3 window = sum of three columns
                sums[0] = colL.s0 + colC.s0 + colR.s0;
                sums[1] = colL.s1 + colC.s1 + colR.s1;
                sums[2] = colL.s2 + colC.s2 + colR.s2;

                counts[0] = colL.c0 + colC.c0 + colR.c0;
                counts[1] = colL.c1 + colC.c1 + colR.c1;
                counts[2] = colL.c2 + colC.c2 + colR.c2;

                // Use RGB enum indices, same as original code
                dst[RGB.R] = (ushort)(sums[RGB.R] / counts[RGB.R]);
                dst[RGB.G] = (ushort)(sums[RGB.G] / counts[RGB.G]);
                dst[RGB.B] = (ushort)(sums[RGB.B] / counts[RGB.B]);

                // Slide window horizontally by reusing two columns and refreshing the rightmost one
                int newX = x + 2; // new right column at x+2

                tmp = colL;
                colL = colC;
                colC = colR;

                InitColumn(ref tmp, rowTop, rowMid, rowBot, newX, pat, baseTop, baseMid, baseBot);
                colR = tmp;
            }

            // Process the final inner pixel at x = lastInnerX
            sums[0] = colL.s0 + colC.s0 + colR.s0;
            sums[1] = colL.s1 + colC.s1 + colR.s1;
            sums[2] = colL.s2 + colC.s2 + colR.s2;

            counts[0] = colL.c0 + colC.c0 + colR.c0;
            counts[1] = colL.c1 + colC.c1 + colR.c1;
            counts[2] = colL.c2 + colC.c2 + colR.c2;

            dst[RGB.R] = (ushort)(sums[RGB.R] / counts[RGB.R]);
            dst[RGB.G] = (ushort)(sums[RGB.G] / counts[RGB.G]);
            dst[RGB.B] = (ushort)(sums[RGB.B] / counts[RGB.B]);
        }

        private unsafe void ProcessBorders(
            int width,
            int height,
            ushort* srcBase,
            ushort* dstBase,
            int srcStride) {

            // Handles edge pixels that do not have a full 3x3 neighborhood; kept out of the main loop.
            void ProcessPixel(int x, int y) {
                ushort* src = srcBase + y * srcStride + x;
                ushort* dst = dstBase + (y * width + x) * 3;

                int[] vals = new int[3];
                int[] cnts = new int[3];

                for (int dy = -1; dy <= 1; dy++) {
                    int ny = y + dy;
                    if (ny < 0 || ny >= height) continue;

                    for (int dx = -1; dx <= 1; dx++) {
                        int nx = x + dx;
                        if (nx < 0 || nx >= width) continue;

                        int bIdx = BayerPattern[ny & 1, nx & 1];
                        vals[bIdx] += src[dy * srcStride + dx];
                        cnts[bIdx]++;
                    }
                }

                dst[RGB.R] = (ushort)(vals[RGB.R] / cnts[RGB.R]);
                dst[RGB.G] = (ushort)(vals[RGB.G] / cnts[RGB.G]);
                dst[RGB.B] = (ushort)(vals[RGB.B] / cnts[RGB.B]);
            }

            // top/bottom
            for (int x = 0; x < width; x++) {
                ProcessPixel(x, 0);
                ProcessPixel(x, height - 1);
            }

            // left/right (excluding corners)
            for (int y = 1; y < height - 1; y++) {
                ProcessPixel(0, y);
                ProcessPixel(width - 1, y);
            }
        }

        private unsafe void ExtractLrgba(ushort* dstPtr, int width, int height) {
            if (LRGBArrays == null)
                return;

            Parallel.For(0, height, y => {
                ushort* row = dstPtr + (y * width * 3);
                int offset = y * width;

                for (int x = 0; x < width; x++) {
                    ushort r = row[RGB.R];
                    ushort g = row[RGB.G];
                    ushort b = row[RGB.B];
                    row += 3;

                    if (SaveColorChannels) {
                        // match original mapping: Red=B, Green=G, Blue=R
                        LRGBArrays.Red[offset + x] = b;
                        LRGBArrays.Green[offset + x] = g;
                        LRGBArrays.Blue[offset + x] = r;
                    }

                    if (SaveLumChannel) {
                        LRGBArrays.Lum[offset + x] = (ushort)((r + g + b) / 3.0);
                    }
                }
            });
        }

        [Conditional("DEBUG")]
        private unsafe void VerifyReference(UnmanagedImage sourceData, UnmanagedImage destinationData) {
            // get width and height
            int width = sourceData.Width;
            int height = sourceData.Height;

            int widthM1 = width - 1;
            int heightM1 = height - 1;

            int srcStride = sourceData.Stride / 2;

            int srcOffset = (srcStride - width) / 2;
            int dstOffset = (destinationData.Stride - width * 6) / 6;

            ushort* src = (ushort*)sourceData.ImageData.ToPointer();
            ushort* dst = (ushort*)destinationData.ImageData.ToPointer();

            int[] rgbValues = new int[3];
            int[] rgbCounters = new int[3];

            if (!PerformDemosaicing) {
                // for each line
                for (int y = 0; y < height; y++) {
                    // for each pixel
                    for (int x = 0; x < width; x++, src++, dst += 3) {
                        ushort expR = 0, expG = 0, expB = 0;

                        int chan = BayerPattern[y & 1, x & 1];
                        if (chan == RGB.R) expR = *src;
                        else if (chan == RGB.G) expG = *src;
                        else expB = *src;

                        Debug.Assert(dst[RGB.R] == expR &&
                                     dst[RGB.G] == expG &&
                                     dst[RGB.B] == expB,
                            $"Bayer reference mismatch at ({x},{y})");
                    }

                    src += srcOffset;
                    dst += dstOffset;
                }
            } else {
                int counter = 0; // kept for structure parity with original, not used

                // for each line
                for (int y = 0; y < height; y++) {
                    // for each pixel
                    for (int x = 0; x < width; x++, src++, dst += 3) {
                        rgbValues[0] = rgbValues[1] = rgbValues[2] = 0;
                        rgbCounters[0] = rgbCounters[1] = rgbCounters[2] = 0;

                        int bayerIndex = BayerPattern[y & 1, x & 1];

                        rgbValues[bayerIndex] += *src;
                        rgbCounters[bayerIndex]++;

                        if (x != 0) {
                            bayerIndex = BayerPattern[y & 1, (x - 1) & 1];

                            rgbValues[bayerIndex] += src[-1];
                            rgbCounters[bayerIndex]++;
                        }

                        if (x != widthM1) {
                            bayerIndex = BayerPattern[y & 1, (x + 1) & 1];

                            rgbValues[bayerIndex] += src[1];
                            rgbCounters[bayerIndex]++;
                        }

                        if (y != 0) {
                            bayerIndex = BayerPattern[(y - 1) & 1, x & 1];

                            rgbValues[bayerIndex] += src[-srcStride];
                            rgbCounters[bayerIndex]++;

                            if (x != 0) {
                                bayerIndex = BayerPattern[(y - 1) & 1, (x - 1) & 1];

                                rgbValues[bayerIndex] += src[-srcStride - 1];
                                rgbCounters[bayerIndex]++;
                            }

                            if (x != widthM1) {
                                bayerIndex = BayerPattern[(y - 1) & 1, (x + 1) & 1];

                                rgbValues[bayerIndex] += src[-srcStride + 1];
                                rgbCounters[bayerIndex]++;
                            }
                        }

                        if (y != heightM1) {
                            bayerIndex = BayerPattern[(y + 1) & 1, x & 1];

                            rgbValues[bayerIndex] += src[srcStride];
                            rgbCounters[bayerIndex]++;

                            if (x != 0) {
                                bayerIndex = BayerPattern[(y + 1) & 1, (x - 1) & 1];

                                rgbValues[bayerIndex] += src[srcStride - 1];
                                rgbCounters[bayerIndex]++;
                            }

                            if (x != widthM1) {
                                bayerIndex = BayerPattern[(y + 1) & 1, (x + 1) & 1];

                                rgbValues[bayerIndex] += src[srcStride + 1];
                                rgbCounters[bayerIndex]++;
                            }
                        }

                        ushort expR = (ushort)(rgbValues[RGB.R] / rgbCounters[RGB.R]);
                        ushort expG = (ushort)(rgbValues[RGB.G] / rgbCounters[RGB.G]);
                        ushort expB = (ushort)(rgbValues[RGB.B] / rgbCounters[RGB.B]);

                        Debug.Assert(dst[RGB.R] == expR &&
                                     dst[RGB.G] == expG &&
                                     dst[RGB.B] == expB,
                            $"Demosaic reference mismatch at ({x},{y})");

                        counter++;
                    }

                    src += srcOffset;
                    dst += dstOffset;
                }
            }
        }
    }
}
