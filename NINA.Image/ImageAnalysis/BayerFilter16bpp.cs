#region "copyright"

/*
    Copyright c 2016 - 2024 Stefan Berg <isbeorn86+NINA@googlemail.com> and the N.I.N.A. contributors

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
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

namespace NINA.Image.ImageAnalysis {
    // High-throughput 16bpp Bayer demosaic tuned for bit-precise output and low GC pressure.
    // All hot paths avoid allocations and keep math deterministic across CPUs.
    public sealed class BayerFilter16bpp : BayerFilter {

        public BayerFilter16bpp() {
            // Map incoming 16bpp grayscale sensor data to 48bpp RGB for Accord's expectations.
            FormatTranslations[
                System.Drawing.Imaging.PixelFormat.Format16bppGrayScale] = System.Drawing.Imaging.PixelFormat.Format48bppRgb;
        }

        // Toggle storing per-channel planes derived from the debayered image.
        public bool SaveColorChannels { get; set; }
        // Toggle storing luminance derived from the debayered image.
        public bool SaveLumChannel { get; set; }

        // Carries optional luminance/R/G/B planes extracted during debayer for downstream consumers.
        public LRGBArrays LRGBArrays { get; private set; }

        internal unsafe void ApplyInto(UnmanagedImage sourceData, UnmanagedImage destinationData) {
            // Pull dimensions once to keep pointer math simple and branch-free inside loops.
            int width = sourceData.Width;
            int height = sourceData.Height;

            int pixelCount = width * height;
            // Pre-size optional L/R/G/B arrays; this is the only place we allow allocating for extraction.
            InitLRGBArrays(pixelCount);

            ushort* srcPtr = (ushort*)sourceData.ImageData.ToPointer();
            ushort* dstPtr = (ushort*)destinationData.ImageData.ToPointer();
            // Stride must be even for 16bpp input; enforce to keep pointer math bit-precise.
            if ((sourceData.Stride & 1) != 0) {
                throw new ArgumentException("Expected even stride for 16bpp source", nameof(sourceData));
            }
            int srcStride = sourceData.Stride / 2; // in ushorts

            if ((destinationData.Stride & 1) != 0) {
                throw new ArgumentException("Expected even stride for 48bpp destination", nameof(destinationData));
            }
            int dstStride = destinationData.Stride / 2; // in ushorts

            // Pin the optional output arrays so we can fill them during processing without an extra post-pass.
            if (!PerformDemosaicing) {
                // Pass null pointers; copy-only path rarely needs the per-channel planes, so keep behavior but no demosaic.
                CopyPatternToRgb(srcPtr, dstPtr, width, height, srcStride, dstStride);
            } else {
                // Fuse demosaic + optional channel/luma extraction so we avoid the extra image-sized pass.
                Demosaic(srcPtr, dstPtr, width, height, srcStride, dstStride);
            }

            if (SaveColorChannels || SaveLumChannel) {

                ExtractLrgba(dstPtr, width, height, dstStride);
            }

#if DEBUG
            // 2) Verify result against reference implementation
            VerifyReference(sourceData, destinationData);
#endif
        }

        private void InitLRGBArrays(int pixelCount) {
            // Respect flags: allocate only what is requested and avoid touching heap when not needed.
            if (SaveColorChannels && SaveLumChannel) {
                LRGBArrays = EnsureArrays(
                    LRGBArrays,
                    pixelCount,
                    needLum: true,
                    needColors: true);
            } else if (SaveLumChannel) {
                LRGBArrays = EnsureArrays(
                    LRGBArrays,
                    pixelCount,
                    needLum: true,
                    needColors: false);
            } else if (SaveColorChannels) {
                LRGBArrays = EnsureArrays(
                    LRGBArrays,
                    pixelCount,
                    needLum: false,
                    needColors: true);
            } else {
                LRGBArrays = null;
            }
        }

        private static LRGBArrays EnsureArrays(
            LRGBArrays current,
            int pixelCount,
            bool needLum,
            bool needColors) {

            // Reuse existing arrays when dimension matches to avoid GC; clear only when reused.
            ushort[] lum = needLum
                ? (current?.Lum != null && current.Lum.Length == pixelCount ? current.Lum : new ushort[pixelCount])
                : Array.Empty<ushort>();

            ushort[] r = needColors
                ? (current?.Red != null && current.Red.Length == pixelCount ? current.Red : new ushort[pixelCount])
                : Array.Empty<ushort>();

            ushort[] g = needColors
                ? (current?.Green != null && current.Green.Length == pixelCount ? current.Green : new ushort[pixelCount])
                : Array.Empty<ushort>();

            ushort[] b = needColors
                ? (current?.Blue != null && current.Blue.Length == pixelCount ? current.Blue : new ushort[pixelCount])
                : Array.Empty<ushort>();

            if (needLum && lum.Length == pixelCount)
                Array.Clear(lum, 0, lum.Length);
            if (needColors) {
                if (r.Length == pixelCount) Array.Clear(r, 0, r.Length);
                if (g.Length == pixelCount) Array.Clear(g, 0, g.Length);
                if (b.Length == pixelCount) Array.Clear(b, 0, b.Length);
            }

            return new LRGBArrays(lum, r, g, b);
        }

        private unsafe void CopyPatternToRgb(
            ushort* srcPtr,
            ushort* dstPtr,
            int width,
            int height,
            int srcStride,
            int dstStride) {

            // dstStride is passed in ushorts
            ushort* src = srcPtr;
            ushort* dst = dstPtr;

            int srcOffset = srcStride - width;
            int dstOffset = dstStride - width * 3; // Jump over padding and the width we just wrote

            for (int y = 0; y < height; y++) {
                // For raw copy we simply place the Bayer value into its channel slot and zero the others.
                for (int x = 0; x < width; x++, src++, dst += 3) {
                    dst[RGB.R] = dst[RGB.G] = dst[RGB.B] = 0;
                    dst[BayerPattern[y & 1, x & 1]] = *src;
                }
                src += srcOffset;
                dst += dstOffset;
            }
        }

        private unsafe void Demosaic(
            ushort* srcPtr,
            ushort* dstPtr,
            int width,
            int height,
            int srcStride,
            int dstStride) {

            // Core pipeline: parallel interior sweep with a 3x3 average, then explicit border handling.
            using (MyStopWatch.Measure()) {
                int innerStartY = 1;
                int innerEndY = height - 1;

                int p00 = BayerPattern[0, 0];
                int p01 = BayerPattern[0, 1];
                int p10 = BayerPattern[1, 0];
                int p11 = BayerPattern[1, 1];
                // Flatten the 2x2 Bayer tile into a contiguous array to keep inner loop indexing branchless.
                int[] flatPattern = { p00, p01, p10, p11 };

                // Use larger static row chunks to reduce scheduler overhead while preserving computation.
                // Chunk size heuristic: at least 16 rows and ~2 chunks per logical core to balance work without overscheduling.
                int targetChunk = Math.Max(16, (innerEndY - innerStartY) / Math.Max(1, Environment.ProcessorCount * 2));
                var part = Partitioner.Create(innerStartY, innerEndY, targetChunk);
                Parallel.ForEach(part, range => {
                    // Thread-local column accumulators reused across rows in this chunk.
                    for (int y = range.Item1; y < range.Item2; y++) {
                        ProcessRowCpu(y, width, srcPtr, dstPtr, srcStride, dstStride, flatPattern);
                    }
                });

                ProcessBorders(width, height, srcPtr, dstPtr, srcStride, dstStride);
            }
        }

        protected override unsafe void ProcessFilter(UnmanagedImage sourceData, UnmanagedImage destinationData) {
            // Accord filter entry point delegates to the optimized implementation above.
            ApplyInto(sourceData, destinationData);
        }

        // Holds running sums/counts for one column of the 3x3 window; reused to avoid recomputing per pixel.
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
            // reset accumulators; per-column counts are not tracked because channel counts are parity-constant and precomputed.
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
            int dstStride,
            int[] pat) {
            // Caller ensures y is away from borders so we can sample y-1..y+1 without bounds checks.
            // Pointers to the three rows around y
            ushort* rowTop = srcBase + (y - 1) * srcStride;
            ushort* rowMid = srcBase + y * srcStride;
            ushort* rowBot = srcBase + (y + 1) * srcStride;

            // Row parity bases for Bayer pattern
            int baseTop = ((y - 1) & 1) << 1;   // (yTop & 1) * 2
            int baseMid = (y & 1) << 1;
            int baseBot = ((y + 1) & 1) << 1;

            // Destination pointer: first inner pixel (x = 1)
            ushort* dst = dstBase + y * dstStride + 3; // y * stride + x*3 (x=1)

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
                ushort rVal = (ushort)(sums[RGB.R] / counts[RGB.R]);
                ushort gVal = (ushort)(sums[RGB.G] / counts[RGB.G]);
                ushort bVal = (ushort)(sums[RGB.B] / counts[RGB.B]);

                dst[RGB.R] = rVal;
                dst[RGB.G] = gVal;
                dst[RGB.B] = bVal;

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

            ushort rLast = (ushort)(sums[RGB.R] / counts[RGB.R]);
            ushort gLast = (ushort)(sums[RGB.G] / counts[RGB.G]);
            ushort bLast = (ushort)(sums[RGB.B] / counts[RGB.B]);

            dst[RGB.R] = rLast;
            dst[RGB.G] = gLast;
            dst[RGB.B] = bLast;
        }

        private unsafe void ProcessBorders(
            int width,
            int height,
            ushort* srcBase,
            ushort* dstBase,
            int srcStride,
            int dstStride) {

            // Explicitly handle edges where the 3x3 window would step outside the frame.
            // Pre-allocate buffers for border processing to avoid repeated stackalloc or method call overhead
            int* vals = stackalloc int[3];
            int* cnts = stackalloc int[3];

            // Local helper to process one pixel using the hoisted buffers
            void ProcessPixel(int x, int y) {
                ushort* src = srcBase + y * srcStride + x;
                ushort* dst = dstBase + y * dstStride + x * 3;

                // Reset accumulators
                vals[0] = vals[1] = vals[2] = 0;
                cnts[0] = cnts[1] = cnts[2] = 0;

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

                ushort rVal = cnts[RGB.R] > 0 ? (ushort)(vals[RGB.R] / cnts[RGB.R]) : (ushort)0;
                ushort gVal = cnts[RGB.G] > 0 ? (ushort)(vals[RGB.G] / cnts[RGB.G]) : (ushort)0;
                ushort bVal = cnts[RGB.B] > 0 ? (ushort)(vals[RGB.B] / cnts[RGB.B]) : (ushort)0;

                dst[RGB.R] = rVal;
                dst[RGB.G] = gVal;
                dst[RGB.B] = bVal;
            }

            // top/bottom
            for (int x = 0; x < width; x++) {
                ProcessPixel(x, 0);
                ProcessPixel(x, height - 1);
            }

            // left/right (excluding corners to avoid double processing, though corners were safe in original loop ranges 0..width/height)
            // Original code: top/bottom X 0..width. Left/Right Y 1..height-2. Correct coverage.
            for (int y = 1; y < height - 1; y++) {
                ProcessPixel(0, y);
                ProcessPixel(width - 1, y);
            }
        }

        private unsafe void ExtractLrgba(ushort* dstPtr, int width, int height, int dstStride) {
            if (LRGBArrays == null)
                return;

            // Post-pass that populates preallocated channel/luma arrays directly from the demosaiced RGB buffer.
            bool doColors = SaveColorChannels;
            bool doLum = SaveLumChannel;

            var redArr = LRGBArrays.Red;
            var greenArr = LRGBArrays.Green;
            var blueArr = LRGBArrays.Blue;
            var lumArr = LRGBArrays.Lum;

            // Chunk rows to reduce Parallel.For scheduling overhead on low-core CPUs.
            int chunk = Math.Max(16, height / Math.Max(1, Environment.ProcessorCount * 2));

            // Maintain legacy channel ordering: arrays store the demosaiced B,G,R to stay byte-for-byte compatible.
            // Pin arrays outside the lambda; copy pointers into locals before entering the Parallel.ForEach.
            fixed (ushort* redBase = redArr, greenBase = greenArr, blueBase = blueArr, lumBase = lumArr) {
                ushort* redPtr = redBase;
                ushort* greenPtr = greenBase;
                ushort* bluePtr = blueBase;
                ushort* lumPtr = lumBase;

                Parallel.ForEach(Partitioner.Create(0, height, chunk), range => {
                    for (int y = range.Item1; y < range.Item2; y++) {
                        ushort* row = dstPtr + (y * dstStride);
                        int offset = y * width;

                        int x = 0;
                        int limit = width - 3; // process 4 pixels per iteration to unroll inner loop

                        if (doColors && doLum) {
                            // Both color planes and luminance requested.
                            for (; x <= limit; x += 4, row += 12) {
                                // Pixel 0
                                ushort r0 = row[RGB.R];
                                ushort g0 = row[RGB.G];
                                ushort b0 = row[RGB.B];
                                redPtr[offset + x] = b0;
                                greenPtr[offset + x] = g0;
                                bluePtr[offset + x] = r0;
                                lumPtr[offset + x] = (ushort)((r0 + g0 + b0) / 3);

                                // Pixel 1
                                ushort r1 = row[3 + RGB.R];
                                ushort g1 = row[3 + RGB.G];
                                ushort b1 = row[3 + RGB.B];
                                redPtr[offset + x + 1] = b1;
                                greenPtr[offset + x + 1] = g1;
                                bluePtr[offset + x + 1] = r1;
                                lumPtr[offset + x + 1] = (ushort)((r1 + g1 + b1) / 3);

                                // Pixel 2
                                ushort r2 = row[6 + RGB.R];
                                ushort g2 = row[6 + RGB.G];
                                ushort b2 = row[6 + RGB.B];
                                redPtr[offset + x + 2] = b2;
                                greenPtr[offset + x + 2] = g2;
                                bluePtr[offset + x + 2] = r2;
                                lumPtr[offset + x + 2] = (ushort)((r2 + g2 + b2) / 3);

                                // Pixel 3
                                ushort r3 = row[9 + RGB.R];
                                ushort g3 = row[9 + RGB.G];
                                ushort b3 = row[9 + RGB.B];
                                redPtr[offset + x + 3] = b3;
                                greenPtr[offset + x + 3] = g3;
                                bluePtr[offset + x + 3] = r3;
                                lumPtr[offset + x + 3] = (ushort)((r3 + g3 + b3) / 3);
                            }
                            // Remainder pixels
                            for (; x < width; x++, row += 3) {
                                ushort r = row[RGB.R];
                                ushort g = row[RGB.G];
                                ushort b = row[RGB.B];
                                redPtr[offset + x] = b;
                                greenPtr[offset + x] = g;
                                bluePtr[offset + x] = r;
                                lumPtr[offset + x] = (ushort)((r + g + b) / 3);
                            }
                        } else if (doColors) {
                            // Only color planes requested.
                            for (; x <= limit; x += 4, row += 12) {
                                ushort r0 = row[RGB.R];
                                ushort g0 = row[RGB.G];
                                ushort b0 = row[RGB.B];
                                redPtr[offset + x] = b0;
                                greenPtr[offset + x] = g0;
                                bluePtr[offset + x] = r0;

                                ushort r1 = row[3 + RGB.R];
                                ushort g1 = row[3 + RGB.G];
                                ushort b1 = row[3 + RGB.B];
                                redPtr[offset + x + 1] = b1;
                                greenPtr[offset + x + 1] = g1;
                                bluePtr[offset + x + 1] = r1;

                                ushort r2 = row[6 + RGB.R];
                                ushort g2 = row[6 + RGB.G];
                                ushort b2 = row[6 + RGB.B];
                                redPtr[offset + x + 2] = b2;
                                greenPtr[offset + x + 2] = g2;
                                bluePtr[offset + x + 2] = r2;

                                ushort r3 = row[9 + RGB.R];
                                ushort g3 = row[9 + RGB.G];
                                ushort b3 = row[9 + RGB.B];
                                redPtr[offset + x + 3] = b3;
                                greenPtr[offset + x + 3] = g3;
                                bluePtr[offset + x + 3] = r3;
                            }
                            for (; x < width; x++, row += 3) {
                                ushort r = row[RGB.R];
                                ushort g = row[RGB.G];
                                ushort b = row[RGB.B];
                                redPtr[offset + x] = b;
                                greenPtr[offset + x] = g;
                                bluePtr[offset + x] = r;
                            }
                        } else if (doLum) {
                            // Only luminance requested.
                            for (; x <= limit; x += 4, row += 12) {
                                ushort r0 = row[RGB.R];
                                ushort g0 = row[RGB.G];
                                ushort b0 = row[RGB.B];
                                lumPtr[offset + x] = (ushort)((r0 + g0 + b0) / 3);

                                ushort r1 = row[3 + RGB.R];
                                ushort g1 = row[3 + RGB.G];
                                ushort b1 = row[3 + RGB.B];
                                lumPtr[offset + x + 1] = (ushort)((r1 + g1 + b1) / 3);

                                ushort r2 = row[6 + RGB.R];
                                ushort g2 = row[6 + RGB.G];
                                ushort b2 = row[6 + RGB.B];
                                lumPtr[offset + x + 2] = (ushort)((r2 + g2 + b2) / 3);

                                ushort r3 = row[9 + RGB.R];
                                ushort g3 = row[9 + RGB.G];
                                ushort b3 = row[9 + RGB.B];
                                lumPtr[offset + x + 3] = (ushort)((r3 + g3 + b3) / 3);
                            }
                            for (; x < width; x++, row += 3) {
                                ushort r = row[RGB.R];
                                ushort g = row[RGB.G];
                                ushort b = row[RGB.B];
                                lumPtr[offset + x] = (ushort)((r + g + b) / 3);
                            }
                        }
                    }
                });
            }
        }

        [Conditional("DEBUG")]
        private unsafe void VerifyReference(UnmanagedImage sourceData, UnmanagedImage destinationData) {
            // Cross-check against the scalar reference implementation to guard against bit drift in hot paths.
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
