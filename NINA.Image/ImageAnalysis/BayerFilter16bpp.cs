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
using ILGPU.Algorithms;
using NINA.Core.Utility;
using NINA.Image.ImageAnalysis.HwAcceleration;
using NINA.Image.ImageData;
using Silk.NET.OpenCL;
using System;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.Windows.Media.TextFormatting;
using System.Collections.Generic;
using System.Diagnostics;

namespace NINA.Image.ImageAnalysis {
    public sealed class BayerFilter16bpp : BayerFilter {

        public BayerFilter16bpp() {
            FormatTranslations[
                System.Drawing.Imaging.PixelFormat.Format16bppGrayScale] =
                System.Drawing.Imaging.PixelFormat.Format48bppRgb;

            InitAccelerator();
        }

        private const string KernelSource = @"
        __kernel void debayer_3x3_inner(
            __global const ushort* src,
            __global ushort* dst,
            int width,
            int height,
            int srcStride,
            int idxR,
            int idxG,
            int idxB,
            int p00, int p01, int p10, int p11,
            int startY)
        {
            int lx = get_global_id(0);
            int ly = get_global_id(1);

            int x = lx + 1;
            int y = ly + startY;

            int rgbValues[3]   = { 0, 0, 0 };
            int rgbCounters[3] = { 0, 0, 0 };

            // tiny helper: decode Bayer pattern without an array
            #define BAYER_IDX(nx, ny) \
                (((ny & 1) << 1) | (nx & 1))

            #define PATTERN(bi) \
                ((bi) == 0 ? p00 : (bi) == 1 ? p01 : (bi) == 2 ? p10 : p11)

            #define TAP(ox, oy) do { \
                int nx = x + (ox); \
                int ny = y + (oy); \
                int pi = BAYER_IDX(nx, ny); \
                int b  = PATTERN(pi); \
                ushort v = src[ny * srcStride + nx]; \
                rgbValues[b]   += v; \
                rgbCounters[b] += 1; \
            } while (0)

            TAP(-1, -1); TAP(0, -1); TAP(1, -1);
            TAP(-1,  0); TAP(0,  0); TAP(1,  0);
            TAP(-1,  1); TAP(0,  1); TAP(1,  1);

            #undef TAP
            #undef PATTERN
            #undef BAYER_IDX

            int dstIndex = (y * width + x) * 3;

            dst[dstIndex + idxR] = (ushort)(rgbValues[idxR] / rgbCounters[idxR]);
            dst[dstIndex + idxG] = (ushort)(rgbValues[idxG] / rgbCounters[idxG]);
            dst[dstIndex + idxB] = (ushort)(rgbValues[idxB] / rgbCounters[idxB]);
        }
        ";

        public bool SaveColorChannels { get; set; }
        public bool SaveLumChannel { get; set; }

        public LRGBArrays LRGBArrays { get; private set; }

        private OpenClAccelerator accelerator = OpenClAccelerator.Instance;

        // New: multi-device executor
        private static MultiDevice2DExecutor? multiDeviceExecutor;

        private void InitAccelerator() {
            // Multi-device executor
            if (multiDeviceExecutor == null) {
                try {
                    var exec = new MultiDevice2DExecutor(
                        KernelSource,
                        "debayer_3x3_inner",
                        maxDevices: -1);

                    if (exec.GpuWorkerCount > 0) {
                        multiDeviceExecutor = exec;
                    } else {
                        exec.Dispose();
                    }
                } catch (Exception ex) {
                    Logger.Warning($"OpenCL multi-device init failed: {ex.Message}");
                    multiDeviceExecutor = null;
                }
            }
        }

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

            int innerStartY = 1;
            int innerEndY = height - 1;
            int innerRows = innerEndY - innerStartY;
            if (innerRows <= 0)
                return;

            int p00 = BayerPattern[0, 0];
            int p01 = BayerPattern[0, 1];
            int p10 = BayerPattern[1, 0];
            int p11 = BayerPattern[1, 1];
            int[] flatPattern = { p00, p01, p10, p11 };

            // CPU worker delegate (bottom strip)
            Action<int, int> cpuWorker = (startY, endY) =>
                Parallel.For(startY, endY, y =>
                    ProcessRowCpu(y, width, srcPtr, dstPtr, srcStride, flatPattern));

            // GPU worker delegate using per-worker buffers
            multiDeviceExecutor!.Execute(
                width: width,
                height: height,
                borderTop: 1,
                borderBottom: 1,
                srcPtr: srcPtr,
                dstPtr: dstPtr,
                srcStride: srcStride,
                cpuWorker: cpuWorker,
                gpuWorker: (acc, kernel, srcBuf, dstBuf, startY, endY) =>
                    ProcessStripOnAccelerator(
                        acc,
                        kernel,
                        srcBuf,
                        dstBuf,
                        dstPtr,
                        width,
                        height,
                        srcStride,
                        startY,
                        endY,
                        flatPattern));

            ProcessBorders(width, height, srcPtr, dstPtr, srcStride);
        }

        /// <summary>
        /// Generic GPU strip processor that can run on any accelerator/kernel pair.
        /// Used by the multi-device executor.
        /// </summary>
        /// <summary>
        /// GPU strip processor: uses per-worker src/dst buffers (no allocation).
        /// Borders excluded.
        /// </summary>
        private unsafe void ProcessStripOnAccelerator(
            OpenClAccelerator acc,
            ClKernel kernel,
            ClBuffer srcBuf,
            ClBuffer dstBuf,
            ushort* dstPtr,
            int width,
            int height,
            int srcStride,
            int startY,
            int endY,
            int[] flatPattern) {

            Debug.Assert(acc != null);
            Debug.Assert(kernel.IsValid);
            Debug.Assert(srcBuf.IsValid);
            Debug.Assert(dstBuf.IsValid);
            Debug.Assert(startY >= 1 && endY <= height - 1);
            Debug.Assert(endY > startY);

            int gpuRows = endY - startY;
            int rowStrideElems = width * 3;

            // 1. Kernel args
            acc.SetKernelArg(kernel, 0, srcBuf);
            acc.SetKernelArg(kernel, 1, dstBuf);
            acc.SetKernelArg(kernel, 2, width);
            acc.SetKernelArg(kernel, 3, height);
            acc.SetKernelArg(kernel, 4, srcStride);
            acc.SetKernelArg(kernel, 5, (int)RGB.R);
            acc.SetKernelArg(kernel, 6, (int)RGB.G);
            acc.SetKernelArg(kernel, 7, (int)RGB.B);
            acc.SetKernelArg(kernel, 8, flatPattern[0]);
            acc.SetKernelArg(kernel, 9, flatPattern[1]);
            acc.SetKernelArg(kernel, 10, flatPattern[2]);
            acc.SetKernelArg(kernel, 11, flatPattern[3]);
            acc.SetKernelArg(kernel, 12, startY);

            // 2. Run inner strip: x = [1, width-2], y = [startY, endY)
            acc.RunKernel2D(kernel, width - 2, gpuRows); // blocking

            // 3. Read back only [startY, endY) rows into dstPtr
            int startElem = startY * rowStrideElems;
            int countElem = gpuRows * rowStrideElems;

            var dstSpan = new Span<ushort>(dstPtr + startElem, countElem);
            acc.ReadBufferRegion(dstBuf, startElem, dstSpan);
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
            int widthM1 = width - 1;

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

            // Initial window centered at x = 1 → columns 0,1,2
            InitColumn(ref colL, rowTop, rowMid, rowBot, 0, pat, baseTop, baseMid, baseBot);
            InitColumn(ref colC, rowTop, rowMid, rowBot, 1, pat, baseTop, baseMid, baseBot);
            InitColumn(ref colR, rowTop, rowMid, rowBot, 2, pat, baseTop, baseMid, baseBot);

            // Temporary arrays for final sums/counts per channel
            int[] sums = new int[3];
            int[] counts = new int[3];

            // Process inner pixels x = 1 .. width-2
            for (int x = 1; x < widthM1; x++, dst += 3) {
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

                // Slide window horizontally, except after last pixel
                if (x + 1 < widthM1) {
                    int newX = x + 2; // new right column at x+2

                    // L <- C, C <- R, R <- new column
                    tmp = colL;
                    colL = colC;
                    colC = colR;

                    InitColumn(ref tmp, rowTop, rowMid, rowBot, newX, pat, baseTop, baseMid, baseBot);
                    colR = tmp;
                }
            }
        }

        private unsafe void ProcessBorders(
            int width,
            int height,
            ushort* srcBase,
            ushort* dstBase,
            int srcStride) {

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
