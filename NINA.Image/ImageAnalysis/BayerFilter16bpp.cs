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

        /// <summary>
        /// 0.0 = CPU only; 1.0 = GPU handles all inner rows.
        /// </summary>
        private double GpuRatio { get; set; } = 0.6666;

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

            double ratio = Math.Clamp(GpuRatio, 0.0, 1.0);

            bool hasMultiGpu = multiDeviceExecutor != null && multiDeviceExecutor.GpuWorkerCount > 0;
            bool useGpu = ratio > 0.0 && hasMultiGpu;

            if (!useGpu) {
                // pure CPU
                Parallel.For(innerStartY, innerEndY, y =>
                    ProcessRowCpu(y, width, srcPtr, dstPtr, srcStride, flatPattern));
                ProcessBorders(width, height, srcPtr, dstPtr, srcStride);
                return;
            }

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

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private unsafe void ProcessRowCpu(
            int y,
            int width,
            ushort* srcBase,
            ushort* dstBase,
            int srcStride,
            int[] pat) {

            int widthM1 = width - 1;

            ushort* src = srcBase + y * srcStride + 1;
            ushort* dst = dstBase + (y * width + 1) * 3;

            int[] rgbValues = new int[3];
            int[] rgbCounters = new int[3];

            for (int x = 1; x < widthM1; x++, src++, dst += 3) {
                rgbValues[0] = rgbValues[1] = rgbValues[2] = 0;
                rgbCounters[0] = rgbCounters[1] = rgbCounters[2] = 0;

                // center
                int cIdx = pat[(y & 1) * 2 + (x & 1)];
                rgbValues[cIdx] += *src;
                rgbCounters[cIdx]++;

                // left/right
                int idxL = pat[(y & 1) * 2 + ((x - 1) & 1)];
                rgbValues[idxL] += src[-1];
                rgbCounters[idxL]++;

                int idxR = pat[(y & 1) * 2 + ((x + 1) & 1)];
                rgbValues[idxR] += src[1];
                rgbCounters[idxR]++;

                // top row
                int yTop = (y - 1) & 1;
                int idxT = pat[yTop * 2 + (x & 1)];
                rgbValues[idxT] += src[-srcStride];
                rgbCounters[idxT]++;

                int idxTL = pat[yTop * 2 + ((x - 1) & 1)];
                rgbValues[idxTL] += src[-srcStride - 1];
                rgbCounters[idxTL]++;

                int idxTR = pat[yTop * 2 + ((x + 1) & 1)];
                rgbValues[idxTR] += src[-srcStride + 1];
                rgbCounters[idxTR]++;

                // bottom row
                int yBot = (y + 1) & 1;
                int idxB = pat[yBot * 2 + (x & 1)];
                rgbValues[idxB] += src[srcStride];
                rgbCounters[idxB]++;

                int idxBL = pat[yBot * 2 + ((x - 1) & 1)];
                rgbValues[idxBL] += src[srcStride - 1];
                rgbCounters[idxBL]++;

                int idxBR = pat[yBot * 2 + ((x + 1) & 1)];
                rgbValues[idxBR] += src[srcStride + 1];
                rgbCounters[idxBR]++;

                dst[RGB.R] = (ushort)(rgbValues[RGB.R] / rgbCounters[RGB.R]);
                dst[RGB.G] = (ushort)(rgbValues[RGB.G] / rgbCounters[RGB.G]);
                dst[RGB.B] = (ushort)(rgbValues[RGB.B] / rgbCounters[RGB.B]);
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
    }
}
