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
using NINA.Core.Enum;
using NINA.Image.ImageData;
using System;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Threading;

namespace NINA.Image.ImageAnalysis {

    internal sealed class DebayerBufferPool : IDisposable {
        private static readonly Lazy<DebayerBufferPool> _instance = new Lazy<DebayerBufferPool>(() => new DebayerBufferPool());
        public static DebayerBufferPool Instance => _instance.Value;

        private sealed class BufferSet : IDisposable {
            public int Width;
            public int Height;
            public IntPtr InputPtr = IntPtr.Zero;
            public IntPtr OutputPtr = IntPtr.Zero;
            public UnmanagedImage InputImage;
            public UnmanagedImage OutputImage;
            public BayerFilter16bpp Filter;
            public SensorType Pattern;
            public bool SaveColors;
            public bool SaveLum;

            public void EnsureBuffers(int width, int height) {
                if (InputPtr != IntPtr.Zero && width == Width && height == Height) {
                    return;
                }

                Dispose();
                Width = width;
                Height = height;

                int inputStride = width * 2;   // 16bpp grayscale
                int outputStride = width * 6;  // 48bpp RGB
                int inputSize = inputStride * height;
                int outputSize = outputStride * height;

                InputPtr = Marshal.AllocHGlobal(inputSize);
                OutputPtr = Marshal.AllocHGlobal(outputSize);
                InputImage = new UnmanagedImage(InputPtr, width, height, inputStride, System.Drawing.Imaging.PixelFormat.Format16bppGrayScale);
                OutputImage = new UnmanagedImage(OutputPtr, width, height, outputStride, System.Drawing.Imaging.PixelFormat.Format48bppRgb);
            }

            public BayerFilter16bpp EnsureFilter(bool saveColorChannels, bool saveLumChannel, SensorType pattern) {
                if (Filter == null || Pattern != pattern || SaveColors != saveColorChannels || SaveLum != saveLumChannel) {
                    Filter = new BayerFilter16bpp {
                        SaveColorChannels = saveColorChannels,
                        SaveLumChannel = saveLumChannel,
                        BayerPattern = pattern switch {
                            // Standard CFA definitions: row0,row1 = top->bottom, col0,col1 = left->right
                            SensorType.RGGB => new int[,] { { RGB.R, RGB.G }, { RGB.G, RGB.B } },
                            SensorType.RGBG => new int[,] { { RGB.R, RGB.G }, { RGB.B, RGB.G } },
                            SensorType.GRGB => new int[,] { { RGB.G, RGB.R }, { RGB.G, RGB.B } },
                            SensorType.GRBG => new int[,] { { RGB.G, RGB.R }, { RGB.B, RGB.G } },
                            SensorType.GBGR => new int[,] { { RGB.G, RGB.B }, { RGB.R, RGB.G } },
                            SensorType.GBRG => new int[,] { { RGB.G, RGB.B }, { RGB.R, RGB.G } },
                            SensorType.BGRG => new int[,] { { RGB.B, RGB.G }, { RGB.R, RGB.G } },
                            SensorType.BGGR => new int[,] { { RGB.B, RGB.G }, { RGB.G, RGB.R } },
                            _ => new int[,] { { RGB.R, RGB.G }, { RGB.G, RGB.B } }
                        }
                    };
                    Pattern = pattern;
                    SaveColors = saveColorChannels;
                    SaveLum = saveLumChannel;
                }
                return Filter;
            }

            public void Dispose() {
                if (InputPtr != IntPtr.Zero) {
                    Marshal.FreeHGlobal(InputPtr);
                    InputPtr = IntPtr.Zero;
                }
                if (OutputPtr != IntPtr.Zero) {
                    Marshal.FreeHGlobal(OutputPtr);
                    OutputPtr = IntPtr.Zero;
                }
                InputImage = null;
                OutputImage = null;
                Filter = null;
            }
        }

        internal readonly struct DebayerResources {
            public DebayerResources(
                IntPtr inputPtr,
                int inputSizeBytes,
                int inputStride,
                UnmanagedImage inputImage,
                IntPtr outputPtr,
                int outputSizeBytes,
                int outputStride,
                UnmanagedImage outputImage,
                BayerFilter16bpp filter) {
                InputPtr = inputPtr;
                InputSizeBytes = inputSizeBytes;
                InputStride = inputStride;
                InputImage = inputImage;
                OutputPtr = outputPtr;
                OutputSizeBytes = outputSizeBytes;
                OutputStride = outputStride;
                OutputImage = outputImage;
                Filter = filter;
            }

            public IntPtr InputPtr { get; }
            public int InputSizeBytes { get; }
            public int InputStride { get; }
            public UnmanagedImage InputImage { get; }
            public IntPtr OutputPtr { get; }
            public int OutputSizeBytes { get; }
            public int OutputStride { get; }
            public UnmanagedImage OutputImage { get; }
            public BayerFilter16bpp Filter { get; }
        }

        private readonly ThreadLocal<BufferSet> _threadBuffers;
        private readonly System.Collections.Concurrent.ConcurrentBag<BufferSet> _allBuffers;

        private DebayerBufferPool() {
            _allBuffers = new System.Collections.Concurrent.ConcurrentBag<BufferSet>();
            _threadBuffers = new ThreadLocal<BufferSet>(() => {
                var set = new BufferSet();
                _allBuffers.Add(set);
                return set;
            });
        }

        internal DebayerResources Acquire(int width, int height, bool saveColorChannels, bool saveLumChannel, SensorType pattern) {
            var set = _threadBuffers.Value;
            set.EnsureBuffers(width, height);
            var filter = set.EnsureFilter(saveColorChannels, saveLumChannel, pattern);

            int inputStride = width * 2;
            int outputStride = width * 6;
            return new DebayerResources(
                set.InputPtr, inputStride * height, inputStride, set.InputImage,
                set.OutputPtr, outputStride * height, outputStride, set.OutputImage,
                filter);
        }

        public void Dispose() {
            foreach (var set in _allBuffers) {
                set.Dispose();
            }
        }
    }
}
