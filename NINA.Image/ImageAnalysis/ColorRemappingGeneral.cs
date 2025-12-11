#region "copyright"

/*
    Copyright ? 2016 - 2024 Stefan Berg <isbeorn86+NINA@googlemail.com> and the N.I.N.A. contributors

    This file is part of N.I.N.A. - Nighttime Imaging 'N' Astronomy.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#endregion "copyright"

using Accord.Imaging;
using Accord.Imaging.Filters;
using System;
using System.Buffers;
using System.Diagnostics;
using System.Drawing;
using System.Threading.Tasks;

namespace NINA.Image.ImageAnalysis {

    public class ColorRemappingGeneral : ColorRemapping {

        // color maps
        private ushort[] redMap;

        private ushort[] greenMap;
        private ushort[] blueMap;
        private ushort[] grayMap;

        public ushort[] RedMap16 {
            get => redMap;
            set {
                // check the map
                if ((value == null) || (value.Length != 65536))
                    throw new ArgumentException("A map should be array with 65536 value.");

                redMap = value;
            }
        }

        public ushort[] GreenMap16 {
            get => greenMap;
            set {
                // check the map
                if ((value == null) || (value.Length != 65536))
                    throw new ArgumentException("A map should be array with 65536 value.");

                greenMap = value;
            }
        }

        public ushort[] BlueMap16 {
            get => blueMap;
            set {
                // check the map
                if ((value == null) || (value.Length != 65536))
                    throw new ArgumentException("A map should be array with 65536 value.");

                blueMap = value;
            }
        }

        public ushort[] GrayMap16 {
            get => grayMap;
            set {
                // check the map
                if ((value == null) || (value.Length != 65536))
                    throw new ArgumentException("A map should be array with 65536 value.");

                grayMap = value;
            }
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="ColorRemapping"/> class.
        /// </summary>
        ///
        /// <remarks>Initializes the filter without any remapping. All
        /// pixel values are mapped to the same values.</remarks>
        ///
        public ColorRemappingGeneral(ushort[] redMap, ushort[] greenMap, ushort[] blueMap) {
            FormatTranslations[System.Drawing.Imaging.PixelFormat.Format48bppRgb] = System.Drawing.Imaging.PixelFormat.Format48bppRgb;
            RedMap16 = redMap;
            GreenMap16 = greenMap;
            BlueMap16 = blueMap;
        }

        public ColorRemappingGeneral(ushort[] grayMap) {
            FormatTranslations[System.Drawing.Imaging.PixelFormat.Format16bppGrayScale] = System.Drawing.Imaging.PixelFormat.Format16bppGrayScale;
            GrayMap16 = grayMap;
        }

        /// <summary>
        /// Process the filter on the specified image.
        /// </summary>
        ///
        /// <param name="image">Source image data.</param>
        /// <param name="rect">Image rectangle for processing by the filter.</param>
        ///
        protected override unsafe void ProcessFilter(UnmanagedImage image, Rectangle rect) {
            // processing start and stop X,Y positions
            int stopX = rect.Width;
            int stopY = rect.Height;
            bool isGray = image.PixelFormat == System.Drawing.Imaging.PixelFormat.Format16bppGrayScale;
            int channels = isGray ? 1 : 3;
            int rowStep = stopX * channels; // assumes tight packing like original implementation

#if DEBUG
            // snapshot original data for reference verification
            int total = rowStep * stopY;
            ushort[] original = ArrayPool<ushort>.Shared.Rent(total);
            ushort* dbgPtr = (ushort*)image.ImageData.ToPointer();
            for (int y = 0; y < stopY; y++) {
                int dstOffset = y * rowStep;
                ushort* srcRow = dbgPtr + y * rowStep;
                for (int i = 0; i < rowStep; i++) {
                    original[dstOffset + i] = srcRow[i];
                }
            }
#endif

            // Optimized path: run per-row mapping in parallel to keep the inner loop branch-free
            // while retaining the original algorithm for debug verification below.
            ushort* basePtr = (ushort*)image.ImageData.ToPointer();

            if (isGray) {
                Parallel.For(0, stopY, y => {
                    ushort* row = basePtr + y * rowStep;
                    for (int x = 0; x < stopX; x++) {
                        row[x] = grayMap[row[x]];
                    }
                });
            } else {
                Parallel.For(0, stopY, y => {
                    ushort* row = basePtr + y * rowStep;
                    for (int x = 0; x < stopX; x++, row += 3) {
                        row[RGB.R] = redMap[row[RGB.R]];
                        row[RGB.G] = greenMap[row[RGB.G]];
                        row[RGB.B] = blueMap[row[RGB.B]];
                    }
                });
            }

#if DEBUG
            // reference verification using the original sequential algorithm (unchanged)
            VerifyReference(basePtr, stopX, stopY, isGray, rowStep, original);
            ArrayPool<ushort>.Shared.Return(original);
#endif
        }

#if DEBUG
        private unsafe void VerifyReference(
            ushort* basePtr,
            int stopX,
            int stopY,
            bool isGray,
            int rowStep,
            ushort[] original) {
            if (isGray) {
                // original sequential gray mapping
                for (int y = 0; y < stopY; y++) {
                    ushort* row = basePtr + y * rowStep;
                    int srcOffset = y * rowStep;
                    for (int x = 0; x < stopX; x++) {
                        ushort expected = grayMap[original[srcOffset + x]];
                        Debug.Assert(row[x] == expected, $"Gray remap mismatch at ({x},{y})");
                    }
                }
            } else {
                // original sequential RGB mapping
                for (int y = 0; y < stopY; y++) {
                    ushort* row = basePtr + y * rowStep;
                    int srcOffset = y * rowStep;
                    for (int x = 0; x < stopX; x++, row += 3) {
                        int o = srcOffset + x * 3;
                        ushort expR = redMap[original[o + RGB.R]];
                        ushort expG = greenMap[original[o + RGB.G]];
                        ushort expB = blueMap[original[o + RGB.B]];
                        Debug.Assert(row[RGB.R] == expR && row[RGB.G] == expG && row[RGB.B] == expB,
                            $"RGB remap mismatch at ({x},{y})");
                    }
                }
            }
        }
#endif
    }
}
