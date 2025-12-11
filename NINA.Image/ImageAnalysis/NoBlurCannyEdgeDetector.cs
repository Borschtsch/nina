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
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Buffers;
using System.Threading.Tasks;
using Accord.Imaging;
using Accord.Imaging.Filters;

namespace NINA.Image.ImageAnalysis {

    public class NoBlurCannyEdgeDetector : BaseUsingCopyPartialFilter {
        private byte lowThreshold = 20;
        private byte highThreshold = 100;

        // private format translation dictionary
        private Dictionary<PixelFormat, PixelFormat> formatTranslations = new Dictionary<PixelFormat, PixelFormat>();

        /// <summary>
        /// Format translations dictionary.
        /// </summary>
        public override Dictionary<PixelFormat, PixelFormat> FormatTranslations => formatTranslations;

        /// <summary>
        /// Low threshold.
        /// </summary>
        ///
        /// <remarks><para>Low threshold value used for hysteresis
        /// (see  <a href="http://www.pages.drexel.edu/~weg22/can_tut.html">tutorial</a>
        /// for more information).</para>
        ///
        /// <para>Default value is set to <b>20</b>.</para>
        /// </remarks>
        ///
        public byte LowThreshold {
            get => lowThreshold;
            set => lowThreshold = value;
        }

        /// <summary>
        /// High threshold.
        /// </summary>
        ///
        /// <remarks><para>High threshold value used for hysteresis
        /// (see  <a href="http://www.pages.drexel.edu/~weg22/can_tut.html">tutorial</a>
        /// for more information).</para>
        ///
        /// <para>Default value is set to <b>100</b>.</para>
        /// </remarks>
        ///
        public byte HighThreshold {
            get => highThreshold;
            set => highThreshold = value;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="CannyEdgeDetector"/> class.
        /// </summary>
        ///
        public NoBlurCannyEdgeDetector() {
            // initialize format translation dictionary
            formatTranslations[PixelFormat.Format8bppIndexed] = PixelFormat.Format8bppIndexed;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="CannyEdgeDetector"/> class.
        /// </summary>
        ///
        /// <param name="lowThreshold">Low threshold.</param>
        /// <param name="highThreshold">High threshold.</param>
        ///
        public NoBlurCannyEdgeDetector(byte lowThreshold, byte highThreshold) : this() {
            this.lowThreshold = lowThreshold;
            this.highThreshold = highThreshold;
        }

        /// <summary>
        /// Process the filter on the specified image.
        /// </summary>
        ///
        /// <param name="source">Source image data.</param>
        /// <param name="destination">Destination image data.</param>
        /// <param name="rect">Image rectangle for processing by the filter.</param>
        ///
        protected override unsafe void ProcessFilter(UnmanagedImage source, UnmanagedImage destination, Rectangle rect) {
            // processing start and stop X,Y positions
            int startX = rect.Left + 1;
            int startY = rect.Top + 1;
            int stopX = startX + rect.Width - 2;
            int stopY = startY + rect.Height - 2;

            int width = rect.Width - 2;
            int height = rect.Height - 2;

            int dstStride = destination.Stride;
            int srcStride = source.Stride;

            int dstOffset = dstStride - rect.Width + 2;
            int srcOffset = srcStride - rect.Width + 2;

            // pixel's value and gradients
            int gx, gy;
            //
            double orientation, toAngle = 180.0 / System.Math.PI;
            float leftPixel = 0, rightPixel = 0;

            // orientation array (pooled to avoid per-frame allocations)
            byte[] orients = ArrayPool<byte>.Shared.Rent(width * height);
            // gradients array
            float[,] gradients = new float[source.Width, source.Height];
            float maxGradient = float.NegativeInfinity;

            // do the job
            byte* src = (byte*)source.ImageData.ToPointer();
            // allign pointer
            src += srcStride * startY + startX;

            // STEP 1 - calculate magnitude and edge orientation (parallelized per row, pooled buffers)
            object maxLock = new object();

            // for each line
            Parallel.For(startY, stopY, y => {
                byte* localSrc = src + (y - startY) * srcStride;
                int localPBase = (y - startY) * width;
                float localMax = float.NegativeInfinity;

                for (int x = startX; x < stopX; x++, localSrc++) {
                    int idx = localPBase + (x - startX);

                    gx = localSrc[-srcStride + 1] + localSrc[srcStride + 1]
                       - localSrc[-srcStride - 1] - localSrc[srcStride - 1]
                       + 2 * (localSrc[1] - localSrc[-1]);

                    gy = localSrc[-srcStride - 1] + localSrc[-srcStride + 1]
                       - localSrc[srcStride - 1] - localSrc[srcStride + 1]
                       + 2 * (localSrc[-srcStride] - localSrc[srcStride]);

                    // get gradient value
                    float g = (float)Math.Sqrt(gx * gx + gy * gy);
                    gradients[x, y] = g;
                    if (g > localMax)
                        localMax = g;

                    // --- get orientation
                    if (gx == 0) {
                        // can not divide by zero
                        orientation = (gy == 0) ? 0 : 90;
                    } else {
                        double div = (double)gy / gx;

                        // handle angles of the 2nd and 4th quads
                        if (div < 0) {
                            orientation = 180 - System.Math.Atan(-div) * toAngle;
                        }
                        // handle angles of the 1st and 3rd quads
                        else {
                            orientation = System.Math.Atan(div) * toAngle;
                        }

                        // get closest angle from 0, 45, 90, 135 set
                        if (orientation < 22.5)
                            orientation = 0;
                        else if (orientation < 67.5)
                            orientation = 45;
                        else if (orientation < 112.5)
                            orientation = 90;
                        else if (orientation < 157.5)
                            orientation = 135;
                        else orientation = 0;
                    }

                    // save orientation
                    orients[idx] = (byte)orientation;
                }

                lock (maxLock) {
                    if (localMax > maxGradient) {
                        maxGradient = localMax;
            }
                }
            });

            // STEP 2 - suppress non maximums (parallel per row; uses orientations computed above)
            byte* dst = (byte*)destination.ImageData.ToPointer();
            // allign pointer
            dst += dstStride * startY + startX;

            // for each line
            Parallel.For(startY, stopY, y => {
                byte* rowDst = dst + (y - startY) * dstStride;
                int pBase = (y - startY) * width;

                for (int x = startX; x < stopX; x++, rowDst++) {
                    int idx = pBase + (x - startX);
                    // get two adjacent pixels
                    switch (orients[idx]) {
                        case 0:
                            leftPixel = gradients[x - 1, y];
                            rightPixel = gradients[x + 1, y];
                            break;

                        case 45:
                            leftPixel = gradients[x - 1, y + 1];
                            rightPixel = gradients[x + 1, y - 1];
                            break;

                        case 90:
                            leftPixel = gradients[x, y + 1];
                            rightPixel = gradients[x, y - 1];
                            break;

                        default: // 135
                            leftPixel = gradients[x + 1, y + 1];
                            rightPixel = gradients[x - 1, y - 1];
                            break;
                    }
                    // compare current pixels value with adjacent pixels
                    if ((gradients[x, y] < leftPixel) || (gradients[x, y] < rightPixel)) {
                        *rowDst = 0;
                    } else {
                        *rowDst = (byte)(gradients[x, y] / maxGradient * 255);
                    }
                }
            });

            // STEP 3 - hysteresis
            dst = (byte*)destination.ImageData.ToPointer();
            // allign pointer
            dst += dstStride * startY + startX;

            // for each line
            for (int y = startY; y < stopY; y++) {
                // for each pixel
                for (int x = startX; x < stopX; x++, dst++) {
                    if (*dst < highThreshold) {
                        if (*dst < lowThreshold) {
                            // non edge
                            *dst = 0;
                        } else {
                            // check 8 neighboring pixels
                            if ((dst[-1] < highThreshold) &&
                                (dst[1] < highThreshold) &&
                                (dst[-dstStride - 1] < highThreshold) &&
                                (dst[-dstStride] < highThreshold) &&
                                (dst[-dstStride + 1] < highThreshold) &&
                                (dst[dstStride - 1] < highThreshold) &&
                                (dst[dstStride] < highThreshold) &&
                                (dst[dstStride + 1] < highThreshold)) {
                                *dst = 0;
                            }
                        }
                    }
                }
                dst += dstOffset;
            }

            // STEP 4 - draw black rectangle to remove those pixels, which were not processed
            // (this needs to be done for those cases, when filter is applied "in place" -
            //  source image is modified instead of creating new copy)
            Drawing.Rectangle(destination, rect, Color.Black);

            // release blurred image
            source.Dispose();
            // Return rented buffers
            ArrayPool<byte>.Shared.Return(orients);
        }
        }
    }
