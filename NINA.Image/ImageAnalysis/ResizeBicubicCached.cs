using System;
using System.Collections.Concurrent;
using System.Threading.Tasks;
using System.Drawing;
using System.Drawing.Imaging;

namespace NINA.Image.ImageAnalysis.Filters {
    internal static class ResizeBicubicCached {
        private sealed class Weights {
            public readonly int[] XIndex;
            public readonly int[] YIndex;
            public readonly double[] XWeight;
            public readonly double[] YWeight;
            public Weights(int[] xi, int[] yi, double[] xw, double[] yw) { XIndex = xi; YIndex = yi; XWeight = xw; YWeight = yw; }
        }

        private static readonly ConcurrentDictionary<(int w, int h, int tw, int th), Weights> _cache = new();

        public static Bitmap Resize(Bitmap source, int targetWidth, int targetHeight) {
            if (source == null) throw new ArgumentNullException(nameof(source));

            PixelFormat dstFormat = source.PixelFormat;
            var dest = new Bitmap(targetWidth, targetHeight, dstFormat);
            if (dstFormat == PixelFormat.Format8bppIndexed) {
                dest.Palette = source.Palette;
            }

            var srcData = source.LockBits(new Rectangle(0, 0, source.Width, source.Height), ImageLockMode.ReadOnly, source.PixelFormat);
            var dstData = dest.LockBits(new Rectangle(0, 0, dest.Width, dest.Height), ImageLockMode.WriteOnly, dstFormat);

            try {
                Process(srcData, dstData);
            } finally {
                source.UnlockBits(srcData);
                dest.UnlockBits(dstData);
            }

            return dest;
        }

        private static Weights GetWeights(int width, int height, int targetWidth, int targetHeight) {
            var key = (width, height, targetWidth, targetHeight);
            if (_cache.TryGetValue(key, out var w)) return w;

            int[] xIndex = new int[targetWidth * 4];
            int[] yIndex = new int[targetHeight * 4];
            double[] xWeight = new double[targetWidth * 4];
            double[] yWeight = new double[targetHeight * 4];

            double scaleX = (double)width / targetWidth;
            double scaleY = (double)height / targetHeight;

            for (int x = 0; x < targetWidth; x++) {
                double gx = x * scaleX - 0.5;
                int gxi = (int)Math.Floor(gx);
                double dx = gx - gxi;
                int baseIdx = x * 4;
                xIndex[baseIdx + 0] = Clamp(gxi - 1, 0, width - 1);
                xIndex[baseIdx + 1] = Clamp(gxi, 0, width - 1);
                xIndex[baseIdx + 2] = Clamp(gxi + 1, 0, width - 1);
                xIndex[baseIdx + 3] = Clamp(gxi + 2, 0, width - 1);

                xWeight[baseIdx + 0] = LocalInterpolation.BiCubicKernel(1 + dx);
                xWeight[baseIdx + 1] = LocalInterpolation.BiCubicKernel(dx);
                xWeight[baseIdx + 2] = LocalInterpolation.BiCubicKernel(1 - dx);
                xWeight[baseIdx + 3] = LocalInterpolation.BiCubicKernel(2 - dx);
            }

            for (int y = 0; y < targetHeight; y++) {
                double gy = y * scaleY - 0.5;
                int gyi = (int)Math.Floor(gy);
                double dy = gy - gyi;
                int baseIdx = y * 4;
                yIndex[baseIdx + 0] = Clamp(gyi - 1, 0, height - 1);
                yIndex[baseIdx + 1] = Clamp(gyi, 0, height - 1);
                yIndex[baseIdx + 2] = Clamp(gyi + 1, 0, height - 1);
                yIndex[baseIdx + 3] = Clamp(gyi + 2, 0, height - 1);

                yWeight[baseIdx + 0] = LocalInterpolation.BiCubicKernel(1 + dy);
                yWeight[baseIdx + 1] = LocalInterpolation.BiCubicKernel(dy);
                yWeight[baseIdx + 2] = LocalInterpolation.BiCubicKernel(1 - dy);
                yWeight[baseIdx + 3] = LocalInterpolation.BiCubicKernel(2 - dy);
            }

            var weights = new Weights(xIndex, yIndex, xWeight, yWeight);
            _cache[key] = weights;
            return weights;
        }

        private static int Clamp(int v, int min, int max) => v < min ? min : (v > max ? max : v);

        private static unsafe void Process(BitmapData sourceData, BitmapData destinationData) {
            int width = sourceData.Width;
            int height = sourceData.Height;
            int pixelSize = System.Drawing.Image.GetPixelFormatSize(sourceData.PixelFormat) / 8;
            int srcStride = sourceData.Stride;
            int dstOffset = destinationData.Stride - destinationData.Width * pixelSize;

            var weights = GetWeights(width, height, destinationData.Width, destinationData.Height);
            int[] xIndex = weights.XIndex; int[] yIndex = weights.YIndex; double[] xWeight = weights.XWeight; double[] yWeight = weights.YWeight;

            byte* src = (byte*)sourceData.Scan0.ToPointer();
            byte* dst = (byte*)destinationData.Scan0.ToPointer();

            Parallel.For(0, destinationData.Height, y => {
                byte* dstRow = dst + y * destinationData.Stride;
                int yIdx = y * 4;
                double wy0 = yWeight[yIdx + 0];
                double wy1 = yWeight[yIdx + 1];
                double wy2 = yWeight[yIdx + 2];
                double wy3 = yWeight[yIdx + 3];
                byte* row0 = src + yIndex[yIdx + 0] * srcStride;
                byte* row1 = src + yIndex[yIdx + 1] * srcStride;
                byte* row2 = src + yIndex[yIdx + 2] * srcStride;
                byte* row3 = src + yIndex[yIdx + 3] * srcStride;

                if (pixelSize == 1) {
                    for (int x = 0; x < destinationData.Width; x++, dstRow += 1) {
                        int xIdx = x * 4;
                        double wx0 = xWeight[xIdx + 0];
                        double wx1 = xWeight[xIdx + 1];
                        double wx2 = xWeight[xIdx + 2];
                        double wx3 = xWeight[xIdx + 3];

                        // Unrolled 4x4 kernel; identical math to original bicubic path.
                        double c0 =
                            wy0 * (wx0 * row0[xIndex[xIdx + 0]] + wx1 * row0[xIndex[xIdx + 1]] + wx2 * row0[xIndex[xIdx + 2]] + wx3 * row0[xIndex[xIdx + 3]]) +
                            wy1 * (wx0 * row1[xIndex[xIdx + 0]] + wx1 * row1[xIndex[xIdx + 1]] + wx2 * row1[xIndex[xIdx + 2]] + wx3 * row1[xIndex[xIdx + 3]]) +
                            wy2 * (wx0 * row2[xIndex[xIdx + 0]] + wx1 * row2[xIndex[xIdx + 1]] + wx2 * row2[xIndex[xIdx + 2]] + wx3 * row2[xIndex[xIdx + 3]]) +
                            wy3 * (wx0 * row3[xIndex[xIdx + 0]] + wx1 * row3[xIndex[xIdx + 1]] + wx2 * row3[xIndex[xIdx + 2]] + wx3 * row3[xIndex[xIdx + 3]]);

                        int v0 = (int)c0;
                        if (v0 < 0) v0 = 0;
                        else if (v0 > 255) v0 = 255;
                        dstRow[0] = (byte)v0;
                    }
                } else if (pixelSize == 3) {
                    for (int x = 0; x < destinationData.Width; x++, dstRow += 3) {
                        int xIdx = x * 4;
                        double wx0 = xWeight[xIdx + 0];
                        double wx1 = xWeight[xIdx + 1];
                        double wx2 = xWeight[xIdx + 2];
                        double wx3 = xWeight[xIdx + 3];

                        byte* p00 = row0 + xIndex[xIdx + 0] * 3;
                        byte* p01 = row0 + xIndex[xIdx + 1] * 3;
                        byte* p02 = row0 + xIndex[xIdx + 2] * 3;
                        byte* p03 = row0 + xIndex[xIdx + 3] * 3;

                        byte* p10 = row1 + xIndex[xIdx + 0] * 3;
                        byte* p11 = row1 + xIndex[xIdx + 1] * 3;
                        byte* p12 = row1 + xIndex[xIdx + 2] * 3;
                        byte* p13 = row1 + xIndex[xIdx + 3] * 3;

                        byte* p20 = row2 + xIndex[xIdx + 0] * 3;
                        byte* p21 = row2 + xIndex[xIdx + 1] * 3;
                        byte* p22 = row2 + xIndex[xIdx + 2] * 3;
                        byte* p23 = row2 + xIndex[xIdx + 3] * 3;

                        byte* p30 = row3 + xIndex[xIdx + 0] * 3;
                        byte* p31 = row3 + xIndex[xIdx + 1] * 3;
                        byte* p32 = row3 + xIndex[xIdx + 2] * 3;
                        byte* p33 = row3 + xIndex[xIdx + 3] * 3;

                        // Unrolled channel accumulation for 24bpp.
                        double c0 =
                            wy0 * (wx0 * p00[0] + wx1 * p01[0] + wx2 * p02[0] + wx3 * p03[0]) +
                            wy1 * (wx0 * p10[0] + wx1 * p11[0] + wx2 * p12[0] + wx3 * p13[0]) +
                            wy2 * (wx0 * p20[0] + wx1 * p21[0] + wx2 * p22[0] + wx3 * p23[0]) +
                            wy3 * (wx0 * p30[0] + wx1 * p31[0] + wx2 * p32[0] + wx3 * p33[0]);

                        double c1 =
                            wy0 * (wx0 * p00[1] + wx1 * p01[1] + wx2 * p02[1] + wx3 * p03[1]) +
                            wy1 * (wx0 * p10[1] + wx1 * p11[1] + wx2 * p12[1] + wx3 * p13[1]) +
                            wy2 * (wx0 * p20[1] + wx1 * p21[1] + wx2 * p22[1] + wx3 * p23[1]) +
                            wy3 * (wx0 * p30[1] + wx1 * p31[1] + wx2 * p32[1] + wx3 * p33[1]);

                        double c2 =
                            wy0 * (wx0 * p00[2] + wx1 * p01[2] + wx2 * p02[2] + wx3 * p03[2]) +
                            wy1 * (wx0 * p10[2] + wx1 * p11[2] + wx2 * p12[2] + wx3 * p13[2]) +
                            wy2 * (wx0 * p20[2] + wx1 * p21[2] + wx2 * p22[2] + wx3 * p23[2]) +
                            wy3 * (wx0 * p30[2] + wx1 * p31[2] + wx2 * p32[2] + wx3 * p33[2]);

                        int v0 = (int)c0; if (v0 < 0) v0 = 0; else if (v0 > 255) v0 = 255;
                        int v1 = (int)c1; if (v1 < 0) v1 = 0; else if (v1 > 255) v1 = 255;
                        int v2 = (int)c2; if (v2 < 0) v2 = 0; else if (v2 > 255) v2 = 255;

                        dstRow[0] = (byte)v0;
                        dstRow[1] = (byte)v1;
                        dstRow[2] = (byte)v2;
                    }
                } else if (pixelSize == 4) {
                    for (int x = 0; x < destinationData.Width; x++, dstRow += 4) {
                        int xIdx = x * 4;
                        double wx0 = xWeight[xIdx + 0];
                        double wx1 = xWeight[xIdx + 1];
                        double wx2 = xWeight[xIdx + 2];
                        double wx3 = xWeight[xIdx + 3];

                        byte* p00 = row0 + xIndex[xIdx + 0] * 4;
                        byte* p01 = row0 + xIndex[xIdx + 1] * 4;
                        byte* p02 = row0 + xIndex[xIdx + 2] * 4;
                        byte* p03 = row0 + xIndex[xIdx + 3] * 4;

                        byte* p10 = row1 + xIndex[xIdx + 0] * 4;
                        byte* p11 = row1 + xIndex[xIdx + 1] * 4;
                        byte* p12 = row1 + xIndex[xIdx + 2] * 4;
                        byte* p13 = row1 + xIndex[xIdx + 3] * 4;

                        byte* p20 = row2 + xIndex[xIdx + 0] * 4;
                        byte* p21 = row2 + xIndex[xIdx + 1] * 4;
                        byte* p22 = row2 + xIndex[xIdx + 2] * 4;
                        byte* p23 = row2 + xIndex[xIdx + 3] * 4;

                        byte* p30 = row3 + xIndex[xIdx + 0] * 4;
                        byte* p31 = row3 + xIndex[xIdx + 1] * 4;
                        byte* p32 = row3 + xIndex[xIdx + 2] * 4;
                        byte* p33 = row3 + xIndex[xIdx + 3] * 4;

                        // Unrolled channel accumulation for 32bpp.
                        double c0 =
                            wy0 * (wx0 * p00[0] + wx1 * p01[0] + wx2 * p02[0] + wx3 * p03[0]) +
                            wy1 * (wx0 * p10[0] + wx1 * p11[0] + wx2 * p12[0] + wx3 * p13[0]) +
                            wy2 * (wx0 * p20[0] + wx1 * p21[0] + wx2 * p22[0] + wx3 * p23[0]) +
                            wy3 * (wx0 * p30[0] + wx1 * p31[0] + wx2 * p32[0] + wx3 * p33[0]);

                        double c1 =
                            wy0 * (wx0 * p00[1] + wx1 * p01[1] + wx2 * p02[1] + wx3 * p03[1]) +
                            wy1 * (wx0 * p10[1] + wx1 * p11[1] + wx2 * p12[1] + wx3 * p13[1]) +
                            wy2 * (wx0 * p20[1] + wx1 * p21[1] + wx2 * p22[1] + wx3 * p23[1]) +
                            wy3 * (wx0 * p30[1] + wx1 * p31[1] + wx2 * p32[1] + wx3 * p33[1]);

                        double c2 =
                            wy0 * (wx0 * p00[2] + wx1 * p01[2] + wx2 * p02[2] + wx3 * p03[2]) +
                            wy1 * (wx0 * p10[2] + wx1 * p11[2] + wx2 * p12[2] + wx3 * p13[2]) +
                            wy2 * (wx0 * p20[2] + wx1 * p21[2] + wx2 * p22[2] + wx3 * p23[2]) +
                            wy3 * (wx0 * p30[2] + wx1 * p31[2] + wx2 * p32[2] + wx3 * p33[2]);

                        double c3 =
                            wy0 * (wx0 * p00[3] + wx1 * p01[3] + wx2 * p02[3] + wx3 * p03[3]) +
                            wy1 * (wx0 * p10[3] + wx1 * p11[3] + wx2 * p12[3] + wx3 * p13[3]) +
                            wy2 * (wx0 * p20[3] + wx1 * p21[3] + wx2 * p22[3] + wx3 * p23[3]) +
                            wy3 * (wx0 * p30[3] + wx1 * p31[3] + wx2 * p32[3] + wx3 * p33[3]);

                        int v0 = (int)c0; if (v0 < 0) v0 = 0; else if (v0 > 255) v0 = 255;
                        int v1 = (int)c1; if (v1 < 0) v1 = 0; else if (v1 > 255) v1 = 255;
                        int v2 = (int)c2; if (v2 < 0) v2 = 0; else if (v2 > 255) v2 = 255;
                        int v3 = (int)c3; if (v3 < 0) v3 = 0; else if (v3 > 255) v3 = 255;

                        dstRow[0] = (byte)v0;
                        dstRow[1] = (byte)v1;
                        dstRow[2] = (byte)v2;
                        dstRow[3] = (byte)v3;
                    }
                } else {
                    for (int x = 0; x < destinationData.Width; x++, dstRow += pixelSize) {
                        int xIdx = x * 4;
                        double wx0 = xWeight[xIdx + 0];
                        double wx1 = xWeight[xIdx + 1];
                        double wx2 = xWeight[xIdx + 2];
                        double wx3 = xWeight[xIdx + 3];

                        byte* base00 = row0 + xIndex[xIdx + 0] * pixelSize;

                        for (int ch = 0; ch < pixelSize; ch++) {
                            double v =
                                wy0 * (wx0 * base00[ch] + wx1 * base00[ch + (xIndex[xIdx + 1] - xIndex[xIdx + 0]) * pixelSize] + wx2 * base00[ch + (xIndex[xIdx + 2] - xIndex[xIdx + 0]) * pixelSize] + wx3 * base00[ch + (xIndex[xIdx + 3] - xIndex[xIdx + 0]) * pixelSize]) +
                                wy1 * (wx0 * (row1 + xIndex[xIdx + 0] * pixelSize)[ch] + wx1 * (row1 + xIndex[xIdx + 1] * pixelSize)[ch] + wx2 * (row1 + xIndex[xIdx + 2] * pixelSize)[ch] + wx3 * (row1 + xIndex[xIdx + 3] * pixelSize)[ch]) +
                                wy2 * (wx0 * (row2 + xIndex[xIdx + 0] * pixelSize)[ch] + wx1 * (row2 + xIndex[xIdx + 1] * pixelSize)[ch] + wx2 * (row2 + xIndex[xIdx + 2] * pixelSize)[ch] + wx3 * (row2 + xIndex[xIdx + 3] * pixelSize)[ch]) +
                                wy3 * (wx0 * (row3 + xIndex[xIdx + 0] * pixelSize)[ch] + wx1 * (row3 + xIndex[xIdx + 1] * pixelSize)[ch] + wx2 * (row3 + xIndex[xIdx + 2] * pixelSize)[ch] + wx3 * (row3 + xIndex[xIdx + 3] * pixelSize)[ch]);
                            int iv = (int)v;
                            if (iv < 0) iv = 0;
                            else if (iv > 255) iv = 255;
                            dstRow[ch] = (byte)iv;
                        }
                    }
                }
            });
        }
    }
}
