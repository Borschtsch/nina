#region "copyright"
/*
    Local copy of Accord/AForge ResizeBicubic to allow buffer reuse and tuning.
*/
#endregion

using System;
using System.Drawing;
using System.Drawing.Imaging;

namespace NINA.Image.ImageAnalysis.Filters {
    public class ResizeBicubic {
        private readonly int newWidth;
        private readonly int newHeight;

        public ResizeBicubic(int width, int height) {
            newWidth = width;
            newHeight = height;
        }

        public Bitmap Apply(Bitmap source) {
            if (source == null) throw new ArgumentNullException(nameof(source));
            PixelFormat dstFormat = source.PixelFormat;
            var dest = new Bitmap(newWidth, newHeight, dstFormat);
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

        private unsafe void Process(BitmapData sourceData, BitmapData destinationData) {
            int width = sourceData.Width;
            int height = sourceData.Height;
            int pixelSize = System.Drawing.Image.GetPixelFormatSize(sourceData.PixelFormat) / 8;
            if (pixelSize == 2) {
                // Preserve legacy behavior: this path never handled 16bpp grayscale; fail fast to avoid silent corruption.
                throw new NotSupportedException("16bpp grayscale resize is not supported in this path.");
            }
            int srcStride = sourceData.Stride;
            int dstOffset = destinationData.Stride - newWidth * pixelSize;
            double xFactor = (double)width / newWidth;
            double yFactor = (double)height / newHeight;

            byte* src = (byte*)sourceData.Scan0.ToPointer();
            byte* dst = (byte*)destinationData.Scan0.ToPointer();

            int ymax = height - 1;
            int xmax = width - 1;

            double[] k1Row = new double[4]; // Y weights reused across X for this row

            for (int y = 0; y < newHeight; y++) {
                double oy = y * yFactor - 0.5;
                int oy1 = (int)oy;
                double dy = oy - oy1;
                for (int n = -1; n < 3; n++) {
                    k1Row[n + 1] = LocalInterpolation.BiCubicKernel(dy - n);
                }

                for (int x = 0; x < newWidth; x++, dst += pixelSize) {
                    double ox = x * xFactor - 0.5;
                    int ox1 = (int)ox;
                    double dx = ox - ox1;

                    double c0 = 0, c1 = 0, c2 = 0, c3 = 0;

                    for (int n = -1; n < 3; n++) {
                        double k1 = k1Row[n + 1]; // reuse precomputed Y weight for this row
                        int oy2 = oy1 + n;
                        if (oy2 < 0) oy2 = 0;
                        if (oy2 > ymax) oy2 = ymax;

                        for (int m = -1; m < 3; m++) {
                            double k2 = k1 * LocalInterpolation.BiCubicKernel(m - dx);
                            int ox2 = ox1 + m;
                            if (ox2 < 0) ox2 = 0;
                            if (ox2 > xmax) ox2 = xmax;

                            byte* p = src + oy2 * srcStride + ox2 * pixelSize;

                            c0 += k2 * p[0];
                            if (pixelSize > 1) {
                                c1 += k2 * p[1];
                                c2 += k2 * p[2];
                            }
                            if (pixelSize > 3) {
                                c3 += k2 * p[3];
                            }
                        }
                    }

                    if (pixelSize == 1) {
                        dst[0] = (byte)Math.Max(0, Math.Min(255, c0));
                    } else if (pixelSize == 3) {
                        dst[0] = (byte)Math.Max(0, Math.Min(255, c0));
                        dst[1] = (byte)Math.Max(0, Math.Min(255, c1));
                        dst[2] = (byte)Math.Max(0, Math.Min(255, c2));
                    } else if (pixelSize == 4) {
                        dst[0] = (byte)Math.Max(0, Math.Min(255, c0));
                        dst[1] = (byte)Math.Max(0, Math.Min(255, c1));
                        dst[2] = (byte)Math.Max(0, Math.Min(255, c2));
                        dst[3] = (byte)Math.Max(0, Math.Min(255, c3));
                    }
                }
                dst += dstOffset;
            }
        }
    }
}
