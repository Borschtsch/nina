#region "copyright"
/*
    Local copy of Accord/AForge Grayscale to allow buffer reuse and tuning.
*/
#endregion

using System;
using System.Drawing;
using System.Drawing.Imaging;
using NINA.Image.ImageAnalysis.Filters;

namespace NINA.Image.ImageAnalysis.Filters {
    public class Grayscale {
        public static class CommonAlgorithms {
            public static readonly Grayscale BT709 = new Grayscale(0.2125, 0.7154, 0.0721);
            public static readonly Grayscale RMY = new Grayscale(0.5000, 0.4190, 0.0810);
            public static readonly Grayscale Y = new Grayscale(0.2990, 0.5870, 0.1140);
        }

        public readonly double RedCoefficient;
        public readonly double GreenCoefficient;
        public readonly double BlueCoefficient;

        public Grayscale(double cr, double cg, double cb) {
            RedCoefficient = cr;
            GreenCoefficient = cg;
            BlueCoefficient = cb;
        }

        public Bitmap Apply(Bitmap source) {
            if (source == null) throw new ArgumentNullException(nameof(source));

            PixelFormat srcPixelFormat = source.PixelFormat;
            PixelFormat dstPixelFormat =
                (srcPixelFormat == PixelFormat.Format48bppRgb || srcPixelFormat == PixelFormat.Format64bppArgb)
                    ? PixelFormat.Format16bppGrayScale
                    : PixelFormat.Format8bppIndexed;

            var dest = new Bitmap(source.Width, source.Height, dstPixelFormat);
            if (dstPixelFormat == PixelFormat.Format8bppIndexed) {
                dest.Palette = CreateGrayPalette();
            }

            var srcData = source.LockBits(new Rectangle(0, 0, source.Width, source.Height), ImageLockMode.ReadOnly, srcPixelFormat);
            var dstData = dest.LockBits(new Rectangle(0, 0, dest.Width, dest.Height), ImageLockMode.WriteOnly, dstPixelFormat);

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
            PixelFormat srcPixelFormat = sourceData.PixelFormat;

            if (srcPixelFormat == PixelFormat.Format24bppRgb ||
                srcPixelFormat == PixelFormat.Format32bppRgb ||
                srcPixelFormat == PixelFormat.Format32bppArgb) {

                int pixelSize = (srcPixelFormat == PixelFormat.Format24bppRgb) ? 3 : 4;
                int srcOffset = sourceData.Stride - width * pixelSize;
                int dstOffset = destinationData.Stride - width;

                int rc = (int)(0x10000 * RedCoefficient);
                int gc = (int)(0x10000 * GreenCoefficient);
                int bc = (int)(0x10000 * BlueCoefficient);

                while (rc + gc + bc < 0x10000) {
                    bc++;
                }

                byte* src = (byte*)sourceData.Scan0.ToPointer();
                byte* dst = (byte*)destinationData.Scan0.ToPointer();

                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++, src += pixelSize, dst++) {
                        *dst = (byte)((rc * src[RgbIdx.R] + gc * src[RgbIdx.G] + bc * src[RgbIdx.B]) >> 16);
                    }
                    src += srcOffset;
                    dst += dstOffset;
                }
            } else {
                int pixelSize = (srcPixelFormat == PixelFormat.Format48bppRgb) ? 3 : 4;
                byte* srcBase = (byte*)sourceData.Scan0.ToPointer();
                byte* dstBase = (byte*)destinationData.Scan0.ToPointer();
                int srcStride = sourceData.Stride;
                int dstStride = destinationData.Stride;

                for (int y = 0; y < height; y++) {
                    ushort* src = (ushort*)(srcBase + y * srcStride);
                    ushort* dst = (ushort*)(dstBase + y * dstStride);

                    for (int x = 0; x < width; x++, src += pixelSize, dst++) {
                        *dst = (ushort)(RedCoefficient * src[RgbIdx.R] + GreenCoefficient * src[RgbIdx.G] + BlueCoefficient * src[RgbIdx.B]);
                    }
                }
            }
        }

        private static ColorPalette CreateGrayPalette() {
            using (var bmp = new Bitmap(1, 1, PixelFormat.Format8bppIndexed)) {
                var palette = bmp.Palette;
                for (int i = 0; i < 256; i++) {
                    palette.Entries[i] = Color.FromArgb(i, i, i);
                }
                return palette;
            }
        }
    }
}
