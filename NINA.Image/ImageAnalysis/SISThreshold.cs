#region "copyright"
/*
    Local copy of AForge SISThreshold (Adaptive Binarization) to allow tuning and buffer reuse.
*/
#endregion

using System;
using System.Drawing;
using System.Drawing.Imaging;

namespace NINA.Image.ImageAnalysis.Filters {
    public class SISThreshold {
        private readonly Threshold threshold = new Threshold();

        public int ThresholdValue => threshold.ThresholdValue;

        public void ApplyInPlace(Bitmap image) {
            if (image == null) throw new ArgumentNullException(nameof(image));
            if (image.PixelFormat != PixelFormat.Format8bppIndexed) throw new ArgumentException("Expected 8bpp indexed bitmap.", nameof(image));

            var rect = new Rectangle(0, 0, image.Width, image.Height);
            using (var data = new BitmapDataLock(image, ImageLockMode.ReadWrite, image.PixelFormat)) {
                var um = data.ToUnmanaged();
                threshold.ThresholdValue = CalculateThreshold(um, rect);
                threshold.ApplyInPlace(um, rect);
            }
        }

        private static int CalculateThreshold(UnmanagedImageLocal image, Rectangle rect) {
            if (image.PixelFormat != PixelFormat.Format8bppIndexed) {
                throw new ArgumentException("Expected 8bpp indexed image.", nameof(image));
            }

            int startX = rect.Left;
            int startY = rect.Top;
            int stopX = startX + rect.Width;
            int stopY = startY + rect.Height;
            int stopXM1 = stopX - 1;
            int stopYM1 = stopY - 1;
            int stride = image.Stride;
            int offset = stride - rect.Width;

            double weightTotal = 0, total = 0;

            unsafe {
                byte* ptr = (byte*)image.ImageData.ToPointer();
                ptr += (startY * image.Stride + startX);
                ptr += stride;

                for (int y = startY + 1; y < stopYM1; y++) {
                    ptr++;
                    for (int x = startX + 1; x < stopXM1; x++, ptr++) {
                        double ex = Math.Abs(ptr[1] - ptr[-1]);
                        double ey = Math.Abs(ptr[stride] - ptr[-stride]);
                        double weight = (ex > ey) ? ex : ey;
                        weightTotal += weight;
                        total += weight * (*ptr);
                    }
                    ptr += offset + 1;
                }
            }

            return (weightTotal == 0) ? 0 : (int)(total / weightTotal);
        }

        private sealed class Threshold {
            public int ThresholdValue { get; set; }

            public void ApplyInPlace(UnmanagedImageLocal image, Rectangle rect) {
                unsafe {
                    int stride = image.Stride;
                    int offset = stride - rect.Width;
                    byte* ptr = (byte*)image.ImageData.ToPointer();
                    ptr += rect.Top * stride + rect.Left;

                    for (int y = rect.Top; y < rect.Bottom; y++) {
                        for (int x = rect.Left; x < rect.Right; x++, ptr++) {
                            *ptr = (byte)((*ptr >= ThresholdValue) ? 255 : 0);
                        }
                        ptr += offset;
                    }
                }
            }
        }
    }
}
