#region "copyright"
/*
    Local copy of AForge BinaryDilation3x3 to allow tuning and buffer reuse.
*/
#endregion

using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Buffers;

namespace NINA.Image.ImageAnalysis.Filters {
    public class BinaryDilation3x3 {
        public void ApplyInPlace(Bitmap image) {
            if (image == null) throw new ArgumentNullException(nameof(image));
            if (image.PixelFormat != PixelFormat.Format8bppIndexed) throw new ArgumentException("Expected 8bpp indexed bitmap.", nameof(image));

            var rect = new Rectangle(0, 0, image.Width, image.Height);
            using var data = new BitmapDataLock(image, ImageLockMode.ReadWrite, image.PixelFormat);

            var srcData = data.ToUnmanaged();
            int bufferSize = srcData.Stride * srcData.Height;
            byte[] temp = ArrayPool<byte>.Shared.Rent(bufferSize);

            try {
                unsafe {
                    // Copy source into temp buffer so we can write the dilation back into the same bitmap (true in-place semantics).
                    fixed (byte* tempBase = temp) {
                        for (int y = 0; y < srcData.Height; y++) {
                            Buffer.MemoryCopy(
                                (byte*)srcData.ImageData + y * srcData.Stride,
                                tempBase + y * srcData.Stride,
                                srcData.Stride,
                                srcData.Stride);
                        }

                        Process(tempBase, srcData.ImageData, srcData.Stride, rect);
                    }
                }
            } finally {
                ArrayPool<byte>.Shared.Return(temp);
            }
        }

        private unsafe void Process(byte* srcBase, IntPtr dstPtr, int stride, Rectangle rect) {
            if (rect.Width < 3 || rect.Height < 3) {
                throw new ArgumentException("Processing rectangle must be at least 3x3 in size.", nameof(rect));
            }

            int startX = rect.Left + 1;
            int startY = rect.Top + 1;
            int stopX = rect.Right - 1;
            int stopY = rect.Bottom - 1;

            int dstStride = stride;
            int srcStride = stride;

            int dstOffset = dstStride - rect.Width + 1;
            int srcOffset = srcStride - rect.Width + 1;

            byte* src = srcBase;
            byte* dst = (byte*)dstPtr.ToPointer();

            src += (startX - 1) + (startY - 1) * srcStride;
            dst += (startX - 1) + (startY - 1) * dstStride;

            // Bitwise OR dilation: intended for binary masks (0 or 255). On grayscale inputs it will brighten values;
            // callers should threshold first to preserve gradients.
            *dst = (byte)(*src | src[1] | src[srcStride] | src[srcStride + 1]);
            src++; dst++;

            for (int x = startX; x < stopX; x++, src++, dst++) {
                *dst = (byte)(*src | src[-1] | src[1] |
                    src[srcStride] | src[srcStride - 1] | src[srcStride + 1]);
            }

            *dst = (byte)(*src | src[-1] | src[srcStride] | src[srcStride - 1]);

            src += srcOffset;
            dst += dstOffset;

            for (int y = startY; y < stopY; y++) {
                *dst = (byte)(*src | src[1] |
                    src[-srcStride] | src[-srcStride + 1] |
                    src[srcStride] | src[srcStride + 1]);

                src++; dst++;

                for (int x = startX; x < stopX; x++, src++, dst++) {
                    *dst = (byte)(*src | src[-1] | src[1] |
                        src[-srcStride] | src[-srcStride - 1] | src[-srcStride + 1] |
                        src[srcStride] | src[srcStride - 1] | src[srcStride + 1]);
                }

                *dst = (byte)(*src | src[-1] |
                    src[-srcStride] | src[-srcStride - 1] |
                    src[srcStride] | src[srcStride - 1]);

                src += srcOffset;
                dst += dstOffset;
            }

            *dst = (byte)(*src | src[1] | src[-srcStride] | src[-srcStride + 1]);
            src++; dst++;

            for (int x = startX; x < stopX; x++, src++, dst++) {
                *dst = (byte)(*src | src[-1] | src[1] |
                    src[-srcStride] | src[-srcStride - 1] | src[-srcStride + 1]);
            }

            *dst = (byte)(*src | src[-1] | src[-srcStride] | src[-srcStride - 1]);
        }
    }
}
