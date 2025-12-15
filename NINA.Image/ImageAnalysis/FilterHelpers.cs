#region "copyright"
/*
    Shared helper definitions for local filters.
*/
#endregion

using System;
using System.Drawing;
using System.Drawing.Imaging;

namespace NINA.Image.ImageAnalysis.Filters {
    internal static class RgbIdx {
        public const int B = 0;
        public const int G = 1;
        public const int R = 2;
        public const int A = 3;
    }

    internal static class LocalInterpolation {
        public static double BiCubicKernel(double x) {
            if (x < 0) x = -x;
            if (x <= 1.0) {
                return (1.5 * x - 2.5) * x * x + 1.0;
            }
            if (x < 2.0) {
                return ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0;
            }
            return 0.0;
        }
    }

    internal sealed class BitmapDataLock : IDisposable {
        private readonly Bitmap _bitmap;
        private readonly BitmapData _data;

        public BitmapDataLock(Bitmap bitmap, ImageLockMode mode, PixelFormat format) {
            _bitmap = bitmap;
            _data = bitmap.LockBits(new Rectangle(0, 0, bitmap.Width, bitmap.Height), mode, format);
        }

        public UnmanagedImageLocal ToUnmanaged() => new UnmanagedImageLocal(_data);

        public void Dispose() {
            _bitmap.UnlockBits(_data);
        }
    }

    internal sealed class UnmanagedImageLocal : IDisposable {
        private readonly bool _owns;
        public IntPtr ImageData { get; }
        public int Width { get; }
        public int Height { get; }
        public int Stride { get; }
        public PixelFormat PixelFormat { get; }

        public UnmanagedImageLocal(BitmapData data) {
            ImageData = data.Scan0;
            Width = data.Width;
            Height = data.Height;
            Stride = data.Stride;
            PixelFormat = data.PixelFormat;
            _owns = false;
        }

        private UnmanagedImageLocal(IntPtr buffer, int width, int height, int stride, PixelFormat format) {
            ImageData = buffer;
            Width = width;
            Height = height;
            Stride = stride;
            PixelFormat = format;
            _owns = true;
        }

        public static UnmanagedImageLocal Create(int width, int height, PixelFormat format) {
            int pixelSize = System.Drawing.Image.GetPixelFormatSize(format) / 8;
            int stride = width * pixelSize;
            int bytes = stride * height;
            IntPtr buffer = System.Runtime.InteropServices.Marshal.AllocHGlobal(bytes);
            return new UnmanagedImageLocal(buffer, width, height, stride, format);
        }

        public unsafe void CopyTo(UnmanagedImageLocal destination) {
            if (destination == null) throw new ArgumentNullException(nameof(destination));
            if (destination.Stride == Stride && destination.Height == Height) {
                System.Buffer.MemoryCopy((void*)ImageData, (void*)destination.ImageData, destination.Stride * destination.Height, Stride * Height);
            } else {
                byte* src = (byte*)ImageData.ToPointer();
                byte* dst = (byte*)destination.ImageData.ToPointer();
                int rows = Math.Min(Height, destination.Height);
                int bytesPerRow = Math.Min(Stride, destination.Stride);
                for (int y = 0; y < rows; y++) {
                    System.Buffer.MemoryCopy(src + y * Stride, dst + y * destination.Stride, destination.Stride, bytesPerRow);
                }
            }
        }

        public void Dispose() {
            if (_owns && ImageData != IntPtr.Zero) {
                System.Runtime.InteropServices.Marshal.FreeHGlobal(ImageData);
            }
        }
    }
}
