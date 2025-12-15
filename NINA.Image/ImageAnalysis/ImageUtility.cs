#region "copyright"

/*
    Copyright пїЅ 2016 - 2024 Stefan Berg <isbeorn86+NINA@googlemail.com> and the N.I.N.A. contributors

    This file is part of N.I.N.A. - Nighttime Imaging 'N' Astronomy.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#endregion "copyright"

using Accord.Imaging;
using NINA.Core.Enum;
using NINA.Core.Locale;
using NINA.Core.Utility;
using NINA.Image.Interfaces;
using NINA.Image.ImageData;
using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Interop;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace NINA.Image.ImageAnalysis {

    public class ImageUtility {

        // Per-thread scratch buffer to avoid ArrayPool rent/return churn in 16->8 conversion.
        private static readonly System.Threading.ThreadLocal<byte[]> _convert16To8Buffer = new System.Threading.ThreadLocal<byte[]>();

        public static ColorRemappingGeneral GetColorRemappingFilter(
            IImageStatistics statistics,
            double targetHistogramMeanPct,
            double shadowsClipping,
            System.Windows.Media.PixelFormat pf) {
            ushort[] map = GetStretchMap(statistics, targetHistogramMeanPct, shadowsClipping);

            if (pf == PixelFormats.Gray16) {
                var filter = new ColorRemappingGeneral(map);
                return filter;
            } else if (pf == PixelFormats.Rgb48) {
                var filter = new ColorRemappingGeneral(map, map, map);
                return filter;
            } else {
                throw new NotSupportedException();
            }
        }

        public static ColorRemappingGeneral GetColorRemappingFilterUnlinked(
            IImageStatistics redStatistics,
            IImageStatistics greenStatistics,
            IImageStatistics blueStatistics,
            double targetHistogramMeanPct,
            double shadowsClipping,
            System.Windows.Media.PixelFormat pf) {
            ushort[] mapRed = GetStretchMap(redStatistics, targetHistogramMeanPct, shadowsClipping);
            ushort[] mapGreen = GetStretchMap(greenStatistics, targetHistogramMeanPct, shadowsClipping);
            ushort[] mapBlue = GetStretchMap(blueStatistics, targetHistogramMeanPct, shadowsClipping);
            if (pf == PixelFormats.Rgb48) {
                var filter = new ColorRemappingGeneral(mapRed, mapGreen, mapBlue);
                return filter;
            } else {
                throw new NotSupportedException();
            }
        }

        /// <summary>
        /// Adjusts x for a given midToneBalance
        /// </summary>
        /// <param name="midToneBalance"></param>
        /// <param name="x"></param>
        /// <returns></returns>
        private static double MidtonesTransferFunction(double midToneBalance, double x) {
            if (x > 0) {
                if (x < 1) {
                    return (midToneBalance - 1) * x / ((2 * midToneBalance - 1) * x - midToneBalance);
                }
                return 1;
            }
            return 0;
        }

        /// <summary>
        /// Converts a value from range [0;65535] to [0;1]
        /// </summary>
        /// <param name="val"></param>
        /// <returns></returns>
        public static double NormalizeUShort(double val, int bitDepth) {
            return val / (double)((1 << bitDepth) - 1);
        }

        /// <summary>
        /// Converts a value from range [0;1] to [0;65535]
        /// </summary>
        /// <param name="val"></param>
        /// <returns></returns>
        public static ushort DenormalizeUShort(double val) {
            return (ushort)(val * ushort.MaxValue + (val < 0.5 ? 0.5 : 0.0));
        }

        private static ushort[] GetStretchMap(IImageStatistics statistics, double targetHistogramMedianPercent, double shadowsClipping) {
            ushort[] map = new ushort[ushort.MaxValue + 1];

            var normalizedMedian = NormalizeUShort(statistics.Median, statistics.BitDepth);
            var normalizedMAD = NormalizeUShort(statistics.MedianAbsoluteDeviation, statistics.BitDepth);

            var scaleFactor = 1.4826; // see https://en.wikipedia.org/wiki/Median_absolute_deviation

            double shadows = 0d;
            double midtones = 0.5d;
            double highlights = 1d;

            //Assume the image is inverted or overexposed when median is higher than half of the possible value
            if (normalizedMedian > 0.5) {
                shadows = 0.0d;
                highlights = normalizedMedian - shadowsClipping * normalizedMAD * scaleFactor;
                midtones = MidtonesTransferFunction(targetHistogramMedianPercent, 1.0 - (highlights - normalizedMedian));
            } else {
                shadows = normalizedMedian + shadowsClipping * normalizedMAD * scaleFactor;
                midtones = MidtonesTransferFunction(targetHistogramMedianPercent, normalizedMedian - shadows);
                highlights = 1;
            }

            for (int i = 0; i < map.Length; i++) {
                double value = NormalizeUShort(i, statistics.BitDepth);

                map[i] = DenormalizeUShort(MidtonesTransferFunction(midtones, 1 - highlights + value - shadows));
            }

            return map;
        }

        public static BitmapSource ConvertBitmap(System.Drawing.Bitmap bitmap) {
            System.Windows.Media.PixelFormat pf;

            switch (bitmap.PixelFormat) {
                case System.Drawing.Imaging.PixelFormat.Format16bppRgb565:
                    pf = System.Windows.Media.PixelFormats.Bgr565;
                    break;

                case System.Drawing.Imaging.PixelFormat.Format32bppRgb:
                    pf = System.Windows.Media.PixelFormats.Bgra32;
                    break;

                case System.Drawing.Imaging.PixelFormat.Format32bppArgb:
                    pf = System.Windows.Media.PixelFormats.Bgra32;
                    break;

                default:
                    pf = System.Windows.Media.PixelFormats.Gray16;
                    break;
            }
            return ConvertBitmap(bitmap, pf);
        }

        public static BitmapSource ConvertBitmap(System.Drawing.Bitmap bitmap, System.Windows.Media.PixelFormat pf) {
            var bitmapData = bitmap.LockBits(
                new System.Drawing.Rectangle(0, 0, bitmap.Width, bitmap.Height),
                System.Drawing.Imaging.ImageLockMode.ReadOnly, bitmap.PixelFormat);

            var bitmapSource = BitmapSource.Create(
                bitmapData.Width, bitmapData.Height, 96, 96, pf, null,
                bitmapData.Scan0, bitmapData.Stride * bitmapData.Height, bitmapData.Stride);

            bitmap.UnlockBits(bitmapData);
            return bitmapSource;
        }

        [DllImport("gdi32.dll")]
        private static extern bool DeleteObject(IntPtr hObject);

        private static BitmapSource CreateBitmapSourceFast(Bitmap bitmap, System.Windows.Media.PixelFormat pf) {
            _ = pf; // format already defined by bitmap; keep parameter for parity with ConvertBitmap
            var hBitmap = bitmap.GetHbitmap();
            try {
                var source = Imaging.CreateBitmapSourceFromHBitmap(
                    hBitmap,
                    IntPtr.Zero,
                    Int32Rect.Empty,
                    BitmapSizeOptions.FromEmptyOptions());
                source.Freeze();
                return source;
            } finally {
                DeleteObject(hBitmap);
            }
        }

        private static void ApplyStretchUnlinkedInPlace(
            ushort[] buffer,
            int width,
            int height,
            IImageStatistics redStats,
            IImageStatistics greenStats,
            IImageStatistics blueStats,
            double factor,
            double blackClipping) {

            if (buffer == null || buffer.Length < width * height * 3)
                return;

            ushort[] mapR = GetStretchMap(redStats, factor, blackClipping);
            ushort[] mapG = GetStretchMap(greenStats, factor, blackClipping);
            ushort[] mapB = GetStretchMap(blueStats, factor, blackClipping);

            int len = width * height;
            int idx = 0;
            for (int i = 0; i < len; i++) {
                ushort r = buffer[idx];
                ushort g = buffer[idx + 1];
                ushort b = buffer[idx + 2];
                buffer[idx] = mapR[r];
                buffer[idx + 1] = mapG[g];
                buffer[idx + 2] = mapB[b];
                idx += 3;
            }
        }

        public static Bitmap BitmapFromSource(BitmapSource source) {
            return BitmapFromSource(source, System.Drawing.Imaging.PixelFormat.Format16bppGrayScale);
        }

        public static Bitmap BitmapFromSource(BitmapSource source, System.Drawing.Imaging.PixelFormat pf) {
            Bitmap bmp = new Bitmap(
                    source.PixelWidth,
                    source.PixelHeight,
                    pf);
            BitmapData data = bmp.LockBits(
                    new Rectangle(System.Drawing.Point.Empty, bmp.Size),
                    ImageLockMode.WriteOnly,
                    pf);
            source.CopyPixels(
                    Int32Rect.Empty,
                    data.Scan0,
                    data.Height * data.Stride,
                    data.Stride);
            bmp.UnlockBits(data);
            return bmp;
        }

        public static Bitmap Convert16BppTo8Bpp(BitmapSource source) {
            using (MyStopWatch.Measure()) {
                // Fast path for 16bpp gray -> 8bpp indexed. Bit-exact: we keep the high byte of each ushort.
                int width = source.PixelWidth;
                int height = source.PixelHeight;
                int srcStride = width * 2;
                int bufferSize = srcStride * height;

                // Reuse thread-local buffer; grow if the current image does not fit.
                byte[] srcBuffer = _convert16To8Buffer.Value;
                if (srcBuffer == null || srcBuffer.Length < bufferSize) {
                    srcBuffer = new byte[bufferSize];
                    _convert16To8Buffer.Value = srcBuffer;
                }
                // Copy WPF source into contiguous byte buffer (little-endian ushort layout).
                source.CopyPixels(new Int32Rect(0, 0, width, height), srcBuffer, srcStride, 0);

                var bmp = new Bitmap(width, height, System.Drawing.Imaging.PixelFormat.Format8bppIndexed);
                bmp.Palette = GetGrayScalePalette();

                var data = bmp.LockBits(
                    new Rectangle(System.Drawing.Point.Empty, bmp.Size),
                    ImageLockMode.WriteOnly,
                    System.Drawing.Imaging.PixelFormat.Format8bppIndexed);

                try {
                    unsafe {
                        byte* dstBase = (byte*)data.Scan0;
                        int dstStride = data.Stride;

                        fixed (byte* srcBase = srcBuffer) {
                            for (int y = 0; y < height; y++) {
                                byte* src = srcBase + y * srcStride;
                                byte* dst = dstBase + y * dstStride;

                                int x = 0;
                                int limit = width - 7; // unroll in groups of 8 pixels
                                while (x <= limit) {
                                    // Unroll 8 pixels: copy high byte of each ushort (bit-precise path).
                                    dst[x + 0] = src[(x + 0) * 2 + 1];
                                    dst[x + 1] = src[(x + 1) * 2 + 1];
                                    dst[x + 2] = src[(x + 2) * 2 + 1];
                                    dst[x + 3] = src[(x + 3) * 2 + 1];
                                    dst[x + 4] = src[(x + 4) * 2 + 1];
                                    dst[x + 5] = src[(x + 5) * 2 + 1];
                                    dst[x + 6] = src[(x + 6) * 2 + 1];
                                    dst[x + 7] = src[(x + 7) * 2 + 1];
                                    x += 8;
                                }
                                for (; x < width; x++) {
                                    dst[x] = src[(x << 1) + 1];
                                }
                            }
                        }
                    }
                } finally {
                    bmp.UnlockBits(data);
                }

                return bmp;
            }
        }

        public static BitmapSource Convert16BppTo8BppSource(BitmapSource source) {
            FormatConvertedBitmap s = new FormatConvertedBitmap();
            s.BeginInit();
            s.Source = source;
            s.DestinationFormat = System.Windows.Media.PixelFormats.Gray8;
            s.EndInit();
            s.Freeze();
            return s;
        }

        public static BitmapSource CreateSourceFromArray(IImageArray arr, ImageProperties props, System.Windows.Media.PixelFormat pf) {
            //int stride = C.CameraYSize * ((Convert.ToString(C.MaxADU, 2)).Length + 7) / 8;
            int stride = (props.Width * pf.BitsPerPixel + 7) / 8;
            double dpi = 96;

            BitmapSource source = BitmapSource.Create(props.Width, props.Height, dpi, dpi, pf, null, arr.FlatArray, stride);
            source.Freeze();
            return source;
        }

        public static DebayeredImageData Debayer(BitmapSource source, System.Drawing.Imaging.PixelFormat pf, bool saveColorChannels = false, bool saveLumChannel = false, SensorType bayerPattern = SensorType.RGGB) {
            using (MyStopWatch.Measure()) {
                if (pf != System.Drawing.Imaging.PixelFormat.Format16bppGrayScale) {
                    throw new NotSupportedException();
                }

                int width = source.PixelWidth;
                int height = source.PixelHeight;

                // Acquire unmanaged buffers and the reusable filter instance up front so we can time every phase explicitly.
                var resources = DebayerBufferPool.Instance.Acquire(width, height, saveColorChannels, saveLumChannel, bayerPattern);
                var filter = resources.Filter;

                // Measure the copy from the WPF source into the unmanaged input buffer to confirm how much of the Debayer time is pure memory traffic.
                using (MyStopWatch.Measure($"{nameof(Debayer)}_CopyInput")) {
                    var rect = new Int32Rect(0, 0, width, height);
                    source.CopyPixels(rect, resources.InputPtr, resources.InputSizeBytes, resources.InputStride);
                }

                // Time the actual filter execution (includes demosaic and optional channel extraction) separately from wrapping and copies.
                using (MyStopWatch.Measure($"{nameof(Debayer)}_ApplyFilter")) {
                    filter.ApplyInto(resources.InputImage, resources.OutputImage);
                }

                BitmapSource imageSource;
                // Track the cost of turning the unmanaged RGB buffer into a frozen BitmapSource (WPF copies here).
                using (MyStopWatch.Measure($"{nameof(Debayer)}_WrapBitmap")) {
                    imageSource = BitmapSource.Create(
                        width,
                        height,
                        96,
                        96,
                        PixelFormats.Rgb48,
                        null,
                        resources.OutputPtr,
                        resources.OutputSizeBytes,
                        resources.OutputStride);
                    imageSource.Freeze();
                }

                // Clone the optional L/R/G/B planes because the pooled filter instance may be reused on the next call.
                LRGBArrays dataCopy = null;
                var arrays = filter.LRGBArrays;
                if (arrays != null) {
                    ushort[] lum = Array.Empty<ushort>();
                    ushort[] r = Array.Empty<ushort>();
                    ushort[] g = Array.Empty<ushort>();
                    ushort[] b = Array.Empty<ushort>();

                    if (saveLumChannel && arrays.Lum != null && arrays.Lum.Length > 0) {
                        lum = new ushort[arrays.Lum.Length];
                        Array.Copy(arrays.Lum, lum, lum.Length);
                    }
                    if (saveColorChannels) {
                        if (arrays.Red != null && arrays.Red.Length > 0) {
                            r = new ushort[arrays.Red.Length];
                            Array.Copy(arrays.Red, r, r.Length);
                        }
                        if (arrays.Green != null && arrays.Green.Length > 0) {
                            g = new ushort[arrays.Green.Length];
                            Array.Copy(arrays.Green, g, g.Length);
                        }
                        if (arrays.Blue != null && arrays.Blue.Length > 0) {
                            b = new ushort[arrays.Blue.Length];
                            Array.Copy(arrays.Blue, b, b.Length);
                        }
                    }

                    if (lum.Length > 0 || r.Length > 0 || g.Length > 0 || b.Length > 0) {
                        dataCopy = new LRGBArrays(lum, r, g, b);
                    }
                }

                return new DebayeredImageData {
                    ImageSource = imageSource,
                    Data = dataCopy
                };
            }
        }

        public static DebayeredImageData Debayer(Bitmap bmp, bool saveColorChannels = false, bool saveLumChannel = false, SensorType bayerPattern = SensorType.RGGB) {
            using (MyStopWatch.Measure()) {
                var filter = new BayerFilter16bpp();
                filter.SaveColorChannels = saveColorChannels;
                filter.SaveLumChannel = saveLumChannel;

                Logger.Debug($"Debayering pattern {bayerPattern}");

                switch (bayerPattern) {
                    case SensorType.RGGB:
                        filter.BayerPattern = new int[,] { { RGB.B, RGB.G }, { RGB.G, RGB.R } };
                        break;

                    case SensorType.RGBG:
                        filter.BayerPattern = new int[,] { { RGB.G, RGB.B }, { RGB.G, RGB.R } };
                        break;

                    case SensorType.GRGB:
                        filter.BayerPattern = new int[,] { { RGB.B, RGB.G }, { RGB.R, RGB.G } };
                        break;

                    case SensorType.GRBG:
                        filter.BayerPattern = new int[,] { { RGB.G, RGB.B }, { RGB.R, RGB.G } };
                        break;

                    case SensorType.GBGR:
                        filter.BayerPattern = new int[,] { { RGB.R, RGB.G }, { RGB.B, RGB.G } };
                        break;

                    case SensorType.GBRG:
                        filter.BayerPattern = new int[,] { { RGB.G, RGB.R }, { RGB.B, RGB.G } };
                        break;

                    case SensorType.BGRG:
                        filter.BayerPattern = new int[,] { { RGB.G, RGB.R }, { RGB.G, RGB.B } };
                        break;

                    case SensorType.BGGR:
                        filter.BayerPattern = new int[,] { { RGB.R, RGB.G }, { RGB.G, RGB.B } };
                        break;

                    default:
                        throw new InvalidImagePropertiesException(string.Format(Loc.Instance["LblUnsupportedCfaPattern"], bayerPattern));
                }

                DebayeredImageData debayered = new DebayeredImageData();
                using (var outBmp = filter.Apply(bmp)) {
                    debayered.ImageSource = ConvertBitmap(outBmp, PixelFormats.Rgb48);
                    debayered.ImageSource.Freeze();
                }
                debayered.Data = filter.LRGBArrays;
                return debayered;
            }
        }

        private static readonly object _paletteLock = new object();
        private static ColorPalette _grayPalette;

        public static ColorPalette GetGrayScalePalette() {
            if (_grayPalette != null) return _grayPalette;
            lock (_paletteLock) {
                if (_grayPalette != null) return _grayPalette;
                using (var bmp = new Bitmap(1, 1, System.Drawing.Imaging.PixelFormat.Format8bppIndexed)) {
                    var monoPalette = bmp.Palette;
                    var entries = monoPalette.Entries;
                    for (int i = 0; i < 256; i++) {
                        entries[i] = System.Drawing.Color.FromArgb(i, i, i);
                    }
                    _grayPalette = monoPalette;
                }
            }
            return _grayPalette;
        }

        public static Task<BitmapSource> Stretch(IRenderedImage image, double factor, double blackClipping) {
            return Task.Run(async () => {
                var imageStatistics = await image.RawImageData.Statistics.Task;
                if (image.OriginalImage.Format == PixelFormats.Gray16) {
                    using (var bmp = ImageUtility.BitmapFromSource(image.OriginalImage, System.Drawing.Imaging.PixelFormat.Format16bppGrayScale)) {
                        return Stretch(imageStatistics, bmp, image.OriginalImage.Format, factor, blackClipping);
                    }
                } else if (image.OriginalImage.Format == PixelFormats.Rgb48) {
                    using (var bmp = ImageUtility.BitmapFromSource(image.OriginalImage, System.Drawing.Imaging.PixelFormat.Format48bppRgb)) {
                        return Stretch(imageStatistics, bmp, image.OriginalImage.Format, factor, blackClipping);
                    }
                } else {
                    throw new NotSupportedException();
                }
            });
        }

        public static Task<BitmapSource> StretchUnlinked(IDebayeredImage data, double factor, double blackClipping) {
            return Task.Run(async () => {
                if (data.OriginalImage.Format != PixelFormats.Rgb48) {
                    throw new NotSupportedException();
                } else {
                    var rgbStats = await Task.Run(() => FastImageStatistics.CreateRgb(data.RawImageData.Properties, data.DebayeredData));

                    using (var img = ImageUtility.BitmapFromSource(data.OriginalImage, System.Drawing.Imaging.PixelFormat.Format48bppRgb)) {
                        return StretchUnlinked(rgbStats.Red, rgbStats.Green, rgbStats.Blue, img, data.OriginalImage.Format, factor, blackClipping);
                    }
                }
            });
        }

        public static BitmapSource StretchUnlinked(
            IImageStatistics redStatistics,
            IImageStatistics greenStatistics,
            IImageStatistics blueStatistics,
            Bitmap img,
            System.Windows.Media.PixelFormat pf,
            double factor,
            double blackClipping) {
            using (MyStopWatch.Measure()) {
                // Fallback to bitmap-based path if needed elsewhere.
                var filter = ImageUtility.GetColorRemappingFilterUnlinked(blueStatistics, greenStatistics, redStatistics, factor, blackClipping, pf);
                filter.ApplyInPlace(img);
                var source = ImageUtility.ConvertBitmap(img, pf);
                source.Freeze();
                return source;
            }
        }

        public static BitmapSource Stretch(IImageStatistics statistics, Bitmap img, System.Windows.Media.PixelFormat pf, double factor, double blackClipping) {
            using (MyStopWatch.Measure()) {
                var filter = ImageUtility.GetColorRemappingFilter(statistics, factor, blackClipping, pf);
                filter.ApplyInPlace(img);

                var source = ImageUtility.ConvertBitmap(img, pf);
                source.Freeze();
                return source;
            }
        }
    }
}
