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
using Accord.Imaging.Filters;
using FluentAssertions;
using NINA.Core.Enum;
using NINA.Core.Model;
using NINA.Image.ImageAnalysis;
using NINA.Image.ImageData;
using NINA.Image.Interfaces;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;

namespace NINA.Test.Image.ImageAnalysis {

    internal static class ImageTestHelpers {
        public static Bitmap Create16bppGray(int width, int height, ushort[] data) {
            var bitmap = new Bitmap(width, height, PixelFormat.Format16bppGrayScale);
            var rect = new Rectangle(0, 0, width, height);
            var bmpData = bitmap.LockBits(rect, ImageLockMode.WriteOnly, PixelFormat.Format16bppGrayScale);

            var strideBytes = bmpData.Stride;
            var buffer = new byte[strideBytes * height];
            for (int y = 0; y < height; y++) {
                Buffer.BlockCopy(data, y * width * sizeof(ushort), buffer, y * strideBytes, width * sizeof(ushort));
            }
            Marshal.Copy(buffer, 0, bmpData.Scan0, buffer.Length);

            bitmap.UnlockBits(bmpData);
            return bitmap;
        }

        public static Bitmap Create8bppGray(int width, int height, byte[] data) {
            var bitmap = new Bitmap(width, height, PixelFormat.Format8bppIndexed);
            var palette = bitmap.Palette;
            for (int i = 0; i < palette.Entries.Length; i++) {
                palette.Entries[i] = System.Drawing.Color.FromArgb(i, i, i);
            }
            bitmap.Palette = palette;

            var rect = new Rectangle(0, 0, width, height);
            var bmpData = bitmap.LockBits(rect, ImageLockMode.WriteOnly, PixelFormat.Format8bppIndexed);
            var stride = bmpData.Stride;
            var target = new byte[stride * height];

            for (int y = 0; y < height; y++) {
                Buffer.BlockCopy(data, y * width, target, y * stride, width);
            }

            Marshal.Copy(target, 0, bmpData.Scan0, target.Length);
            bitmap.UnlockBits(bmpData);
            return bitmap;
        }

        public static ushort[] Read48bppPixelBlock(Bitmap bitmap) {
            var rect = new Rectangle(0, 0, bitmap.Width, bitmap.Height);
            var data = bitmap.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format48bppRgb);
            try {
                var buffer = new byte[data.Stride * data.Height];
                Marshal.Copy(data.Scan0, buffer, 0, buffer.Length);
                var pixels = new ushort[buffer.Length / 2];
                Buffer.BlockCopy(buffer, 0, pixels, 0, buffer.Length);
                return pixels;
            } finally {
                bitmap.UnlockBits(data);
            }
        }

        public static ushort[] Read16bppPixelBlock(Bitmap bitmap) {
            var rect = new Rectangle(0, 0, bitmap.Width, bitmap.Height);
            var data = bitmap.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format16bppGrayScale);
            try {
                var buffer = new byte[data.Stride * data.Height];
                Marshal.Copy(data.Scan0, buffer, 0, buffer.Length);
                var pixels = new ushort[bitmap.Width * bitmap.Height];
                var stride = data.Stride / sizeof(ushort);

                for (int y = 0; y < bitmap.Height; y++) {
                    Buffer.BlockCopy(buffer, y * stride * sizeof(ushort), pixels, y * bitmap.Width * sizeof(ushort), bitmap.Width * sizeof(ushort));
                }

                return pixels;
            } finally {
                bitmap.UnlockBits(data);
            }
        }

        public static byte[] Read8bppPixelBlock(Bitmap bitmap) {
            var rect = new Rectangle(0, 0, bitmap.Width, bitmap.Height);
            var data = bitmap.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format8bppIndexed);
            try {
                var buffer = new byte[data.Stride * data.Height];
                Marshal.Copy(data.Scan0, buffer, 0, buffer.Length);
                return buffer;
            } finally {
                bitmap.UnlockBits(data);
            }
        }

        public static UnmanagedImage CreateManagedStrideImage(byte[] pixels, int width, int height, PixelFormat format) {
            int pixelSize = System.Drawing.Image.GetPixelFormatSize(format) / 8;
            var stride = width * pixelSize;
            var unmanaged = UnmanagedImage.Create(width, height, stride, format);
            Marshal.Copy(pixels, 0, unmanaged.ImageData, pixels.Length);
            return unmanaged;
        }
    }

    [TestFixture]
    [Apartment(System.Threading.ApartmentState.STA)]
    public class BayerFilter16bppTests {
        [Test]
        public void Demosaic2x2ProducesExpectedRGB() {
            // Arrange
            const int width = 12;
            const int height = 12;
            var sourcePixels = new ushort[width * height];
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    sourcePixels[y * width + x] = (ushort)((y % 2 == 0 && x % 2 == 0)
                        ? 100
                        : (y % 2 == 1 && x % 2 == 1 ? 400 : 250));
                }
            }
            using var src = ImageTestHelpers.Create16bppGray(width, height, sourcePixels);

            var filter = new BayerFilter16bpp {
                BayerPattern = new int[,] { { RGB.B, RGB.G }, { RGB.G, RGB.R } },
                SaveColorChannels = true,
                SaveLumChannel = true
            };

            // Act
            using var result = filter.Apply(src);
            var rect = new Rectangle(0, 0, result.Width, result.Height);
            var data = result.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format48bppRgb);
            var pixels = new byte[data.Stride * data.Height];
            Marshal.Copy(data.Scan0, pixels, 0, pixels.Length);
            result.UnlockBits(data);

            var channelData = new ushort[pixels.Length / 2];
            Buffer.BlockCopy(pixels, 0, channelData, 0, pixels.Length);

            // Assert
            var stridePixels = data.Stride / sizeof(ushort);
            for (int y = 0; y < result.Height; y++) {
                for (int x = 0; x < result.Width; x++) {
                    var idx = y * stridePixels + x * 3;
                    channelData[idx + RGB.B].Should().Be((ushort)100);
                    channelData[idx + RGB.G].Should().Be((ushort)250);
                    channelData[idx + RGB.R].Should().Be((ushort)400);
                }
            }

            filter.LRGBArrays.Should().NotBeNull();
            filter.LRGBArrays.Red.Should().OnlyContain(v => v == 100);
            filter.LRGBArrays.Green.Should().OnlyContain(v => v == 250);
            filter.LRGBArrays.Blue.Should().OnlyContain(v => v == 400);
            filter.LRGBArrays.Lum.Should().OnlyContain(v => v == 250);
        }
    }

    [TestFixture]
    public class ColorRemappingGeneralTests {
        [Test]
        public void RemapGrayValuesWithCustomMap() {
            // Arrange
            var map = new ushort[ushort.MaxValue + 1];
            for (int i = 0; i < map.Length; i++) {
                map[i] = (ushort)i;
            }
            map[1] = 1234;
            map[2] = 4321;
            const int width = 12;
            const int height = 12;
            var pixels = new ushort[width * height];
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    pixels[y * width + x] = (ushort)(x % 3);
                }
            }
            using var src = ImageTestHelpers.Create16bppGray(width, height, pixels);

            var filter = new ColorRemappingGeneral(map);

            // Act
            filter.ApplyInPlace(src);
            var data = ImageTestHelpers.Read16bppPixelBlock(src);

            // Assert
            data[0].Should().Be(0);
            data[1].Should().Be(1234);
            data[2].Should().Be(4321);
            data[width].Should().Be(0);
            data[width + 1].Should().Be(1234);
            data[width + 2].Should().Be(4321);
        }

        [Test]
        public void RemapRgbValuesWithSeparatedMaps() {
            // Arrange
            var redMap = CreateIdentityMap(10, 5555);
            var greenMap = CreateIdentityMap(20, 4444);
            var blueMap = CreateIdentityMap(30, 3333);
            const int width = 12;
            const int height = 12;
            using var src = new Bitmap(width, height, PixelFormat.Format48bppRgb);
            var rect = new Rectangle(0, 0, width, height);
            var data = src.LockBits(rect, ImageLockMode.WriteOnly, PixelFormat.Format48bppRgb);
            try {
                var bytes = new byte[Math.Abs(data.Stride) * data.Height];
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        var offset = y * data.Stride + x * 6;
                        BitConverter.GetBytes((ushort)30).CopyTo(bytes, offset + RGB.B * 2);
                        BitConverter.GetBytes((ushort)20).CopyTo(bytes, offset + RGB.G * 2);
                        BitConverter.GetBytes((ushort)10).CopyTo(bytes, offset + RGB.R * 2);
                    }
                }
                Marshal.Copy(bytes, 0, data.Scan0, bytes.Length);
            } finally {
                src.UnlockBits(data);
            }

            var filter = new ColorRemappingGeneral(redMap, greenMap, blueMap);

            // Act
            filter.ApplyInPlace(src);
            var pixels = ImageTestHelpers.Read48bppPixelBlock(src);

            // Assert
            pixels[RGB.B].Should().Be((ushort)3333);
            pixels[RGB.G].Should().Be((ushort)4444);
            pixels[RGB.R].Should().Be((ushort)5555);
        }

        private static ushort[] CreateIdentityMap(int key, ushort value) {
            var map = new ushort[ushort.MaxValue + 1];
            for (int i = 0; i < map.Length; i++) {
                map[i] = (ushort)i;
            }
            map[key] = value;
            return map;
        }
    }

    [TestFixture]
    public class DetectionUtilityTests {
        [Test]
        public void LaplacianOfGaussianKernel_Size3Sigma1MatchesSnapshot() {
            var kernel = DetectionUtility.LaplacianOfGaussianKernel(3, 1.0);
            var expected = new int[,] {
                { 0, -58, 0 },
                { -58, 232, -58 },
                { 0, -58, 0 }
            };

            kernel.Should().BeEquivalentTo(expected);
        }

        [Test]
        public void InRoiHonorsInnerAndOuterCrop() {
            var size = new Size(100, 100);
            var blobInside = new Rectangle(45, 45, 10, 10);
            var blobOutside = new Rectangle(5, 5, 10, 10);

            DetectionUtility.InROI(size, blobInside, outerCropRatio: 0.8, innerCropRatio: 0.5).Should().BeFalse();
            DetectionUtility.InROI(size, blobOutside, outerCropRatio: 0.8, innerCropRatio: 0.5).Should().BeFalse();
        }
    }

    [TestFixture]
    public class FastGaussianBlurTests {
        [Test]
        public void BlurRadius1MatchesSnapshot() {
            // Arrange
            const int width = 12;
            const int height = 12;
            var source = new byte[width * height];
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    var clampedX = Math.Min(x, 3);
                    var clampedY = Math.Min(y, 2);
                    source[y * width + x] = (byte)(clampedY * 40 + clampedX * 10);
                }
            }
            using var bmp = ImageTestHelpers.Create8bppGray(width, height, source);
            var blur = new FastGaussianBlur(bmp);

            // Act
            using var blurred = blur.Process(1);
            var data = ImageTestHelpers.Read8bppPixelBlock(blurred);

            // Assert
            var rect = new Rectangle(0, 0, blurred.Width, blurred.Height);
            var lockData = blurred.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format8bppIndexed);
            var stride = lockData.Stride;
            blurred.UnlockBits(lockData);

            var expected = new byte[,] {
                { 16, 23, 32, 38 },
                { 43, 50, 59, 65 },
                { 69, 76, 85, 91 }
            };

            for (int y = 0; y < expected.GetLength(0); y++) {
                for (int x = 0; x < expected.GetLength(1); x++) {
                    data[y * stride + x].Should().Be(expected[y, x]);
                }
            }
        }
    }

    [TestFixture]
    [Apartment(System.Threading.ApartmentState.STA)]
    public class HistogramAndImageUtilityTests {
        [Test]
        public void HistogramMathExposureStateMatchesBoundaries() {
            HistogramMath.GetExposureAduState(40, 0.5, 8, 0.25).Should().Be(HistogramMath.ExposureAduState.ExposureBelowLowerBound);
            HistogramMath.GetExposureAduState(160, 0.5, 8, 0.25).Should().Be(HistogramMath.ExposureAduState.ExposureWithinBounds);
            HistogramMath.GetExposureAduState(128, 0.5, 8, 0.25).Should().Be(HistogramMath.ExposureAduState.ExposureWithinBounds);
        }

        [Test]
        public void NormalizeAndDenormalizeRoundTrip() {
            var normalized = ImageUtility.NormalizeUShort(32768, 16);
            normalized.Should().BeApproximately(0.5, 1e-5);
            ImageUtility.DenormalizeUShort(normalized).Should().Be((ushort)32768);
        }

        [Test]
        [Apartment(System.Threading.ApartmentState.STA)]
        public void DebayerUsesExpectedChannelValues() {
            const int width = 12;
            const int height = 12;
            var data = new ushort[width * height];
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    data[y * width + x] = (ushort)((y % 2 == 0 && x % 2 == 0)
                        ? 100
                        : (y % 2 == 1 && x % 2 == 1 ? 400 : 250));
                }
            }
            var imageData = new ImageDataFactoryTestUtility();
            var exposure = imageData.ImageDataFactory.CreateBaseImageData(data, width, height, 16, true, new ImageMetaData());
            var source = exposure.RenderBitmapSource();

            var debayered = ImageUtility.Debayer(source, System.Drawing.Imaging.PixelFormat.Format16bppGrayScale, saveColorChannels: true, saveLumChannel: true, bayerPattern: SensorType.RGGB);

            debayered.Data.Red.Should().OnlyContain(v => v == 100);
            debayered.Data.Green.Should().OnlyContain(v => v == 250);
            debayered.Data.Blue.Should().OnlyContain(v => v == 400);
            debayered.Data.Lum.Should().OnlyContain(v => v == 250);
        }
    }

    [TestFixture]
    public class NoBlurCannyEdgeDetectorTests {
        [Test]
        public void SingleEdgeProducesCenteredResponse() {
            // Arrange: 3x3 image with bright right column
            const int width = 12;
            const int height = 12;
            var pixels = new byte[width * height];
            byte[,] pattern = {
                { 0, 0, 255 },
                { 0, 0, 255 },
                { 0, 0, 255 }
            };
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    pixels[y * width + x] = pattern[y % 3, x % 3];
                }
            }
            using var src = ImageTestHelpers.CreateManagedStrideImage(pixels, width, height, PixelFormat.Format8bppIndexed);
            using var dst = UnmanagedImage.Create(width, height, width, PixelFormat.Format8bppIndexed);
            var filter = new TestableNoBlurCannyEdgeDetector();

            // Act
            filter.Run(src, dst);

            // Assert
            var output = new byte[width * height];
            Marshal.Copy(dst.ImageData, output, 0, output.Length);
            output.Should().OnlyContain(v => v == 0 || v == 255);
            output.Count(v => v == 255).Should().BeGreaterThan(0);
            var centerIndex = width + 1;
            output[centerIndex].Should().Be(255);
        }

        private sealed class TestableNoBlurCannyEdgeDetector : NoBlurCannyEdgeDetector {
            public void Run(UnmanagedImage source, UnmanagedImage destination) {
                var rect = new Rectangle(0, 0, source.Width, source.Height);
                ProcessFilter(source, destination, rect);
            }
        }
    }

    [TestFixture]
    public class StarDetectionMathTests {
        [Test]
        public void StarCalculateProducesExpectedHfrAndCentroid() {
            var star = new StarDetection.Star {
                Radius = 1,
                Rectangle = new Rectangle(0, 0, 3, 3),
                Position = new Accord.Point(1, 1),
                SurroundingMean = 0
            };

            var pixels = new List<StarDetection.PixelData> {
                new StarDetection.PixelData(1,1,10),
                new StarDetection.PixelData(1,0,10),
                new StarDetection.PixelData(0,1,10),
                new StarDetection.PixelData(1,2,10),
                new StarDetection.PixelData(2,1,10)
            };

            star.Calculate(pixels);

            star.HFR.Should().BeApproximately(0.8, 1e-6);
            star.Average.Should().BeApproximately(10, 1e-6);
            ((double)star.Position.X).Should().BeApproximately(1d, 1e-6);
            ((double)star.Position.Y).Should().BeApproximately(1d, 1e-6);
            star.MaxPixelValue.Should().Be(0);
        }
    }

    [TestFixture]
    [Apartment(System.Threading.ApartmentState.STA)]
    public class StarAnnotatorTests {
        [Test]
        public async Task AnnotationWritesVisibleOverlay() {
            // Arrange
            const int size = 32;
            var pixels = new ushort[size * size];
            var source = BitmapSource.Create(size, size, 96, 96, System.Windows.Media.PixelFormats.Gray16, null, pixels, size * 2);
            source.Freeze();
            var star = new DetectedStar {
                HFR = 2.0,
                Position = new Accord.Point(1, 1),
                BoundingBox = new Rectangle(0, 0, 2, 2),
                AverageBrightness = 50
            };
            var result = new StarDetectionResult {
                StarList = new List<DetectedStar> { star }
            };
            var annotator = new StarAnnotator();
            var parameters = new StarDetectionParams {
                UseROI = false
            };

            // Act
            var annotated = await annotator.GetAnnotatedImage(parameters, result, source);

            // Assert
            annotated.Should().NotBeNull();
            annotated.Format.Should().Be(System.Windows.Media.PixelFormats.Bgr24);
            var bytes = new byte[annotated.PixelHeight * annotated.PixelWidth * 3];
            annotated.CopyPixels(bytes, annotated.PixelWidth * 3, 0);
            bytes.Any(b => b != 0).Should().BeTrue("overlay should introduce non-zero pixels");
        }
    }

    [TestFixture]
    [Apartment(System.Threading.ApartmentState.STA)]
    public class BahtinovAndContrastTests {
        [Test]
        public void TranslateHoughLineProducesDeterministicLine() {
            var analysis = new TestableBahtinovAnalysis();
            var line = new HoughLine(theta: 45, radius: 10, intensity: 0, relativeIntensity: 1.0);

            var result = analysis.Translate(line, 100, 80);

            ((double)result.Slope).Should().BeApproximately(0.99, 1e-2);
            ((double)result.Intercept).Should().BeApproximately(-24, 1e-1);
        }

        [Test]
        public async Task ContrastDetectionReturnsZeroForFlatFrame() {
            var utility = new ImageDataFactoryTestUtility();
            var arr = new int[16, 16];
            var exposure = await utility.ExposureDataFactory.CreateFlipped2DExposureData(arr, 16, false, new ImageMetaData()).ToImageData();
            var rendered = RenderedImage.Create(exposure.RenderBitmapSource(), exposure, utility.ProfileServiceMock.Object, utility.StarDetectionMock.Object, utility.StarAnnotatorMock.Object);

            var parameters = new ContrastDetectionParams {
                Method = ContrastDetectionMethodEnum.Sobel,
                NoiseReduction = NoiseReductionEnum.None
            };

            var detector = new ContrastDetection();
            var result = await detector.Measure(rendered, parameters, progress: null, token: CancellationToken.None);

            result.AverageContrast.Should().Be(0);
            result.ContrastStdev.Should().Be(0.01);
        }

        private sealed class TestableBahtinovAnalysis : BahtinovAnalysis {
            public TestableBahtinovAnalysis() : base(CreateDummySource(), System.Windows.Media.Colors.Black) { }

            public Accord.Math.Geometry.Line Translate(HoughLine line, int width, int height) {
                var method = typeof(BahtinovAnalysis).GetMethod("TranslateHughLineToLine", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
                return (Accord.Math.Geometry.Line)method!.Invoke(this, new object[] { line, width, height });
            }

            private static BitmapSource CreateDummySource() {
                const int size = 32;
                var pixels = new byte[size * size];
                return BitmapSource.Create(size, size, 96, 96, System.Windows.Media.PixelFormats.Gray8, null, pixels, size);
            }
        }
    }
}
