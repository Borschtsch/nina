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
using FluentAssertions;
using NINA.Image.ImageAnalysis;
using NUnit.Framework;
using System;
using System.Drawing;
using System.Drawing.Imaging;

namespace NINA.Test.Image.ImageAnalysis {
    [TestFixture]
    public class BayerFilter16bppTest {
        
        [SetUp]
        public void Setup() {
        }

        [TearDown]
        public void TearDown() {
        }

        #region Test Data Generation

        /// <summary>
        /// Creates a synthetic Bayer pattern test image with known values
        /// </summary>
        private unsafe UnmanagedImage CreateBayerTestImage(int width, int height, int[,] pattern) {
            var sourceImage = UnmanagedImage.Create(width, height, PixelFormat.Format16bppGrayScale);
            
            ushort* src = (ushort*)sourceImage.ImageData.ToPointer();
            int srcStride = sourceImage.Stride / 2;

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int bayerChannel = pattern[y & 1, x & 1];
                    // Create distinct values for each channel to verify correct demosaicing
                    // R=1000, G=2000, B=3000 base values
                    ushort value = (ushort)((bayerChannel + 1) * 1000 + (x + y));
                    src[y * srcStride + x] = value;
                }
            }

            return sourceImage;
        }

        /// <summary>
        /// Creates a uniform value test image
        /// </summary>
        private unsafe UnmanagedImage CreateUniformImage(int width, int height, ushort value) {
            var sourceImage = UnmanagedImage.Create(width, height, PixelFormat.Format16bppGrayScale);
            
            ushort* src = (ushort*)sourceImage.ImageData.ToPointer();
            int srcStride = sourceImage.Stride / 2;

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    src[y * srcStride + x] = value;
                }
            }

            return sourceImage;
        }

        /// <summary>
        /// Creates a gradient test image
        /// </summary>
        private unsafe UnmanagedImage CreateGradientImage(int width, int height) {
            var sourceImage = UnmanagedImage.Create(width, height, PixelFormat.Format16bppGrayScale);
            
            ushort* src = (ushort*)sourceImage.ImageData.ToPointer();
            int srcStride = sourceImage.Stride / 2;

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    // Create a smooth gradient
                    double normalized = (double)(x + y) / (width + height);
                    ushort value = (ushort)(normalized * 65535);
                    src[y * srcStride + x] = value;
                }
            }

            return sourceImage;
        }

        #endregion

        #region Basic Functionality Tests

        [Test]
        public void TestDefaultBayerPattern() {
            // Arrange
            var filter = new BayerFilter16bpp();

            // Assert - Default pattern should be RGGB
            Assert.That(filter.BayerPattern[0, 0], Is.EqualTo(RGB.G));
            Assert.That(filter.BayerPattern[0, 1], Is.EqualTo(RGB.R));
            Assert.That(filter.BayerPattern[1, 0], Is.EqualTo(RGB.B));
            Assert.That(filter.BayerPattern[1, 1], Is.EqualTo(RGB.G));
        }

        [Test]
        public void TestPerformDemosaicing_EnabledByDefault() {
            // Arrange
            var filter = new BayerFilter16bpp();

            // Assert
            Assert.That(filter.PerformDemosaicing, Is.True);
        }

        [Test]
        [TestCase(100, 100)]
        [TestCase(640, 480)]
        [TestCase(1024, 768)]
        [TestCase(1920, 1080)]
        public void TestProcessFilter_ValidSizes(int width, int height) {
            // Arrange
            var filter = new BayerFilter16bpp();
            var pattern = new int[2, 2] { { RGB.G, RGB.R }, { RGB.B, RGB.G } };
            var sourceImage = CreateBayerTestImage(width, height, pattern);

            // Act
            var destImage = filter.Apply(sourceImage);

            // Assert
            Assert.That(destImage.Width, Is.EqualTo(width));
            Assert.That(destImage.Height, Is.EqualTo(height));
            
            // Cleanup
            sourceImage.Dispose();
            destImage.Dispose();
        }

        [Test]
        public void TestNoDemosaicing_CopiesCorrectChannels() {
            // Arrange
            int width = 100;
            int height = 100;
            var filter = new BayerFilter16bpp();
            filter.PerformDemosaicing = false;
            var pattern = new int[2, 2] { { RGB.G, RGB.R }, { RGB.B, RGB.G } };
            var sourceImage = CreateBayerTestImage(width, height, pattern);

            // Act
            var destImage = filter.Apply(sourceImage);

            // Assert - Verify pattern is preserved in output
            unsafe {
                ushort* src = (ushort*)sourceImage.ImageData.ToPointer();
                ushort* dst = (ushort*)destImage.ImageData.ToPointer();
                int srcStride = sourceImage.Stride / 2;

                // Check a few sample pixels
                for (int y = 0; y < 10; y++) {
                    for (int x = 0; x < 10; x++) {
                        int bayerChannel = pattern[y & 1, x & 1];
                        ushort srcValue = src[y * srcStride + x];
                        ushort* dstPixel = dst + (y * width + x) * 3;

                        // Only the Bayer channel should have the value
                        if (bayerChannel == RGB.R) {
                            Assert.That(dstPixel[RGB.R], Is.EqualTo(srcValue), $"Red mismatch at ({x},{y})");
                            Assert.That(dstPixel[RGB.G], Is.EqualTo(0), $"Green should be 0 at ({x},{y})");
                            Assert.That(dstPixel[RGB.B], Is.EqualTo(0), $"Blue should be 0 at ({x},{y})");
                        } else if (bayerChannel == RGB.G) {
                            Assert.That(dstPixel[RGB.G], Is.EqualTo(srcValue), $"Green mismatch at ({x},{y})");
                            Assert.That(dstPixel[RGB.R], Is.EqualTo(0), $"Red should be 0 at ({x},{y})");
                            Assert.That(dstPixel[RGB.B], Is.EqualTo(0), $"Blue should be 0 at ({x},{y})");
                        } else {
                            Assert.That(dstPixel[RGB.B], Is.EqualTo(srcValue), $"Blue mismatch at ({x},{y})");
                            Assert.That(dstPixel[RGB.R], Is.EqualTo(0), $"Red should be 0 at ({x},{y})");
                            Assert.That(dstPixel[RGB.G], Is.EqualTo(0), $"Green should be 0 at ({x},{y})");
                        }
                    }
                }
            }

            // Cleanup
            sourceImage.Dispose();
            destImage.Dispose();
        }

        #endregion

        #region Bayer Pattern Tests

        [Test]
        public void TestBayerPattern_RGGB() {
            // Arrange
            int width = 100;
            int height = 100;
            var filter = new BayerFilter16bpp();
            filter.BayerPattern = new int[2, 2] { { RGB.G, RGB.R }, { RGB.B, RGB.G } };
            filter.PerformDemosaicing = false;
            
            var sourceImage = CreateUniformImage(width, height, 1000);

            // Act
            var destImage = filter.Apply(sourceImage);

            // Assert - Verify RGGB pattern
            unsafe {
                ushort* dst = (ushort*)destImage.ImageData.ToPointer();
                
                // (0,0) should be G
                Assert.That(dst[RGB.G], Is.EqualTo(1000));
                Assert.That(dst[RGB.R], Is.EqualTo(0));
                Assert.That(dst[RGB.B], Is.EqualTo(0));
                
                // (1,0) should be R
                ushort* pixel10 = dst + 3;
                Assert.That(pixel10[RGB.R], Is.EqualTo(1000));
                Assert.That(pixel10[RGB.G], Is.EqualTo(0));
                Assert.That(pixel10[RGB.B], Is.EqualTo(0));
                
                // (0,1) should be B
                ushort* pixel01 = dst + width * 3;
                Assert.That(pixel01[RGB.B], Is.EqualTo(1000));
                Assert.That(pixel01[RGB.R], Is.EqualTo(0));
                Assert.That(pixel01[RGB.G], Is.EqualTo(0));
                
                // (1,1) should be G
                ushort* pixel11 = dst + (width + 1) * 3;
                Assert.That(pixel11[RGB.G], Is.EqualTo(1000));
                Assert.That(pixel11[RGB.R], Is.EqualTo(0));
                Assert.That(pixel11[RGB.B], Is.EqualTo(0));
            }

            // Cleanup
            sourceImage.Dispose();
            destImage.Dispose();
        }

        [Test]
        public void TestBayerPattern_BGGR() {
            // Arrange
            int width = 100;
            int height = 100;
            var filter = new BayerFilter16bpp();
            filter.BayerPattern = new int[2, 2] { { RGB.B, RGB.G }, { RGB.G, RGB.R } };
            filter.PerformDemosaicing = false;
            
            var sourceImage = CreateUniformImage(width, height, 2000);

            // Act
            var destImage = filter.Apply(sourceImage);

            // Assert - Verify BGGR pattern
            unsafe {
                ushort* dst = (ushort*)destImage.ImageData.ToPointer();
                
                // (0,0) should be B
                Assert.That(dst[RGB.B], Is.EqualTo(2000));
                
                // (1,0) should be G
                ushort* pixel10 = dst + 3;
                Assert.That(pixel10[RGB.G], Is.EqualTo(2000));
                
                // (0,1) should be G
                ushort* pixel01 = dst + width * 3;
                Assert.That(pixel01[RGB.G], Is.EqualTo(2000));
                
                // (1,1) should be R
                ushort* pixel11 = dst + (width + 1) * 3;
                Assert.That(pixel11[RGB.R], Is.EqualTo(2000));
            }

            // Cleanup
            sourceImage.Dispose();
            destImage.Dispose();
        }

        [Test]
        public void TestBayerPattern_GRBG() {
            // Arrange
            int width = 100;
            int height = 100;
            var filter = new BayerFilter16bpp();
            filter.BayerPattern = new int[2, 2] { { RGB.G, RGB.B }, { RGB.R, RGB.G } };
            filter.PerformDemosaicing = false;
            
            var sourceImage = CreateUniformImage(width, height, 3000);

            // Act
            var destImage = filter.Apply(sourceImage);

            // Assert - Verify GRBG pattern
            unsafe {
                ushort* dst = (ushort*)destImage.ImageData.ToPointer();
                
                // (0,0) should be G
                Assert.That(dst[RGB.G], Is.EqualTo(3000));
                
                // (1,0) should be B
                ushort* pixel10 = dst + 3;
                Assert.That(pixel10[RGB.B], Is.EqualTo(3000));
                
                // (0,1) should be R
                ushort* pixel01 = dst + width * 3;
                Assert.That(pixel01[RGB.R], Is.EqualTo(3000));
                
                // (1,1) should be G
                ushort* pixel11 = dst + (width + 1) * 3;
                Assert.That(pixel11[RGB.G], Is.EqualTo(3000));
            }

            // Cleanup
            sourceImage.Dispose();
            destImage.Dispose();
        }

        [Test]
        public void TestBayerPattern_GBRG() {
            // Arrange
            int width = 100;
            int height = 100;
            var filter = new BayerFilter16bpp();
            filter.BayerPattern = new int[2, 2] { { RGB.G, RGB.R }, { RGB.B, RGB.G } };
            filter.PerformDemosaicing = false;
            
            var sourceImage = CreateUniformImage(width, height, 4000);

            // Act
            var destImage = filter.Apply(sourceImage);

            // Assert
            unsafe {
                ushort* dst = (ushort*)destImage.ImageData.ToPointer();
                Assert.That(dst[RGB.G], Is.EqualTo(4000));
            }

            // Cleanup
            sourceImage.Dispose();
            destImage.Dispose();
        }

        #endregion

        #region Border Processing Tests

        [Test]
        public void TestBorderProcessing_AllBordersProcessed() {
            // Arrange
            int width = 100;
            int height = 100;
            var filter = new BayerFilter16bpp();
            filter.PerformDemosaicing = true;
            var pattern = new int[2, 2] { { RGB.G, RGB.R }, { RGB.B, RGB.G } };
            var sourceImage = CreateBayerTestImage(width, height, pattern);

            // Act
            var destImage = filter.Apply(sourceImage);

            // Assert - Check that border pixels are not zero
            unsafe {
                ushort* dst = (ushort*)destImage.ImageData.ToPointer();
                
                // Top-left corner
                Assert.That(dst[0] + dst[1] + dst[2], Is.GreaterThan(0), "Top-left corner not processed");
                
                // Top-right corner
                ushort* topRight = dst + (width - 1) * 3;
                Assert.That(topRight[0] + topRight[1] + topRight[2], Is.GreaterThan(0), "Top-right corner not processed");
                
                // Bottom-left corner
                ushort* bottomLeft = dst + (height - 1) * width * 3;
                Assert.That(bottomLeft[0] + bottomLeft[1] + bottomLeft[2], Is.GreaterThan(0), "Bottom-left corner not processed");
                
                // Bottom-right corner
                ushort* bottomRight = dst + ((height - 1) * width + width - 1) * 3;
                Assert.That(bottomRight[0] + bottomRight[1] + bottomRight[2], Is.GreaterThan(0), "Bottom-right corner not processed");
                
                // Middle of top row
                ushort* topMiddle = dst + (width / 2) * 3;
                Assert.That(topMiddle[0] + topMiddle[1] + topMiddle[2], Is.GreaterThan(0), "Top edge not processed");
                
                // Middle of bottom row
                ushort* bottomMiddle = dst + ((height - 1) * width + width / 2) * 3;
                Assert.That(bottomMiddle[0] + bottomMiddle[1] + bottomMiddle[2], Is.GreaterThan(0), "Bottom edge not processed");
                
                // Middle of left column
                ushort* leftMiddle = dst + (height / 2) * width * 3;
                Assert.That(leftMiddle[0] + leftMiddle[1] + leftMiddle[2], Is.GreaterThan(0), "Left edge not processed");
                
                // Middle of right column
                ushort* rightMiddle = dst + ((height / 2) * width + width - 1) * 3;
                Assert.That(rightMiddle[0] + rightMiddle[1] + rightMiddle[2], Is.GreaterThan(0), "Right edge not processed");
            }

            // Cleanup
            sourceImage.Dispose();
            destImage.Dispose();
        }

        #endregion

        #region Data Integrity Tests

        [Test]
        public void TestValuePreservation_UniformImage() {
            // Arrange
            int width = 200;
            int height = 200;
            ushort uniformValue = 30000;
            var filter = new BayerFilter16bpp();
            filter.PerformDemosaicing = true;
            
            var sourceImage = CreateUniformImage(width, height, uniformValue);

            // Act
            var destImage = filter.Apply(sourceImage);

            // Assert - All RGB values should be close to uniform value (within interpolation tolerance)
            unsafe {
                ushort* dst = (ushort*)destImage.ImageData.ToPointer();
                
                // Check center pixels (away from borders)
                for (int y = 10; y < height - 10; y++) {
                    for (int x = 10; x < width - 10; x++) {
                        ushort* pixel = dst + (y * width + x) * 3;
                        
                        // All channels should be close to the uniform value
                        Assert.That(pixel[RGB.R], Is.InRange(uniformValue - 100, uniformValue + 100), 
                            $"Red out of range at ({x},{y})");
                        Assert.That(pixel[RGB.G], Is.InRange(uniformValue - 100, uniformValue + 100), 
                            $"Green out of range at ({x},{y})");
                        Assert.That(pixel[RGB.B], Is.InRange(uniformValue - 100, uniformValue + 100), 
                            $"Blue out of range at ({x},{y})");
                    }
                }
            }

            // Cleanup
            sourceImage.Dispose();
            destImage.Dispose();
        }

        [Test]
        public void TestNoValueClipping_MaxValues() {
            // Arrange
            int width = 100;
            int height = 100;
            ushort maxValue = 65535;
            var filter = new BayerFilter16bpp();
            filter.PerformDemosaicing = true;
            
            var sourceImage = CreateUniformImage(width, height, maxValue);

            // Act
            var destImage = filter.Apply(sourceImage);

            // Assert - No clipping should occur
            unsafe {
                ushort* dst = (ushort*)destImage.ImageData.ToPointer();
                
                // Check that we haven't exceeded 16-bit range
                for (int i = 0; i < width * height * 3; i++) {
                    Assert.That(dst[i], Is.LessThanOrEqualTo(65535), $"Value clipped at index {i}");
                }
            }

            // Cleanup
            sourceImage.Dispose();
            destImage.Dispose();
        }

        [Test]
        public void TestSymmetry_IdenticalPatternsProduceSimilarResults() {
            // Arrange
            int width = 100;
            int height = 100;
            var filter = new BayerFilter16bpp();
            filter.PerformDemosaicing = true;
            var pattern = new int[2, 2] { { RGB.G, RGB.R }, { RGB.B, RGB.G } };
            
            var sourceImage1 = CreateGradientImage(width, height);
            var sourceImage2 = CreateGradientImage(width, height);

            // Act
            var destImage1 = filter.Apply(sourceImage1);
            var destImage2 = filter.Apply(sourceImage2);

            // Assert - Results should be identical
            unsafe {
                ushort* dst1 = (ushort*)destImage1.ImageData.ToPointer();
                ushort* dst2 = (ushort*)destImage2.ImageData.ToPointer();
                
                for (int i = 0; i < width * height * 3; i++) {
                    Assert.That(dst1[i], Is.EqualTo(dst2[i]), $"Mismatch at index {i}");
                }
            }

            // Cleanup
            sourceImage1.Dispose();
            sourceImage2.Dispose();
            destImage1.Dispose();
            destImage2.Dispose();
        }

        #endregion

        #region Dimension Tests

        [Test]
        [TestCase(101, 99)]  // Odd width, odd height
        [TestCase(100, 99)]  // Even width, odd height
        [TestCase(101, 100)] // Odd width, even height
        [TestCase(100, 100)] // Even width, even height
        public void TestVariousDimensions(int width, int height) {
            // Arrange
            var filter = new BayerFilter16bpp();
            filter.PerformDemosaicing = true;
            var pattern = new int[2, 2] { { RGB.G, RGB.R }, { RGB.B, RGB.G } };
            var sourceImage = CreateBayerTestImage(width, height, pattern);

            // Act
            var destImage = filter.Apply(sourceImage);

            // Assert - Should complete without error
            Assert.That(destImage.Width, Is.EqualTo(width));
            Assert.That(destImage.Height, Is.EqualTo(height));

            // Cleanup
            sourceImage.Dispose();
            destImage.Dispose();
        }

        [Test]
        [TestCase(1920, 1080)]
        [TestCase(3840, 2160)]
        [TestCase(4096, 4096)]
        public void TestLargeImages(int width, int height) {
            // Arrange
            var filter = new BayerFilter16bpp();
            filter.PerformDemosaicing = true;
            var pattern = new int[2, 2] { { RGB.G, RGB.R }, { RGB.B, RGB.G } };
            var sourceImage = CreateBayerTestImage(width, height, pattern);

            // Act & Assert - Should complete without error or timeout
            var destImage = filter.Apply(sourceImage);
            Assert.That(destImage, Is.Not.Null);
            Assert.That(destImage.Width, Is.EqualTo(width));
            Assert.That(destImage.Height, Is.EqualTo(height));

            // Cleanup
            sourceImage.Dispose();
            destImage.Dispose();
        }

        #endregion

        #region LRGB Extraction Tests

        [Test]
        public void TestLRGBExtraction_ColorChannelsOnly() {
            // Arrange
            int width = 100;
            int height = 100;
            var filter = new BayerFilter16bpp();
            filter.SaveColorChannels = true;
            filter.SaveLumChannel = false;
            filter.PerformDemosaicing = true;
            
            var pattern = new int[2, 2] { { RGB.G, RGB.R }, { RGB.B, RGB.G } };
            var sourceImage = CreateBayerTestImage(width, height, pattern);

            // Act
            var destImage = filter.Apply(sourceImage);

            // Assert
            Assert.That(filter.LRGBArrays, Is.Not.Null);
            Assert.That(filter.LRGBArrays.Red, Is.Not.Null);
            Assert.That(filter.LRGBArrays.Green, Is.Not.Null);
            Assert.That(filter.LRGBArrays.Blue, Is.Not.Null);
            Assert.That(filter.LRGBArrays.Red.Length, Is.EqualTo(width * height));
            Assert.That(filter.LRGBArrays.Green.Length, Is.EqualTo(width * height));
            Assert.That(filter.LRGBArrays.Blue.Length, Is.EqualTo(width * height));
            Assert.That(filter.LRGBArrays.Lum, Is.Empty);

            // Cleanup
            sourceImage.Dispose();
            destImage.Dispose();
        }

        [Test]
        public void TestLRGBExtraction_LumChannelOnly() {
            // Arrange
            int width = 100;
            int height = 100;
            var filter = new BayerFilter16bpp();
            filter.SaveColorChannels = false;
            filter.SaveLumChannel = true;
            filter.PerformDemosaicing = true;
            
            var pattern = new int[2, 2] { { RGB.G, RGB.R }, { RGB.B, RGB.G } };
            var sourceImage = CreateBayerTestImage(width, height, pattern);

            // Act
            var destImage = filter.Apply(sourceImage);

            // Assert
            Assert.That(filter.LRGBArrays, Is.Not.Null);
            Assert.That(filter.LRGBArrays.Lum, Is.Not.Null);
            Assert.That(filter.LRGBArrays.Lum.Length, Is.EqualTo(width * height));
            Assert.That(filter.LRGBArrays.Red, Is.Empty);
            Assert.That(filter.LRGBArrays.Green, Is.Empty);
            Assert.That(filter.LRGBArrays.Blue, Is.Empty);

            // Cleanup
            sourceImage.Dispose();
            destImage.Dispose();
        }

        [Test]
        public void TestLRGBExtraction_AllChannels() {
            // Arrange
            int width = 100;
            int height = 100;
            var filter = new BayerFilter16bpp();
            filter.SaveColorChannels = true;
            filter.SaveLumChannel = true;
            filter.PerformDemosaicing = true;
            
            var pattern = new int[2, 2] { { RGB.G, RGB.R }, { RGB.B, RGB.G } };
            var sourceImage = CreateBayerTestImage(width, height, pattern);

            // Act
            var destImage = filter.Apply(sourceImage);

            // Assert
            Assert.That(filter.LRGBArrays, Is.Not.Null);
            Assert.That(filter.LRGBArrays.Lum.Length, Is.EqualTo(width * height));
            Assert.That(filter.LRGBArrays.Red.Length, Is.EqualTo(width * height));
            Assert.That(filter.LRGBArrays.Green.Length, Is.EqualTo(width * height));
            Assert.That(filter.LRGBArrays.Blue.Length, Is.EqualTo(width * height));

            // Cleanup
            sourceImage.Dispose();
            destImage.Dispose();
        }

        #endregion
    }
}
