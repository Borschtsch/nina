#region "copyright"

/*
    Copyright ? 2016 - 2024 Stefan Berg <isbeorn86+NINA@googlemail.com> and the N.I.N.A. contributors

    This file is part of N.I.N.A. - Nighttime Imaging 'N' Astronomy.

    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#endregion "copyright"

using System;
using NINA.Image.Interfaces;

namespace NINA.Image.ImageData {

    public sealed class LRGBArrays : IImageArray {
        // Arrays are immutable after construction to keep dimensions consistent for all consumers.
        public ushort[] Lum { get; }
        public ushort[] Red { get; }
        public ushort[] Green { get; }
        public ushort[] Blue { get; }

        public LRGBArrays(ushort[] lum, ushort[] red, ushort[] green, ushort[] blue) {
            Lum = lum ?? Array.Empty<ushort>();
            Red = red ?? Array.Empty<ushort>();
            Green = green ?? Array.Empty<ushort>();
            Blue = blue ?? Array.Empty<ushort>();
        }

        // Flat view prefers luminance; if not present, we fail fast to avoid silent channel mismatches.
        public ushort[] FlatArray {
            get {
                // Prefer luminance; otherwise fall back to the first available color plane to preserve legacy callers that expect a flat buffer.
                if (Lum.Length > 0) return Lum;
                if (Red.Length > 0) return Red;
                if (Green.Length > 0) return Green;
                if (Blue.Length > 0) return Blue;
                return Array.Empty<ushort>();
            }
        }

        public int[] FlatArrayInt => null;
        public byte[] RAWData => null;
        public string RAWType => null;
    }
}
