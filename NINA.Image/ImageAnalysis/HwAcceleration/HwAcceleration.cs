using NINA.Core.Utility;
using Silk.NET.OpenCL;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace NINA.Image.ImageAnalysis.HwAcceleration {

    /// <summary>
    /// Opaque handle for an OpenCL program.
    /// </summary>
    public readonly struct ClProgram {
        internal readonly nint Handle;
        internal ClProgram(nint handle) => Handle = handle;
        public bool IsValid => Handle != 0;
    }

    /// <summary>
    /// Opaque handle for an OpenCL kernel.
    /// </summary>
    public readonly struct ClKernel {
        internal readonly nint Handle;
        internal ClKernel(nint handle) => Handle = handle;
        public bool IsValid => Handle != 0;
    }

    /// <summary>
    /// Opaque handle for an OpenCL buffer.
    /// </summary>
    public readonly struct ClBuffer {
        internal readonly nint Handle;
        internal ClBuffer(nint handle) => Handle = handle;
        public bool IsValid => Handle != 0;
    }

    /// <summary>
    /// Internal descriptor for an OpenCL device handle.
    /// Not exposed to application code.
    /// </summary>
    internal sealed class OpenClDeviceInfo {
        internal nint PlatformHandle { get; }
        internal nint DeviceHandle { get; }
        internal DeviceType DeviceType { get; }
        internal string Name { get; }
        internal string Vendor { get; }
        internal string Version { get; }

        internal OpenClDeviceInfo(
            nint platform,
            nint device,
            DeviceType deviceType,
            string name,
            string vendor,
            string version) {

            PlatformHandle = platform;
            DeviceHandle = device;
            DeviceType = deviceType;
            Name = name;
            Vendor = vendor;
            Version = version;
        }

        public override string ToString() =>
            $"{Name} | {Vendor} | {Version} | {DeviceType}";
    }

    /// <summary>
    /// Simple OpenCL 1.1 accelerator.
    /// 
    /// - Singleton: use OpenClAccelerator.Instance
    /// - Hides Silk.NET/OpenCL types from public API
    /// - Always blocking operations (no async / flags)
    /// - Programs/kernels are tracked and released on Dispose
    /// - Buffers must be released explicitly by caller
    /// </summary>
    public sealed class OpenClAccelerator : IDisposable {
        private readonly CL _cl = CL.GetApi();

        private nint _context;
        private nint _queue;
        private nint _device;

        private bool _initialized;
        private bool _disposed;

        private readonly List<nint> _programs = new();
        private readonly List<nint> _kernels = new();

        public static OpenClAccelerator Instance { get; } = new OpenClAccelerator();

        public bool IsInitialized => _initialized;

        public string DeviceName { get; private set; } = string.Empty;
        public string DeviceVendor { get; private set; } = string.Empty;
        public string DeviceVersion { get; private set; } = string.Empty;

        private OpenClAccelerator() {
            Init();
        }

        /// <summary>
        /// Create an accelerator bound to a specific OpenCL device.
        /// Used by the multi-device executor. Not exposed to callers.
        /// </summary>
        internal OpenClAccelerator(OpenClDeviceInfo deviceInfo) {
            InitFromExistingDevice(deviceInfo.DeviceHandle);
        }

        // ---------------------------------------------------------------------
        // Initialization
        // ---------------------------------------------------------------------
        private unsafe void Init() {
            if (_initialized)
                return;

            // 1. Query platforms
            uint numPlatforms = 0;
            int err = _cl.GetPlatformIDs(0, null, &numPlatforms);
            if (err != (int)ErrorCodes.Success || numPlatforms == 0) {
                Logger.Warning("OpenCL: no platforms found.");
                return;
            }

            var platforms = stackalloc nint[(int)numPlatforms];
            err = _cl.GetPlatformIDs(numPlatforms, platforms, null);
            if (err != (int)ErrorCodes.Success) {
                Logger.Warning($"OpenCL: GetPlatformIDs failed with {err}.");
                return;
            }

            // 2. Choose device: GPU preferred, then CPU
            _device = FindFirstDevice(DeviceType.Gpu, platforms, numPlatforms);
            if (_device == 0) {
                _device = FindFirstDevice(DeviceType.Cpu, platforms, numPlatforms);
            }

            if (_device == 0) {
                Logger.Warning("OpenCL: no suitable device (GPU/CPU) found.");
                return;
            }

            InitFromExistingDevice(_device);
        }

        /// <summary>
        /// Initialize this accelerator for an already chosen device handle.
        /// </summary>
        private unsafe void InitFromExistingDevice(nint device) {
            if (_initialized)
                return;

            _device = device;

            // 3. Create context
            int error;
            _context = _cl.CreateContext(null, 1, ref _device, null, null, &error);
            if (error != (int)ErrorCodes.Success || _context == 0) {
                Logger.Warning($"OpenCL: CreateContext failed with {error}.");
                return;
            }

            // 4. Create command queue (OpenCL 1.1 API)
            _queue = _cl.CreateCommandQueue(_context, _device, CommandQueueProperties.None, &error);
            if (error != (int)ErrorCodes.Success || _queue == 0) {
                Logger.Warning($"OpenCL: CreateCommandQueue failed with {error}.");
                _cl.ReleaseContext(_context);
                _context = 0;
                return;
            }

            // 5. Log device info
            DeviceName = GetDeviceInfoString(_device, DeviceInfo.Name);
            DeviceVendor = GetDeviceInfoString(_device, DeviceInfo.Vendor);
            DeviceVersion = GetDeviceInfoString(_device, DeviceInfo.Version);

            Logger.Info($"OpenCL accelerator: {DeviceName} | Vendor: {DeviceVendor} | Version: {DeviceVersion}");

            _initialized = true;
        }

        private unsafe nint FindFirstDevice(DeviceType type, nint* platforms, uint numPlatforms) {
            for (int i = 0; i < numPlatforms; i++) {
                nint platform = platforms[i];
                uint numDevices = 0;
                int err = _cl.GetDeviceIDs(platform, type, 0, null, &numDevices);
                if (err != (int)ErrorCodes.Success || numDevices == 0)
                    continue;

                var devices = stackalloc nint[(int)numDevices];
                err = _cl.GetDeviceIDs(platform, type, numDevices, devices, null);
                if (err != (int)ErrorCodes.Success || numDevices == 0)
                    continue;

                return devices[0];
            }
            return 0;
        }

        private unsafe string GetDeviceInfoString(nint device, DeviceInfo info) {
            nuint size = 0;
            int err = _cl.GetDeviceInfo(device, info, 0, null, &size);
            if (err != (int)ErrorCodes.Success || size == 0)
                return string.Empty;

            byte* buffer = stackalloc byte[(int)size];
            err = _cl.GetDeviceInfo(device, info, size, buffer, null);
            if (err != (int)ErrorCodes.Success)
                return string.Empty;

            return Marshal.PtrToStringAnsi((IntPtr)buffer) ?? string.Empty;
        }

        private static unsafe string GetDeviceInfoString(CL cl, nint device, DeviceInfo info) {
            nuint size = 0;
            int err = cl.GetDeviceInfo(device, info, 0, null, &size);
            if (err != (int)ErrorCodes.Success || size == 0)
                return string.Empty;

            byte* buffer = stackalloc byte[(int)size];
            err = cl.GetDeviceInfo(device, info, size, buffer, null);
            if (err != (int)ErrorCodes.Success)
                return string.Empty;

            return Marshal.PtrToStringAnsi((IntPtr)buffer) ?? string.Empty;
        }

        /// <summary>
        /// Enumerate all OpenCL devices (GPU, ACCELERATOR, CPU) on all platforms.
        /// Internal only; used by the multi-device executor.
        /// </summary>
        internal static unsafe IReadOnlyList<OpenClDeviceInfo> EnumerateAllDevices() {
            var cl = CL.GetApi();
            var result = new List<OpenClDeviceInfo>();

            uint numPlatforms = 0;
            int err = cl.GetPlatformIDs(0, null, &numPlatforms);
            if (err != (int)ErrorCodes.Success || numPlatforms == 0)
                return result;

            var platforms = stackalloc nint[(int)numPlatforms];
            err = cl.GetPlatformIDs(numPlatforms, platforms, null);
            if (err != (int)ErrorCodes.Success)
                return result;

            DeviceType[] deviceTypes = {
                DeviceType.Gpu,
                DeviceType.Accelerator,
                DeviceType.Cpu
            };

            for (int i = 0; i < numPlatforms; i++) {
                nint platform = platforms[i];

                foreach (var type in deviceTypes) {
                    uint numDevices = 0;
                    err = cl.GetDeviceIDs(platform, type, 0, null, &numDevices);
                    if (err != (int)ErrorCodes.Success || numDevices == 0)
                        continue;

                    var devices = stackalloc nint[(int)numDevices];
                    err = cl.GetDeviceIDs(platform, type, numDevices, devices, null);
                    if (err != (int)ErrorCodes.Success)
                        continue;

                    for (int d = 0; d < numDevices; d++) {
                        nint dev = devices[d];
                        string name = GetDeviceInfoString(cl, dev, DeviceInfo.Name);
                        string vendor = GetDeviceInfoString(cl, dev, DeviceInfo.Vendor);
                        string version = GetDeviceInfoString(cl, dev, DeviceInfo.Version);

                        result.Add(new OpenClDeviceInfo(platform, dev, type, name, vendor, version));
                    }
                }
            }

            return result;
        }

        private void EnsureInitialized() {
            if (!_initialized)
                throw new InvalidOperationException("OpenCL accelerator is not initialized.");
        }

        // ---------------------------------------------------------------------
        // Program + Kernel
        // ---------------------------------------------------------------------

        /// <summary>
        /// Create and build an OpenCL program from source.
        /// Program is tracked and released when the accelerator is disposed.
        /// </summary>
        public unsafe ClProgram CreateProgramFromSource(string source) {
            EnsureInitialized();

            byte[] srcBytes = System.Text.Encoding.ASCII.GetBytes(source);
            int error;

            fixed (byte* pSrc = srcBytes) {
                byte** strings = stackalloc byte*[1];
                strings[0] = pSrc;
                nuint* lengths = stackalloc nuint[1];
                lengths[0] = (nuint)srcBytes.Length;

                nint program = _cl.CreateProgramWithSource(
                    _context,
                    1,
                    strings,
                    lengths,
                    &error);

                if (error != (int)ErrorCodes.Success || program == 0)
                    throw new Exception($"OpenCL: CreateProgramWithSource failed with {error}.");

                error = _cl.BuildProgram(program, 0, null, (byte*)IntPtr.Zero, null, null);
                if (error != (int)ErrorCodes.Success)
                    throw new Exception($"OpenCL: BuildProgram failed with {error}.");

                _programs.Add(program);
                return new ClProgram(program);
            }
        }

        /// <summary>
        /// Create a kernel from a program.
        /// Kernel is tracked and released when the accelerator is disposed.
        /// </summary>
        public unsafe ClKernel CreateKernel(ClProgram program, string kernelName) {
            EnsureInitialized();
            if (!program.IsValid)
                throw new ArgumentException("Invalid program handle.", nameof(program));

            byte[] nameBytes = System.Text.Encoding.ASCII.GetBytes(kernelName);
            int error;
            fixed (byte* pName = nameBytes) {
                nint kernel = _cl.CreateKernel(program.Handle, pName, &error);
                if (error != (int)ErrorCodes.Success || kernel == 0)
                    throw new Exception($"OpenCL: CreateKernel('{kernelName}') failed with {error}.");

                _kernels.Add(kernel);
                return new ClKernel(kernel);
            }
        }

        // Optional explicit release if you want to reclaim early
        public void ReleaseProgram(ClProgram program) {
            if (!program.IsValid)
                return;
            _cl.ReleaseProgram(program.Handle);
            _programs.Remove(program.Handle);
        }

        public void ReleaseKernel(ClKernel kernel) {
            if (!kernel.IsValid)
                return;
            _cl.ReleaseKernel(kernel.Handle);
            _kernels.Remove(kernel.Handle);
        }

        // ---------------------------------------------------------------------
        // Buffers (caller explicitly releases)
        // ---------------------------------------------------------------------

        /// <summary>
        /// Create a read-only buffer and copy the data into device memory.
        /// Blocking, always.
        /// </summary>
        public unsafe ClBuffer CreateSourceBufferAndCopy<T>(ReadOnlySpan<T> data) where T : unmanaged {
            EnsureInitialized();

            nuint sizeInBytes = (nuint)(data.Length * sizeof(T));

            int err;
            T* hostPtr;
            fixed (T* pData = data) {
                hostPtr = pData;
                nint buffer = _cl.CreateBuffer(
                    _context,
                    MemFlags.ReadOnly | MemFlags.CopyHostPtr,
                    sizeInBytes,
                    hostPtr,
                    &err);

                if (err != (int)ErrorCodes.Success || buffer == 0)
                    throw new Exception($"OpenCL: CreateBuffer(source) failed with {err}.");

                return new ClBuffer(buffer);
            }
        }

        /// <summary>
        /// Create a write-only buffer of given element count (no initial data).
        /// </summary>
        public unsafe ClBuffer CreateWriteBuffer<T>(int elementCount) where T : unmanaged {
            EnsureInitialized();
            if (elementCount <= 0)
                throw new ArgumentOutOfRangeException(nameof(elementCount));

            nuint sizeInBytes = (nuint)(elementCount * sizeof(T));

            int err;
            nint buffer = _cl.CreateBuffer(
                _context,
                MemFlags.WriteOnly,
                sizeInBytes,
                null,
                &err);

            if (err != (int)ErrorCodes.Success || buffer == 0)
                throw new Exception($"OpenCL: CreateBuffer(write) failed with {err}.");

            return new ClBuffer(buffer);
        }

        /// <summary>
        /// Blocking write into an existing buffer from host memory.
        /// </summary>
        public unsafe void WriteBuffer<T>(ClBuffer buffer, ReadOnlySpan<T> data) where T : unmanaged {
            EnsureInitialized();
            if (!buffer.IsValid)
                throw new ArgumentException("Invalid buffer handle.", nameof(buffer));

            nuint sizeInBytes = (nuint)(data.Length * sizeof(T));

            fixed (T* pData = data) {
                int err = _cl.EnqueueWriteBuffer(
                    _queue,
                    buffer.Handle,
                    true,          // blocking
                    0,
                    sizeInBytes,
                    pData,
                    0,
                    null,
                    null);

                if (err != (int)ErrorCodes.Success)
                    throw new Exception($"OpenCL: EnqueueWriteBuffer failed with {err}.");
            }
        }

        /// <summary>
        /// Blocking read from buffer into destination span.
        /// </summary>
        public unsafe void ReadBuffer<T>(ClBuffer buffer, Span<T> destination) where T : unmanaged {
            EnsureInitialized();
            if (!buffer.IsValid)
                throw new ArgumentException("Invalid buffer handle.", nameof(buffer));

            nuint sizeInBytes = (nuint)(destination.Length * sizeof(T));

            fixed (T* pDst = destination) {
                int err = _cl.EnqueueReadBuffer(
                    _queue,
                    buffer.Handle,
                    true,          // blocking
                    0,
                    sizeInBytes,
                    pDst,
                    0,
                    null,
                    null);

                if (err != (int)ErrorCodes.Success)
                    throw new Exception($"OpenCL: EnqueueReadBuffer failed with {err}.");
            }
        }

        public unsafe void ReadBufferRegion<T>(ClBuffer buffer, int elementOffset, Span<T> destination)
            where T : unmanaged {
            EnsureInitialized();
            if (!buffer.IsValid)
                throw new ArgumentException("Invalid buffer handle.", nameof(buffer));

            nuint sizeInBytes = (nuint)(destination.Length * sizeof(T));
            nuint offsetInBytes = (nuint)(elementOffset * sizeof(T));

            fixed (T* pDst = destination) {
                int err = _cl.EnqueueReadBuffer(
                    _queue,
                    buffer.Handle,
                    true,               // blocking
                    offsetInBytes,
                    sizeInBytes,
                    pDst,
                    0,
                    null,
                    null);

                if (err != (int)ErrorCodes.Success)
                    throw new Exception($"OpenCL: EnqueueReadBuffer(region) failed with {err}.");
            }
        }

        /// <summary>
        /// Release a buffer. After this, the handle becomes invalid.
        /// </summary>
        public void ReleaseBuffer(ClBuffer buffer) {
            if (!buffer.IsValid)
                return;
            _cl.ReleaseMemObject(buffer.Handle);
        }

        // ---------------------------------------------------------------------
        // Kernel arguments + execution
        // ---------------------------------------------------------------------

        /// <summary>
        /// Set kernel arg to a buffer.
        /// </summary>
        public unsafe void SetKernelArg(ClKernel kernel, uint index, ClBuffer buffer) {
            EnsureInitialized();
            if (!kernel.IsValid)
                throw new ArgumentException("Invalid kernel handle.", nameof(kernel));
            if (!buffer.IsValid)
                throw new ArgumentException("Invalid buffer handle.", nameof(buffer));

            nint handle = buffer.Handle;
            int err = _cl.SetKernelArg(kernel.Handle, index, (nuint)sizeof(nint), &handle);
            if (err != (int)ErrorCodes.Success)
                throw new Exception($"OpenCL: SetKernelArg(buffer, index={index}) failed with {err}.");
        }

        /// <summary>
        /// Set kernel arg to a scalar value.
        /// </summary>
        public unsafe void SetKernelArg<T>(ClKernel kernel, uint index, T value) where T : unmanaged {
            EnsureInitialized();
            if (!kernel.IsValid)
                throw new ArgumentException("Invalid kernel handle.", nameof(kernel));

            int err;
            T v = value;
            err = _cl.SetKernelArg(kernel.Handle, index, (nuint)sizeof(T), &v);
            if (err != (int)ErrorCodes.Success)
                throw new Exception($"OpenCL: SetKernelArg(scalar, index={index}) failed with {err}.");
        }

        /// <summary>
        /// Run a 1D kernel with the given global size.
        /// Blocking; returns after completion.
        /// </summary>
        public unsafe void RunKernel1D(ClKernel kernel, int globalSize) {
            EnsureInitialized();
            if (!kernel.IsValid)
                throw new ArgumentException("Invalid kernel handle.", nameof(kernel));
            if (globalSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(globalSize));

            nuint size = (nuint)globalSize;
            int err;
            err = _cl.EnqueueNdrangeKernel(
                _queue,
                kernel.Handle,
                1,
                null,
                &size,
                null,
                0,
                null,
                null);

            if (err != (int)ErrorCodes.Success)
                throw new Exception($"OpenCL: EnqueueNDRangeKernel(1D) failed with {err}.");

            _cl.Finish(_queue);
        }

        /// <summary>
        /// Run a 2D kernel with the given global sizes.
        /// Blocking; returns after completion.
        /// </summary>
        public unsafe void RunKernel2D(ClKernel kernel, int globalSizeX, int globalSizeY) {
            EnsureInitialized();
            if (!kernel.IsValid)
                throw new ArgumentException("Invalid kernel handle.", nameof(kernel));
            if (globalSizeX <= 0 || globalSizeY <= 0)
                throw new ArgumentOutOfRangeException("Global sizes must be > 0.");

            nuint[] sizes = { (nuint)globalSizeX, (nuint)globalSizeY };

            fixed (nuint* pSizes = sizes) {
                int err = _cl.EnqueueNdrangeKernel(
                    _queue,
                    kernel.Handle,
                    2,
                    null,      // no offset; handle offset in kernel index math
                    pSizes,
                    null,
                    0,
                    null,
                    null);

                if (err != (int)ErrorCodes.Success)
                    throw new Exception($"OpenCL: EnqueueNDRangeKernel(2D) failed with {err}.");
            }

            _cl.Finish(_queue);
        }

        // ---------------------------------------------------------------------
        // Dispose
        // ---------------------------------------------------------------------

        public void Dispose() {
            if (_disposed)
                return;

            _disposed = true;

            // Release kernels
            foreach (var k in _kernels) {
                _cl.ReleaseKernel(k);
            }
            _kernels.Clear();

            // Release programs
            foreach (var p in _programs) {
                _cl.ReleaseProgram(p);
            }
            _programs.Clear();

            if (_queue != 0) {
                _cl.ReleaseCommandQueue(_queue);
                _queue = 0;
            }

            if (_context != 0) {
                _cl.ReleaseContext(_context);
                _context = 0;
            }

            _device = 0;
            _initialized = false;
        }
    }

    /// <summary>
    /// Schedules 2D image work across multiple OpenCL devices + optional CPU worker.
    /// Public API is application-agnostic: it only knows about width/height and row ranges.
    /// </summary>
    public sealed class MultiDevice2DExecutor : IDisposable {
        private const int PerfWindowSize = 3;

        private sealed class GpuWorker : IDisposable {
            public OpenClAccelerator Accelerator { get; }
            public ClProgram Program { get; }
            public ClKernel Kernel { get; }

            // Per-worker buffers
            public ClBuffer SrcBuffer;
            public int SrcCapacity;             // number of elements, not bytes
            public ClBuffer DstBuffer;
            public int DstCapacity;             // number of elements

            public GpuWorker(OpenClAccelerator accelerator, ClProgram program, ClKernel kernel) {
                Accelerator = accelerator;
                Program = program;
                Kernel = kernel;
                SrcBuffer = default;
                DstBuffer = default;
                SrcCapacity = 0;
                DstCapacity = 0;
            }

            public void Dispose() {
                try {
                    if (SrcBuffer.IsValid) {
                        Accelerator.ReleaseBuffer(SrcBuffer);
                        SrcBuffer = default;
                        SrcCapacity = 0;
                    }
                    if (DstBuffer.IsValid) {
                        Accelerator.ReleaseBuffer(DstBuffer);
                        DstBuffer = default;
                        DstCapacity = 0;
                    }

                    Accelerator.ReleaseKernel(Kernel);
                    Accelerator.ReleaseProgram(Program);
                    Accelerator.Dispose();
                } catch {
                    // swallow on shutdown
                }
            }
        }

        /// <summary>
        /// Per-worker performance model with sliding window over last N runs.
        /// </summary>
        private sealed class WorkerPerf {
            private readonly int _windowSize;
            private readonly Queue<(int Rows, double Seconds)> _samples = new();
            private long _sumRows;
            private double _sumSeconds;

            public WorkerPerf(int windowSize) {
                _windowSize = windowSize;
            }

            public double EstimatedRowsPerSecond =>
                _sumSeconds <= 0 || _sumRows <= 0 ? 0.0 : _sumRows / _sumSeconds;

            /// <summary>
            /// Add a new run sample for this worker (rows processed in 'seconds').
            /// Keeps only last _windowSize runs.
            /// </summary>
            public void AddSample(int rows, double seconds) {
                if (rows <= 0 || seconds <= 0)
                    return;

                _samples.Enqueue((rows, seconds));
                _sumRows += rows;
                _sumSeconds += seconds;

                while (_samples.Count > _windowSize) {
                    var old = _samples.Dequeue();
                    _sumRows -= old.Rows;
                    _sumSeconds -= old.Seconds;
                }
            }

            public void Reset() {
                _samples.Clear();
                _sumRows = 0;
                _sumSeconds = 0;
            }
        }

        private readonly List<GpuWorker> _gpuWorkers = new();
        private readonly List<WorkerPerf> _gpuPerf = new();
        private readonly WorkerPerf _cpuPerf;
        private readonly object _perfLock = new object();

        private readonly bool _includeCpuWorker = true;

        public int GpuWorkerCount => _gpuWorkers.Count;

        public MultiDevice2DExecutor(
            string kernelSource,
            string kernelName,
            int maxDevices = -1) {

            _cpuPerf = new WorkerPerf(PerfWindowSize);

            var devices = OpenClAccelerator.EnumerateAllDevices();
            if (devices == null || devices.Count == 0)
                return;

            var gpuLike = devices
                .Where(d => d.DeviceType == DeviceType.Gpu || d.DeviceType == DeviceType.Accelerator)
                .ToList();

            var selected = gpuLike.Count > 0 ? gpuLike : devices.ToList();
            if (maxDevices > 0)
                selected = selected.Take(maxDevices).ToList();

            foreach (var dev in selected) {
                var acc = new OpenClAccelerator(dev);
                if (!acc.IsInitialized) {
                    acc.Dispose();
                    continue;
                }

                try {
                    var prog = acc.CreateProgramFromSource(kernelSource);
                    var kernel = acc.CreateKernel(prog, kernelName);
                    _gpuWorkers.Add(new GpuWorker(acc, prog, kernel));
                    _gpuPerf.Add(new WorkerPerf(PerfWindowSize));
                } catch {
                    acc.Dispose();
                }
            }
        }

        public void ResetPerformanceHistory() {
            lock (_perfLock) {
                foreach (var p in _gpuPerf)
                    p.Reset();
                _cpuPerf.Reset();
            }
        }

        private static void EnsureBuffersForWorker(
            GpuWorker worker,
            int totalSrcElements,
            int totalDstElements) {

            var acc = worker.Accelerator;

            // src
            if (worker.SrcCapacity < totalSrcElements || !worker.SrcBuffer.IsValid) {
                if (worker.SrcBuffer.IsValid)
                    acc.ReleaseBuffer(worker.SrcBuffer);

                worker.SrcBuffer = acc.CreateWriteBuffer<ushort>(totalSrcElements);
                worker.SrcCapacity = totalSrcElements;
            }

            // dst
            if (worker.DstCapacity < totalDstElements || !worker.DstBuffer.IsValid) {
                if (worker.DstBuffer.IsValid)
                    acc.ReleaseBuffer(worker.DstBuffer);

                worker.DstBuffer = acc.CreateWriteBuffer<ushort>(totalDstElements);
                worker.DstCapacity = totalDstElements;
            }
        }

        /// <summary>
        /// Execute a 2D kernel on [borderTop, height - borderBottom) rows,
        /// splitting rows between multiple GPUs and optional CPU.
        /// Work is distributed proportionally to measured throughput
        /// (rows/second) of each worker.
        /// </summary>
        public unsafe void Execute(
            int width,
            int height,
            int borderTop,
            int borderBottom,
            ushort* srcPtr,
            ushort* dstPtr,
            int srcStride,   // in ushorts
            Action<int, int>? cpuWorker,
            Action<OpenClAccelerator, ClKernel, ClBuffer, ClBuffer, int, int> gpuWorker) {

            if (gpuWorker == null)
                throw new ArgumentNullException(nameof(gpuWorker));

            int innerStartY = borderTop;
            int innerEndY = height - borderBottom; // exclusive
            if (innerEndY <= innerStartY)
                return;

            int innerRows = innerEndY - innerStartY;

            int gpuCount = _gpuWorkers.Count;
            bool useCpu = _includeCpuWorker && cpuWorker != null;
            int workerCount = gpuCount + (useCpu ? 1 : 0);

            // CPU-only fallback
            if (workerCount == 0) {
                cpuWorker?.Invoke(innerStartY, innerEndY);
                return;
            }

            int baseRows = innerRows / workerCount;
            int remainder = innerRows % workerCount;

            int currentY = innerStartY;

            int totalSrcElements = srcStride * height;      // ushort elements
            int totalDstElements = width * height * 3;      // ushort elements

            var tasks = new List<Task>(workerCount);

            // 1) GPU workers (each with its own buffers)
            for (int i = 0; i < gpuCount; i++) {
                int rows = baseRows;
                if (remainder > 0) {
                    rows++;
                    remainder--;
                }

                if (rows <= 0)
                    continue;

                int startY = currentY;
                int endY = startY + rows;
                currentY = endY;

                var worker = _gpuWorkers[i];

                tasks.Add(Task.Run(() => {
                    try {
                        // allocate/resize once per worker if needed
                        EnsureBuffersForWorker(worker, totalSrcElements, totalDstElements);

                        // upload full source frame into this worker's src buffer
                        var srcSpan = new ReadOnlySpan<ushort>(srcPtr, totalSrcElements);
                        worker.Accelerator.WriteBuffer(worker.SrcBuffer, srcSpan);

                        // call app kernel callback with this worker's buffers
                        gpuWorker(worker.Accelerator, worker.Kernel, worker.SrcBuffer, worker.DstBuffer, startY, endY);
                    } catch (Exception ex) {
                        Logger.Warning($"MultiDevice2DExecutor GPU worker {i} failed: {ex.Message}");
                        throw;
                    }
                }));
            }

            // 2) CPU worker: last chunk
            if (useCpu) {
                int rows = baseRows;
                if (remainder > 0) {
                    rows++;
                    remainder--;
                }

                if (rows > 0) {
                    int startY = currentY;
                    int endY = startY + rows;

                    tasks.Add(Task.Run(() => cpuWorker!(startY, endY)));
                }
            }

            Task.WaitAll(tasks);
        }

        private double[] BuildWorkerWeights(int gpuCount, bool useCpu) {
            int workerCount = gpuCount + (useCpu ? 1 : 0);
            var weights = new double[workerCount];

            lock (_perfLock) {
                for (int i = 0; i < gpuCount; i++) {
                    double rps = _gpuPerf[i].EstimatedRowsPerSecond;
                    // Default to 1.0 if we have no history yet.
                    weights[i] = rps > 0 ? rps : 1.0;
                }

                if (useCpu) {
                    double rps = _cpuPerf.EstimatedRowsPerSecond;
                    weights[gpuCount] = rps > 0 ? rps : 1.0;
                }
            }

            return weights;
        }

        /// <summary>
        /// Compute integer row allocation for each worker so that
        /// sum(rows[i]) == totalRows and rows[i] is roughly proportional to weights[i].
        /// </summary>
        private static int[] ComputeRowDistribution(int totalRows, double[] weights) {
            int n = weights.Length;
            var rows = new int[n];

            if (totalRows <= 0 || n == 0)
                return rows;

            double sumW = 0.0;
            for (int i = 0; i < n; i++) {
                if (weights[i] > 0)
                    sumW += weights[i];
            }

            // If all weights <= 0 for some reason, fall back to equal distribution.
            if (sumW <= 0.0) {
                int baseRows = totalRows / n;
                int remainder = totalRows % n;
                for (int i = 0; i < n; i++) {
                    rows[i] = baseRows + (i < remainder ? 1 : 0);
                }
                return rows;
            }

            var quotas = new double[n];
            int assigned = 0;

            for (int i = 0; i < n; i++) {
                double normalized = weights[i] > 0 ? (weights[i] / sumW) : 0.0;
                double q = normalized * totalRows;
                quotas[i] = q;
                int r = (int)Math.Floor(q);
                rows[i] = r;
                assigned += r;
            }

            int remaining = totalRows - assigned;
            if (remaining <= 0)
                return rows;

            // Largest remainder method: give extra rows to workers
            // with largest fractional parts first.
            var order = Enumerable.Range(0, n)
                .OrderByDescending(i => quotas[i] - Math.Floor(quotas[i]))
                .ToArray();

            int idx = 0;
            while (remaining > 0 && idx < order.Length) {
                rows[order[idx]]++;
                remaining--;
                idx++;
            }

            return rows;
        }

        private void UpdateGpuPerf(int index, int rows, double seconds) {
            if (index < 0)
                return;

            lock (_perfLock) {
                if (index >= _gpuPerf.Count)
                    return;

                _gpuPerf[index].AddSample(rows, seconds);
            }
        }

        private void UpdateCpuPerf(int rows, double seconds) {
            lock (_perfLock) {
                _cpuPerf.AddSample(rows, seconds);
            }
        }

        public void Dispose() {
            foreach (var w in _gpuWorkers) {
                w.Dispose();
            }
            _gpuWorkers.Clear();
            _gpuPerf.Clear();
        }
    }

}
