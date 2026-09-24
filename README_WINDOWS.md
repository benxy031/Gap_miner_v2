# GapMiner V2 on Windows

Native Windows build of GapMiner V2 (no WSL), tested in production on
**5x NVIDIA P104-100 (Pascal, sm_61, 8 GB)** with an i5 quad-core.

## How it is built

`nvcc` on Windows only accepts MSVC as host compiler, while the host code uses
POSIX threads, GMP, libcurl and jansson, which are easiest to get from MSYS2.
The build is therefore split along the existing plain-C GPU API
(`gpu_fermat.h` / `gpu_sieve.h`, both `extern "C"`):

| Part | Sources | Toolchain | Output |
|---|---|---|---|
| GPU | `new_src/gpu/gpu_fermat.cu`, `gpu_sieve.cu` | CUDA + MSVC (VS 2022) | `bin/gapgpu.dll` (exports in `windows/gapgpu.def`) |
| Host | everything else | MSYS2 MINGW64 GCC (`Makefile.win`) | `bin/gapminer.exe` + tests, linked directly against `gapgpu.dll` |

The Linux build (`Makefile`) is unchanged.

## Requirements

- NVIDIA driver supporting your CUDA toolkit.
- **CUDA Toolkit 12.x** for Pascal (12.9 is the last release that compiles
  `sm_61`; CUDA 13 removed Maxwell/Pascal/Volta). CUDA 13 works for `sm_75`+.
- **Visual Studio 2022** (Community is fine) with the *Desktop development with C++* workload.
- **MSYS2** with the MINGW64 packages:
  ```
  pacman -S --needed mingw-w64-x86_64-gcc make mingw-w64-x86_64-pkgconf \
      mingw-w64-x86_64-gmp mingw-w64-x86_64-curl mingw-w64-x86_64-jansson \
      mingw-w64-x86_64-openssl
  ```

## Build

Double-click `windows\build_all.bat` (or run it from `cmd`). It:

1. builds `bin\gapgpu.dll` (`windows\build_gpu_dll.bat`, ~10-15 min the first
   time; objects are cached per arch/width in `build-win\`),
2. builds `bin\gapminer.exe` and the tests with MinGW (`windows\build_host.sh`),
3. runs the CPU tests and `test_gpu_fermat` / `test_gpu_sieve` / `test_gpu_resolve`.

`build_all.bat host` rebuilds only the host side when `gapgpu.dll` exists.
Logs are in `windows\logs\`.

Optional environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `CUDA_ARCH` | `sm_61` | GPU architecture (`sm_75`, `sm_86`, ...) |
| `GPU_NLIMBS` | `32` | candidate width in 64-bit limbs (2048 bits, same as `GPU_BITS=2048` on Linux); must be the same for the DLL and the host |
| `CUDA_HOME` | `%CUDA_PATH_V12_9%`, else `%CUDA_PATH%` | CUDA toolkit directory |
| `MSYS2_ROOT` | `C:\msys64` | MSYS2 installation |
| `TEST_GPU` | all GPUs | run the GPU tests on one GPU only (PCI bus order) |

## Run

Same options as on Linux. From `cmd`, in the repository root:

```bat
set PATH=%CD%\bin;C:\msys64\mingw64\bin;%PATH%
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set FUSED_GPU=1
bin\gapminer.exe --host 127.0.0.1 --port 31397 --user USER --pass PASS ^
    --crt-file data\crt\m23\shift509_p74_covermax_m38.txt --threads 5 ^
    --enable-gpu-fermat --enable-submission --coinbase-script-hex 76a914<hash160>88ac
```

`--threads` = number of GPUs (one worker per card). On Windows stdout is
unbuffered, so redirecting it to a log file (`>> miner.log 2>&1`) keeps the log live.

## Windows-specific changes

- **LLP64:** on Windows `unsigned long` is 32 bits, so GMP's `mpz_*_ui()`
  functions silently truncate `uint64_t` arguments >= 2^32 (e.g. a non-CRT
  nAdd at high shift, or large offsets), which would test and submit the wrong
  number. `new_src/win_compat.h` (force-included by `Makefile.win` only)
  wraps `mpz_{set,init_set,add,sub,mul,addmul}_ui` with 64-bit-safe versions
  that keep the GMP fast path when the value fits; the call sites now cast to
  `uint64_t` instead of `unsigned long` (identical on LP64 Linux). It also
  provides `setenv`/`unsetenv`.
- `printf("%lu")` on `uint64_t` counters replaced by `PRIu64`.
- `new_src/gpu/compat_win32.h` (MSVC only): `pthread` mutex/condvar on SRW
  locks, `clock_gettime(CLOCK_MONOTONIC)` on QueryPerformanceCounter, and the
  GCC `__atomic_*` builtins on Interlocked intrinsics, for the host side of
  the `.cu` files.
- `gpu_sieve.cu`: `(unsigned __int128)v * inv >> 64` replaced by the CUDA
  intrinsic `__umul64hi(v, inv)` (same value, compiles under MSVC; the device
  code is otherwise unchanged).
- `stratum.c`: Winsock port (WSAStartup, `ioctlsocket` non-blocking connect,
  `closesocket`, errno mapping). Windows uses a `select()` idle tick instead
  of `SO_RCVTIMEO`, because a timed-out blocking `recv` leaves a Winsock
  connection in an undefined state.
- `gap_hunt.c`: `signal()` instead of `sigaction()`; `block_assembly.c`: no
  `arpa/inet.h`.

## Validation (5x P104-100, CUDA 12.9, MSVC 14.44, MSYS2 GCC)

- `test_gpu_fermat`: 0 mismatches vs GMP, device path == H2D path at every width tested;
  `test_gpu_sieve`, `test_gpu_resolve` and the CPU tests pass.
- Production, `FUSED_GPU=1` + MINING_JUMP2, `shift509_p74_covermax_m38`, `--threads 5`:
  ~11,000 windows/s (~2,200 per card), about 0.105 winning windows per million at difficulty ~24.3,
  75+ blocks in the first 18 h with 0 rejected / 0 stale and no crash,
  and 4 prime gap records found on the way (verified and accepted on primegaps.cloudygo.com).
- With 5 workers on a 4-core CPU the host side starts to limit a little
  (~2,600 windows/s per card with 2 workers vs ~2,200 with 5).
