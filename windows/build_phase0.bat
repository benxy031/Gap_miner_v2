@echo off
REM ===========================================================================
REM build_phase0.bat - builds bin\phase0gpu.dll (the Phase-0 kernels) with
REM nvcc + MSVC.
REM
REM Why a DLL: the Phase-0 tools (phase0_scan_gpu / mr68_gpu / bench_p0sieve)
REM need POSIX threads and GMP on the host side (MSYS2/MinGW), while nvcc on
REM Windows only accepts MSVC as host compiler.  The kernels therefore live in
REM this DLL behind the plain-C API of tools\phase0gpu_api.h (exports via
REM __declspec(dllexport) in tools\phase0gpu_dll.cu); the tools are built by
REM Makefile.win (MinGW) with -DPHASE0_KERNEL_DLL and link against it.
REM
REM Unlike gapgpu.dll this DLL needs NO gmp.h (the phase0 kernels do not use
REM GMP; GMP stays on the MinGW host side only).
REM
REM Environment (all optional):
REM   CUDA_ARCH   target GPU arch                      (default sm_61 = Pascal)
REM               CUDA 12.x still compiles sm_61; with CUDA 13 use sm_75+.
REM   CUDA_HOME   CUDA toolkit directory  (default %CUDA_PATH_V12_9%, else %CUDA_PATH%)
REM
REM Output : bin\phase0gpu.dll (+ cudart64_*.dll copied next to it if missing)
REM Log    : windows\logs\build_phase0.log
REM Objects are cached per arch in build-win\gpu-phase0-<arch>: the single .cu
REM is recompiled only when its source or a shared header changed.
REM ===========================================================================
setlocal EnableExtensions
cd /d "%~dp0.."
if not exist windows\logs mkdir windows\logs
if not exist bin mkdir bin
set "LOG=%CD%\windows\logs\build_phase0.log"
(echo [%date% %time%] build_phase0 start> "%LOG%") 2>nul || set "LOG=%CD%\windows\logs\build_phase0_%RANDOM%.log"
echo [%date% %time%] build_phase0 start >> "%LOG%"
echo Log file: %LOG%

if "%CUDA_ARCH%"=="" set "CUDA_ARCH=sm_61"
echo CUDA_ARCH=%CUDA_ARCH% >> "%LOG%"

REM --- CUDA toolkit -----------------------------------------------------------
set "CUDADIR=%CUDA_HOME%"
if "%CUDADIR%"=="" set "CUDADIR=%CUDA_PATH_V12_9%"
if "%CUDADIR%"=="" set "CUDADIR=%CUDA_PATH%"
if not exist "%CUDADIR%\bin\nvcc.exe" (
    echo ERROR: nvcc.exe not found in "%CUDADIR%\bin" >> "%LOG%"
    echo ERROR: nvcc.exe not found. Install the CUDA Toolkit or set CUDA_HOME.
    exit /b 1
)

REM --- MSVC environment (VS 2022) ---------------------------------------------
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "VSINSTALL="
if exist "%VSWHERE%" (
    for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINSTALL=%%i"
)
if "%VSINSTALL%"=="" set "VSINSTALL=C:\Program Files\Microsoft Visual Studio\2022\Community"
if not exist "%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat" (
    echo ERROR: vcvars64.bat not found under "%VSINSTALL%" >> "%LOG%"
    echo ERROR: MSVC x64 tools not found. Install the "Desktop development with C++" workload.
    exit /b 1
)
REM vcvars64 may spawn background telemetry that keeps the log file open: never
REM redirect it into the build log (same lesson as build_gpu_dll.bat).
set VSCMD_SKIP_SENDTELEMETRY=1
call "%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
where cl >nul 2>&1
if errorlevel 1 (
    echo ERROR: vcvars64.bat did not put cl.exe on PATH >> "%LOG%"
    exit /b 1
)
set "PATH=%CUDADIR%\bin;%PATH%"
echo. >> "%LOG%"
nvcc --version >> "%LOG%" 2>&1

REM --- Build (single TU; header set = shared kernel headers) ------------------
REM -allow-unsupported-compiler: newer MSVC versions than the toolkit knows.
set "NVFLAGS=-O3 -arch=%CUDA_ARCH% -Wno-deprecated-gpu-targets -allow-unsupported-compiler -std=c++17 -Itools -Xcompiler "/MD /O2 /EHsc /W3" -cudart shared"
set "OBJDIR=build-win\gpu-phase0-%CUDA_ARCH%"
if not exist "%OBJDIR%" mkdir "%OBJDIR%"

echo [%date% %time%] nvcc phase0gpu_dll.cu (%CUDA_ARCH%) >> "%LOG%"
echo Compiling phase0gpu.dll for %CUDA_ARCH% - see the log above
nvcc %NVFLAGS% -c tools\phase0gpu_dll.cu -o "%OBJDIR%\phase0gpu_dll.obj" >> "%LOG%" 2>&1
if errorlevel 1 (
    echo ERROR: phase0gpu_dll.cu failed to compile, see windows\logs\build_phase0.log
    exit /b 1
)

echo [%date% %time%] link phase0gpu.dll >> "%LOG%"
nvcc -arch=%CUDA_ARCH% -Wno-deprecated-gpu-targets -cudart shared --shared -Xcompiler "/MD" -Xlinker /NODEFAULTLIB:LIBCMT -o bin\phase0gpu.dll "%OBJDIR%\phase0gpu_dll.obj" >> "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo ERROR: link failed with code %RC%, see windows\logs\build_phase0.log
    exit /b %RC%
)
if not exist bin\phase0gpu.dll (
    echo ERROR: bin\phase0gpu.dll was not produced, see windows\logs\build_phase0.log
    exit /b 1
)

REM --- CUDA runtime DLL next to the tools (they link it directly, like the
REM     GPU tests link the same cudart gapgpu.dll uses) ----------------------
if not exist bin\cudart64_*.dll copy /y "%CUDADIR%\bin\cudart64_*.dll" bin\ >> "%LOG%" 2>&1
dumpbin /exports bin\phase0gpu.dll >> "%LOG%" 2>&1
echo [%date% %time%] build_phase0 OK >> "%LOG%"
echo phase0gpu.dll OK
exit /b 0
