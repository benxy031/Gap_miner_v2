@echo off
REM ===========================================================================
REM build_gpu_dll.bat - builds bin\gapgpu.dll (the CUDA code) with nvcc + MSVC.
REM
REM nvcc on Windows only accepts MSVC as host compiler, so the two .cu files
REM are built into a DLL exposing their plain-C API (gpu_fermat.h /
REM gpu_sieve.h, exports listed in windows\gapgpu.def).  The rest of the miner
REM is built with MinGW-w64 (Makefile.win) and links directly against it.
REM
REM Environment (all optional):
REM   CUDA_ARCH   target GPU arch                      (default sm_61 = Pascal)
REM               CUDA 12.x still compiles sm_61; CUDA 13 dropped Pascal, so
REM               with CUDA 13 use sm_75 or newer (e.g. sm_86 for RTX 30xx).
REM   GPU_NLIMBS  candidate width in 64-bit limbs      (default 32 = 2048 bits,
REM               same as GPU_BITS=2048 in the Linux Makefile; must match the
REM               value used by Makefile.win / build_host.sh)
REM   CUDA_HOME   CUDA toolkit directory  (default %CUDA_PATH_V12_9%, else %CUDA_PATH%)
REM   MSYS2_ROOT  MSYS2 install directory (default C:\msys64), for gmp.h
REM
REM Output : bin\gapgpu.dll, bin\gapgpu.lib, bin\cudart64_*.dll
REM Log    : windows\logs\build_gpu_dll.log
REM Objects are cached per arch/width in build-win\gpu-<arch>-nl<N>: each .cu
REM is recompiled only when its source changed (gpu_fermat.cu takes ~10 min).
REM ===========================================================================
setlocal EnableExtensions
cd /d "%~dp0.."
if not exist windows\logs mkdir windows\logs
if not exist bin mkdir bin
set "LOG=%CD%\windows\logs\build_gpu_dll.log"
REM If a previous run left the log locked, write to a fresh file instead.
(echo [%date% %time%] build_gpu_dll start> "%LOG%") 2>nul || set "LOG=%CD%\windows\logs\build_gpu_dll_%RANDOM%.log"
echo [%date% %time%] build_gpu_dll start >> "%LOG%"
echo Log file: %LOG%

if "%CUDA_ARCH%"=="" set "CUDA_ARCH=sm_61"
if "%GPU_NLIMBS%"=="" set "GPU_NLIMBS=32"
if "%MSYS2_ROOT%"=="" set "MSYS2_ROOT=C:\msys64"
echo CUDA_ARCH=%CUDA_ARCH% GPU_NLIMBS=%GPU_NLIMBS% MSYS2_ROOT=%MSYS2_ROOT% >> "%LOG%"

REM --- CUDA toolkit ------------------------------------------------------------
set "CUDADIR=%CUDA_HOME%"
if "%CUDADIR%"=="" set "CUDADIR=%CUDA_PATH_V12_9%"
if "%CUDADIR%"=="" set "CUDADIR=%CUDA_PATH%"
if not exist "%CUDADIR%\bin\nvcc.exe" (
    echo ERROR: nvcc.exe not found in "%CUDADIR%\bin" >> "%LOG%"
    echo ERROR: nvcc.exe not found. Install the CUDA Toolkit or set CUDA_HOME.
    exit /b 1
)
echo CUDADIR=%CUDADIR% >> "%LOG%"

REM --- MSVC environment (VS 2022) ----------------------------------------------
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
echo VSINSTALL=%VSINSTALL% >> "%LOG%"
REM vcvars64 may start a background telemetry process that inherits (and keeps
REM open) whatever file its output is redirected to: disable the telemetry and
REM never redirect vcvars into the build log, or the log gets locked.
set VSCMD_SKIP_SENDTELEMETRY=1
call "%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
where cl >nul 2>&1
if errorlevel 1 (
    echo ERROR: vcvars64.bat did not put cl.exe on PATH >> "%LOG%"
    echo ERROR: MSVC environment setup failed.
    exit /b 1
)
set "PATH=%CUDADIR%\bin;%PATH%"

echo. >> "%LOG%"
where cl >> "%LOG%" 2>&1
nvcc --version >> "%LOG%" 2>&1

REM --- gmp.h for CGBN's host-side routing (header only, nothing is linked) -----
REM Only gmp.h is copied: adding the MinGW include directory to the MSVC
REM include path would shadow the MSVC C runtime headers.
if not exist windows\include_gmp mkdir windows\include_gmp
copy /y "%MSYS2_ROOT%\mingw64\include\gmp.h" windows\include_gmp\gmp.h >> "%LOG%" 2>&1
if errorlevel 1 (
    echo ERROR: could not copy %MSYS2_ROOT%\mingw64\include\gmp.h >> "%LOG%"
    echo ERROR: gmp.h not found - install mingw-w64-x86_64-gmp in MSYS2 or set MSYS2_ROOT.
    exit /b 1
)

REM --- Build --------------------------------------------------------------------
echo. >> "%LOG%"
echo [%date% %time%] nvcc start (%CUDA_ARCH%, %GPU_NLIMBS% limbs) >> "%LOG%"
echo Compiling gapgpu.dll for %CUDA_ARCH% - this takes a while, see the log above
set "NVFLAGS=-O3 -arch=%CUDA_ARCH% -Wno-deprecated-gpu-targets -allow-unsupported-compiler -std=c++17 -DGPU_NLIMBS=%GPU_NLIMBS% -DWITH_CGBN_FERMAT -Itools\cgbn\include -Iwindows\include_gmp -Xcompiler "/MD /O2 /EHsc /W3" -cudart shared"
set "OBJDIR=build-win\gpu-%CUDA_ARCH%-nl%GPU_NLIMBS%"
if not exist "%OBJDIR%" mkdir "%OBJDIR%"

call :compile gpu_fermat
if errorlevel 1 exit /b 1
call :compile gpu_sieve
if errorlevel 1 exit /b 1

echo [%date% %time%] link gapgpu.dll >> "%LOG%"
REM Link against the DLL C runtime (/MD), like the objects were compiled.
nvcc -arch=%CUDA_ARCH% -Wno-deprecated-gpu-targets -cudart shared --shared -Xcompiler "/MD" ^
     -Xlinker /NODEFAULTLIB:LIBCMT ^
     -o bin\gapgpu.dll "%OBJDIR%\gpu_fermat.obj" "%OBJDIR%\gpu_sieve.obj" ^
     -Xlinker /DEF:windows\gapgpu.def >> "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
echo [%date% %time%] nvcc link exit code %RC% >> "%LOG%"
if not "%RC%"=="0" (
    echo ERROR: link failed with code %RC%, see windows\logs\build_gpu_dll.log
    exit /b %RC%
)
REM Do not trust the exit code alone: a failed redirection can leave it at 0.
if not exist bin\gapgpu.dll (
    echo ERROR: bin\gapgpu.dll was not produced >> "%LOG%"
    echo ERROR: bin\gapgpu.dll was not produced, see windows\logs\build_gpu_dll.log
    exit /b 1
)

REM --- CUDA runtime DLL next to the miner --------------------------------------
copy /y "%CUDADIR%\bin\cudart64_*.dll" bin\ >> "%LOG%" 2>&1
dumpbin /exports bin\gapgpu.dll >> "%LOG%" 2>&1
echo [%date% %time%] build_gpu_dll OK >> "%LOG%"
echo gapgpu.dll OK
exit /b 0

REM -------------------------------------------------------------------------------
REM :compile <name>  -> %OBJDIR%\<name>.obj (skipped if the source is unchanged)
:compile
set "SRC=new_src\gpu\%~1.cu"
set "OBJ=%OBJDIR%\%~1.obj"
set "STAMP=%OBJDIR%\%~1.cu.stamp"
if exist "%OBJ%" if exist "%STAMP%" (
    fc /b "%SRC%" "%STAMP%" >nul 2>&1
    if not errorlevel 1 (
        echo [%date% %time%] %~1.cu unchanged - cached object reused >> "%LOG%"
        exit /b 0
    )
)
echo [%date% %time%] compiling %~1.cu >> "%LOG%"
echo   compiling %~1.cu ...
if exist "%OBJ%" del /q "%OBJ%"
nvcc %NVFLAGS% -c "%SRC%" -o "%OBJ%" >> "%LOG%" 2>&1
if errorlevel 1 (
    echo [%date% %time%] %~1.cu FAILED >> "%LOG%"
    echo ERROR: %~1.cu failed to compile, see windows\logs\build_gpu_dll.log
    exit /b 1
)
if not exist "%OBJ%" (
    echo ERROR: %OBJ% was not produced >> "%LOG%"
    exit /b 1
)
copy /y "%SRC%" "%STAMP%" >nul
echo [%date% %time%] %~1.cu OK >> "%LOG%"
exit /b 0
