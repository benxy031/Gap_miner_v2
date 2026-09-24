@echo off
REM ===========================================================================
REM build_all.bat - full Windows build + validation of GapMiner V2.
REM
REM   1. bin\gapgpu.dll    (nvcc + MSVC)          -> windows\build_gpu_dll.bat
REM   2. bin\gapminer.exe  (MSYS2 MinGW-w64 GCC)  -> windows\build_host.sh
REM   3. CPU + GPU tests                          -> windows\logs\tests.log
REM
REM   build_all.bat          full build
REM   build_all.bat host     skip the (slow) DLL build if bin\gapgpu.dll exists
REM
REM Optional environment: CUDA_ARCH, GPU_NLIMBS, CUDA_HOME, MSYS2_ROOT (see
REM build_gpu_dll.bat) and TEST_GPU (run the GPU tests on one GPU only).
REM Logs: windows\logs\build_gpu_dll.log, build_host.log, tests.log, build_all.log
REM ===========================================================================
setlocal EnableExtensions
cd /d "%~dp0.."
if not exist windows\logs mkdir windows\logs
set "ALOG=%CD%\windows\logs\build_all.log"
echo [%date% %time%] build_all start %* > "%ALOG%"
if "%MSYS2_ROOT%"=="" set "MSYS2_ROOT=C:\msys64"
if "%GPU_NLIMBS%"=="" set "GPU_NLIMBS=32"

if /i "%~1"=="host" if exist bin\gapgpu.dll goto host

call windows\build_gpu_dll.bat
if errorlevel 1 (
    echo [%date% %time%] STEP 1 FAILED - gapgpu.dll >> "%ALOG%"
    echo.
    echo *** STEP 1 FAILED: gapgpu.dll - see windows\logs\build_gpu_dll.log
    goto end
)
echo [%date% %time%] STEP 1 OK - gapgpu.dll >> "%ALOG%"

:host
if not exist "%MSYS2_ROOT%\usr\bin\bash.exe" (
    echo *** MSYS2 not found in %MSYS2_ROOT% - install it or set MSYS2_ROOT
    goto end
)
set MSYSTEM=MINGW64
set CHERE_INVOKING=1
"%MSYS2_ROOT%\usr\bin\bash.exe" -lc "./windows/build_host.sh"
if errorlevel 1 (
    echo [%date% %time%] STEP 2/3 FAILED - host build or tests >> "%ALOG%"
    echo.
    echo *** STEP 2/3 FAILED - see windows\logs\build_host.log and tests.log
    goto end
)
echo [%date% %time%] STEP 2/3 OK - gapminer.exe built, all tests passed >> "%ALOG%"
echo.
echo *** BUILD + TESTS OK
:end
echo [%date% %time%] build_all end >> "%ALOG%"
echo.
pause
