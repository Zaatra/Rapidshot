@echo off
setlocal
set "rapidshot_vc=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207"
set "rapidshot_sdk=%ProgramFiles(x86)%\Windows Kits\10\Lib\10.0.26100.0"
set "PATH=%rapidshot_vc%\bin\Hostx64\x64;%SystemRoot%\System32;%PATH%"
set "LIB=%rapidshot_vc%\lib\x64;%rapidshot_sdk%\um\x64;%rapidshot_sdk%\ucrt\x64"
"%USERPROFILE%\.cargo\bin\cargo.exe" build --offline --release --manifest-path "%~dp0Cargo.toml"
exit /b %errorlevel%
