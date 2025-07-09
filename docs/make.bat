@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=.
set BUILDDIR=_build

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.https://sphinx-doc.org/
	exit /b 1
)

if "%1" == "" goto help
if "%1" == "livehtml" goto livehtml

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS% %O%
echo.
echo.Additional targets:
echo.  livehtml   Start live rebuild server using sphinx-autobuild
goto end

:livehtml
echo Starting live documentation server...
echo Open http://localhost:8000 in your browser
echo Press Ctrl+C to stop the server
sphinx-autobuild %SOURCEDIR% %BUILDDIR%\html %SPHINXOPTS% %O% --host 0.0.0.0 --port 8000
if errorlevel 1 (
	echo.
	echo.sphinx-autobuild not found. Install with: pip install sphinx-autobuild
	echo.Or use: build_docs.bat live
	exit /b 1
)
goto end

:end
popd
