@echo off

REM Build the documentation
REM Usage: build_docs.bat [live]
REM   build_docs.bat      - Build HTML documentation once
REM   build_docs.bat live - Start live rebuild server

if "%1"=="live" (
    echo Starting live documentation server...
    echo Open http://localhost:8000 in your browser
    echo Press Ctrl+C to stop the server
    sphinx-autobuild . _build/html --host localhost --port 8000
) else (
    echo Building HTML documentation...
    sphinx-build -b html . _build/html

    REM Check if build directory was created successfully
    if not exist "_build\html\index.html" (
        echo Build failed - no output generated
        exit /b 1
    )

    echo Build completed successfully
    echo Documentation is available at: _build\html\index.html

    REM Change to the build directory
    cd _build\html

    REM Serve the documentation locally
    echo Starting local server at http://localhost:8000
    echo Press Ctrl+C to stop the server
    python -m http.server 8000
)
