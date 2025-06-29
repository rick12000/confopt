# Documentation

This directory contains the documentation for the ConfOpt project, built using [Sphinx](https://www.sphinx-doc.org/).

## Building Documentation

### Prerequisites

Ensure you have Python 3.8+ installed, then install the documentation dependencies:

```bash
# Install documentation dependencies
pip install -r requirements.txt

# Or install the project with documentation extras
pip install -e ".[docs]"
```

### Building HTML Documentation

To build the documentation:

```bash
# Using make (recommended)
make html

# Or using sphinx-build directly
sphinx-build -b html . _build/html
```

The built documentation will be available in `_build/html/index.html`.

### Live Development Server

For active development with live reload:

```bash
make livehtml
```

This will start a development server at `http://localhost:8000` that automatically rebuilds and refreshes when you make changes.

### Other Build Targets

```bash
# Check for broken links
make linkcheck

# Clean build directory
make clean

# Build PDF (requires LaTeX)
make latexpdf

# Build EPUB
make epub
```

## Documentation Structure

- `conf.py` - Sphinx configuration file
- `index.rst` - Main documentation index
- `_static/` - Static files (CSS, images, etc.)
- `_templates/` - Custom HTML templates
- `_build/` - Generated documentation (ignored by git)

## Writing Documentation

### reStructuredText

The documentation is primarily written in reStructuredText (`.rst` files). See the [reStructuredText primer](https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html) for syntax help.

### Markdown Support

Markdown files (`.md`) are also supported via MyST parser. See [MyST documentation](https://myst-parser.readthedocs.io/) for advanced features.

### API Documentation

API documentation is automatically generated from docstrings using the `autodoc` extension. Ensure your code has proper docstrings following Google or NumPy style.

## Configuration

Key configuration options in `conf.py`:

- **Extensions**: Enabled Sphinx extensions
- **Theme**: Currently using `sphinx_rtd_theme`
- **Intersphinx**: Links to external documentation
- **Autodoc**: Automatic API documentation settings

## Deployment

Documentation is automatically built and deployed via GitHub Actions:

- **On Pull Requests**: Validates documentation builds correctly
- **On Main Branch**: Deploys to GitHub Pages

## Troubleshooting

### Common Issues

1. **Build Errors**: Check that all dependencies are installed and up to date
2. **Import Errors**: Ensure the source code is importable (check `sys.path` in `conf.py`)
3. **Link Check Failures**: Some external links may be temporarily unavailable

### Getting Help

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [reStructuredText Guide](https://docutils.sourceforge.io/rst.html)
- [MyST Parser](https://myst-parser.readthedocs.io/)

## Contributing

When contributing to documentation:

1. Test your changes locally using `make html` or `make livehtml`
2. Run link checks with `make linkcheck`
3. Follow the existing style and structure
4. Update this README if you add new build targets or change the workflow
