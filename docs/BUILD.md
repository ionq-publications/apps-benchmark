# Documentation Build Guide

This directory contains the documentation source files in Markdown format.

## Files

- **README.md** - Main documentation and quick start guide
- **LICENSE.md** - Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0) license information
- **BUILD.md** - This file (build instructions)

## Building Documentation

### Local Build

To build the documentation locally, you need `pandoc` installed:

```bash
# macOS
brew install pandoc

# Ubuntu/Debian
sudo apt-get install pandoc

# From the repository root, build the HTML docs
./build_docs.sh
```

This will create HTML files in the `docs_html/` directory.

To view the documentation locally:

```bash
cd docs_html
python -m http.server 8000
# Visit http://localhost:8000 in your browser
```

### Automated Build (GitHub Actions)

The documentation is automatically built and deployed when:

1. Changes are pushed to `main` branch in the `docs/` directory
2. A pull request modifies files in `docs/`
3. Manually triggered via the Actions tab

The workflow (`.github/workflows/docs-to-html.yml`) will:

- Convert all `.md` files in `docs/` to HTML
- Apply consistent styling with CSS
- Deploy to GitHub Pages (on main branch pushes)
- Upload artifacts for PR previews

### GitHub Pages Deployment

Once enabled in repository settings:

1. Go to repository Settings > Pages
2. Set Source to "GitHub Actions"
3. Push changes to `docs/` on main branch
4. Documentation will be available at: `https://<username>.github.io/<repo>/`

## Adding New Documentation

1. Create a new `.md` file in the `docs/` directory
2. Write content using standard Markdown
3. Run `./build_docs.sh` to test locally
4. Commit and push - automatic build will handle the rest

## Markdown Features Supported

The pandoc converter supports:

- Headers (H1-H6)
- Code blocks with syntax highlighting
- Inline code
- Lists (ordered and unordered)
- Tables
- Links
- Blockquotes
- Bold and italic text

## Styling

The generated HTML uses a GitHub-inspired CSS theme with:

- Responsive layout (max-width: 900px)
- System font stack for readability
- Syntax-highlighted code blocks
- Clean table styling
- Proper heading hierarchy

## Troubleshooting

**Problem**: `pandoc: command not found`
- **Solution**: Install pandoc as shown above

**Problem**: GitHub Pages not deploying
- **Solution**: Check repository Settings > Pages is configured for "GitHub Actions"

**Problem**: Styles not applied
- **Solution**: Ensure `style.css` is in the same directory as HTML files
