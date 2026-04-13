# Documentation

This directory contains all documentation for the apps-benchmark project.

## Documentation Files

- **[LICENSE](LICENSE.md)** - Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0) license information
- **[CHANGELOG.md](CHANGELOG.md)** - Version history and release notes
- **[API_REFERENCE.md](API_REFERENCE.md)** - API documentation for developers
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Development workflows, tools, best practices, and code of conduct
- **[DIY_BENCHMARK.md](DIY_BENCHMARK.md)** - Guide for creating custom benchmark algorithms
- **[DIY_BACKEND.md](DIY_BACKEND.md)** - Guide for creating custom quantum backends
- **[diagrams/project_function_calls_and_dependencies.png](diagrams/project_function_calls_and_dependencies.png)** - Rendered architecture diagram for function calls and dependencies
- **[diagrams/project_function_calls_and_dependencies.dot](diagrams/project_function_calls_and_dependencies.dot)** - Editable Graphviz source for the architecture diagram, rendered locally

## Viewing Documentation

### Locally (Markdown)
All documentation is written in Markdown and can be viewed directly:
- In GitHub's web interface
- In any Markdown viewer
- In your IDE/editor with Markdown preview

### HTML Version
HTML documentation is automatically generated from Markdown files via GitHub Actions.

The workflow (`.github/workflows/docs-to-html.yml`) runs when:
- Changes are pushed to the `main` branch
- Pull requests modify files in `docs/`
- Manually triggered via GitHub Actions UI

Generated HTML is:
1. Uploaded as workflow artifacts (available for 90 days)
2. Deployed to GitHub Pages (if enabled)

### Accessing HTML Docs

**From GitHub Pages** (if enabled):
```
https://ionq.github.io/apps-benchmark/
```

**From Workflow Artifacts**:
1. Go to Actions tab in GitHub
2. Click on latest "Generate HTML Documentation" workflow run
3. Download "documentation-html" artifact

**Build Locally**:
```bash
# Install pandoc
# macOS:
brew install pandoc

# Ubuntu/Debian:
sudo apt-get install pandoc

# Then generate HTML
cd /path/to/apps-benchmark
mkdir -p docs_html
pandoc docs/CONTRIBUTING.md --from=gfm --to=html5 --standalone --css=style.css -o docs_html/CONTRIBUTING.html
```

## Documentation Workflow

### Adding New Documentation

1. Create or edit Markdown files in `docs/`
2. Use GitHub-flavored Markdown (GFM)
3. Commit and push changes
4. GitHub Actions automatically generates HTML
5. Update `docs_html/index.html` template in workflow if adding new files

### Updating Existing Documentation

Simply edit the Markdown files and commit. The HTML will regenerate automatically.

### Previewing Changes

Before committing, preview your Markdown:
- Use your IDE's Markdown preview
- Use a tool like `grip` for GitHub-flavored preview:
  ```bash
  pip install grip
  grip docs/CONTRIBUTING.md
  ```

## Documentation Standards

### Markdown Style
- Use GitHub-flavored Markdown
- Include table of contents for long documents
- Use code blocks with language tags: ` ```python `
- Include examples where appropriate
- Keep line length reasonable (80-100 chars when possible)

### Structure
- Start with h1 title (`#`)
- Use hierarchical headings (h2 `##`, h3 `###`, etc.)
- Include "See also" links to related docs
- Add a helpful description at the top

### Code Examples
- Test all code examples
- Include necessary imports
- Show expected output when helpful
- Use realistic examples

## GitHub Actions Workflow

The HTML generation workflow:

1. **Triggers on**:
   - Push to main (docs changes)
   - Pull requests (docs changes)
   - Manual dispatch

2. **Process**:
   - Checks out repository
   - Installs pandoc
   - Converts each `.md` file to `.html`
   - Applies CSS styling
   - Creates index page
   - Uploads artifacts
   - Deploys to GitHub Pages (main branch only)

3. **Output**:
   - Clean, styled HTML files
   - Responsive design
   - Consistent navigation

## Enabling GitHub Pages

To enable GitHub Pages for this repository:

1. Go to repository Settings
2. Navigate to "Pages" section
3. Source: Deploy from a branch
4. Branch: `gh-pages`, folder: `/ (root)`
5. Save

The workflow will automatically deploy to this branch when changes are pushed to main.

## Troubleshooting

### HTML not updating
- Check GitHub Actions workflow status
- Ensure changes were pushed to main branch
- Verify workflow has proper permissions

### Styling issues
- CSS is embedded in workflow file
- Edit `.github/workflows/docs-to-html.yml` to modify styles
- Re-run workflow after changes

### Broken links
- Use relative links for docs in same directory: `[text](FILE.md)`
- Use `../` for files in parent directory: `[text](../README.md)`
- External links: `[text](https://example.com)`

## Contributing to Docs

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

For documentation-specific contributions:
- Fix typos and improve clarity
- Add missing examples
- Update outdated information
- Improve organization and structure
- Add helpful diagrams or visuals

## Questions

For questions about documentation:
- Open an issue on GitHub
- Contact App Benchmark Support at apps-benchmark-support@ionq.co
