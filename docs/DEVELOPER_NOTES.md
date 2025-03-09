# Documentation Build Process Overview

This document provides an overview of how the documentation build process works, which files are responsible for generating the final HTML output, and the dependencies required to build the documentation.

## Build Process Overview

1. **Makefile in `docs/`**  
   - Orchestrates the entire build process by running commands sequentially.
   - Creates `LICENSE.md` from the repository’s root `LICENSE.txt`.
   - Uses **Quarto** to render the Quarto Markdown source (`docs/source/_static/README.qmd`) into a Markdown file.
   - Uses **Pandoc** to process the rendered Markdown files.
   - Performs text replacements (using `sed`) so that links (like the license reference) are updated (e.g., converting references from `LICENSE.txt` to `LICENSE.html`).

2. **`docs/source/_static/README.qmd`**  
   - This Quarto file is the source for the README content.
   - When built, Quarto renders this file into `README.md`.
   - Changes here affect the content that appears in the README sections of the documentation.

3. **`docs/source/README.md`**  
   - The generated `README.md` is used as the basis for documentation.
   - The Makefile adjusts `README.md` (applying a `sed` replacement so that the license link points to `LICENSE.html` instead of `LICENSE.txt`).
   - This file is later included in the Sphinx documentation.

4. **`docs/source/index.rst`**  
   - The Sphinx master index file.
   - Uses reStructuredText directives (e.g., `.. include::`) to pull in the contents of `README.md`.
   - Defines the table of contents (toctree) for the documentation sections (e.g., examples, documentation, license).

5. **`docs/source/conf.py`**  
   - The Sphinx configuration file that sets up themes, extensions, and other settings.
   - Influences the look and behavior of the final HTML output.

6. **Other `.rst` Files (e.g., `base.rst`, `panel.rst`, `training.rst`, etc.)**  
   - These files are individual sections of the documentation.
   - They are referenced by the toctree in `index.rst` and become separate pages in the final HTML output.

## Documentation Build Dependencies

The following tools and packages are required to build the documentation. Note that these are **development/documentation-only dependencies** and are not needed for the runtime of your package:

- **Python Packages:**  
  - `Sphinx` – Core documentation generator.  
  - `sphinx_rtd_theme` – The theme used for the HTML output.  
  - `myst-parser` – Parses Markdown files for Sphinx.  
  - `sphinx-autodoc-typehints` – Automatically documents type hints (if used).  
  - `PyYAML` – Needed for YAML parsing in some build steps.

  These can be installed via pip:
  ```bash
  pip install sphinx sphinx_rtd_theme myst-parser sphinx-autodoc-typehints PyYAML
  ```

### External Tools

- **Pandoc:** Converts Markdown files (install from [Pandoc's website](https://pandoc.org/installing.html) or via a package manager).
- **Quarto:** Renders the Quarto Markdown file (install from [Quarto's website](https://quarto.org/docs/get-started/)).
- **Jupyter:** Required if Quarto executes notebooks (install via pip: `pip install jupyter`).

> It's a good practice to list these dependencies in a separate file (e.g., `docs/requirements.txt` or `requirements-dev.txt`) and include the installation instructions in this document.

## Which Files Change the Output?

### Content Changes

- **`docs/source/_static/README.qmd`**  
  *Change this file to update the main README content.*

- **Other `.rst` files (e.g., `base.rst`, `panel.rst`, etc.):**  
  *Edit these to update the corresponding documentation sections.*

### Build and Link Adjustments

- **`docs/Makefile`**  
  *Controls how files are processed and how links (like the license reference) are modified.*

- **`sed` Commands in the Makefile:**  
  *Replace instances of license references in the generated Markdown files (e.g., converting `[MIT license](LICENSE.txt)` to `[MIT license](LICENSE.html)` or an HTML anchor tag).*

### Configuration and Appearance

- **`docs/source/conf.py`**  
  *Adjust this file to change Sphinx settings or themes, which affects the final output’s appearance.*

## Summary

- **Makefile:**  
  Orchestrates the build process (renders Quarto, runs Pandoc, performs `sed` replacements, and builds with Sphinx).

- **README Files Sequence:**  
  `README.qmd` → `README.md`
  These files are processed in sequence. The Quarto source produces the Markdown that ends up being included in your final documentation.

- **Index and RST Files:**  
  `index.rst` and other `.rst` files define the structure and content of the final HTML pages.

- **Sphinx Configuration:**  
  `conf.py` controls the Sphinx settings.

## Building the Documentation

When any of these source files or configurations are modified, the changes will be reflected in the final HTML output (located in `docs/build/html`) once you run:

```bash
make clean html
```


