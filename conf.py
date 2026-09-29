"""Build the book without importing TVM, CUDA, or the kernel toolchain."""

project = "Agentic GPU Programming for MLSys"
author = "TIRx Contributors"
copyright = "2026, TIRx Contributors"
language = "en"
root_doc = "index"
extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx.ext.extlinks",
    "sphinx.ext.mathjax",
]
source_suffix = {".md": "markdown"}
exclude_patterns = [
    "_build", ".venv", "README.md", ".DS_Store",
]
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath"]
myst_heading_anchors = 3

# Use bundled SVG rendering so equations work without a CDN or web fonts.
# textmacros handles escaped underscores in text such as \texttt{A\_log}.
mathjax_path = "mathjax/tex-svg.js"
mathjax3_config = {
    "loader": {"load": ["[tex]/textmacros"]},
    "tex": {"packages": {"[+]": ["textmacros"]}},
    "options": {"enableMenu": False},
    "svg": {"fontCache": "local"},
}

# Follow each repository's default branch for source and documentation links.
extlinks = {
    "harness": ("https://github.com/mlc-ai/TIRx-harness/blob/main/%s", "%s"),
    "kernels": ("https://github.com/mlc-ai/TIRx-kernels/blob/main/tirx_kernels/%s", "%s"),
}

html_theme = "sphinx_book_theme"
templates_path = ["_templates"]
html_sidebars = {
    "**": [
        "navbar-logo.html",
        "book-title.html",
        "icon-links.html",
        "search-button-field.html",
        "sbt-sidebar-nav.html",
    ],
}
html_context = {"default_mode": "auto"}
html_title = project
html_baseurl = "https://mlc.ai/agentic-gpu-programming-for-mlsys/"
html_logo = "_static/mlc-logo-with-text-landscape.svg"
html_static_path = ["_static"]
html_css_files = ["book.css"]
html_js_files = ["book-tabs.js", "fold-code.js"]
html_theme_options = {
    "logo": {"link": "https://mlc.ai/", "alt_text": "Machine Learning Compilation home"},
    "repository_url": "https://github.com/mlc-ai/agentic-gpu-programming-for-mlsys",
    "use_repository_button": True,
    "use_download_button": False,
    "use_fullscreen_button": False,
    "show_navbar_depth": 1,
    "show_toc_level": 2,
    "home_page_in_toc": False,
    "navbar_persistent": [],
}
html_show_sourcelink = True
html_show_sphinx = False
html_use_index = False
html_last_updated_fmt = None
copybutton_prompt_text = r"\$ "
copybutton_prompt_is_regexp = True
