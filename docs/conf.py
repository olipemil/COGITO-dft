import os
import sys

# Make repo root importable so Sphinx can import COGITO_dft
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("_ext"))

project = "COGITO"
author = "Emily Oliphant"
copyright = "Emily Oliphant, 2026"

extensions = [
    "file_cards",
    "sphinx_inline_tabs",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx_design",
]

def linkcode_resolve(domain, info):
    if domain != "py" or not info["module"]:
        return None
    import importlib
    import inspect
    filename = info["module"].replace(".", "/")
    base_url = f"https://github.com/olipemil/COGITO-dft/blob/main/{filename}.py"
    try:
        mod = importlib.import_module(info["module"])
        obj = mod
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        lines, start = inspect.getsourcelines(obj)
        return f"{base_url}#L{start}"
    except Exception:
        return base_url

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"

myst_enable_extensions = ["colon_fence", "attrs_inline"]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "furo"
html_title = "COGITO Homepage"
html_subtitle = "Peer into the quantum blackbox"

html_static_path = ["_static"]
html_css_files = ["css/custom.css", "css/file_form.css"]
html_extra_path = ["./","Si/", "PbO/"]
