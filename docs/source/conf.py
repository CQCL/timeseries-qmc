# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'TimeseriesQMC'
copyright = '2023, Quantinuum GmbH'
author = 'Khaldoon Ghanem'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.graphviz',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.intersphinx', 
    'sphinxcontrib.bibtex',
    'sphinx_copybutton',
    # 'nbsphinx'
    'jupyter_sphinx'
]

graphviz_output_format = 'svg'
inheritance_graph_attrs = dict(rankdir="TB", size='""')

napoleon_google_docstring = False

templates_path = ['_templates']
exclude_patterns = []

# Looks for objects in external projects
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/1.22', None),
    'pytket': ('https://cqcl.github.io/tket/pytket/api/', None),
    'tenpy': ('https://tenpy.readthedocs.io/en/latest/', None),
    'quspin': ('https://quspin.github.io/QuSpin/', None),    
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

bibtex_bibfiles = ['refs.bib']
# bibtex_default_style = 'unsrt'

# html_sidebars = {
#     '**': [
#         'globaltoc.html',
#     ]
# }