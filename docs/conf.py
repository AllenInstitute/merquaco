import os
import sys

sys.path.insert(0, os.path.abspath('..'))

project = 'merquaco'
copyright = '2024, Allen Institue'
author = 'Naomi Martin, Paul Olsen'

version = '0.0'
release = '0.0.1'

extensions = ['sphinx.ext.autodoc']
source_suffix = '.rst'
master_doc = 'index'

html_theme = 'alabaster'
