# -*- coding: utf-8 -*-

from distutils.core import setup
from setuptools import find_packages
import numpy
from Cython.Build import cythonize
from Cython.Distutils import build_ext


ext_modules = cythonize("cell_localization/flow/*.pyx")

def main():
    setup(
        name             = 'cell_localization',
        version          = '0.0.1',
        packages         = find_packages(),
        cmdclass = {'build_ext': build_ext},
        ext_modules = ext_modules,
        include_dirs = [numpy.get_include()],
        author           = 'Avelino Javer',
        author_email     = 'avelino.javer@eng.ox.ac.uk',
        url              = "https://github.com/ver228/cell_localization",
        description      = 'Cell Localization using U-Net',
        install_requires = [], #I want to use annaconda packages so for the moment i do not want to specify dependencies since they will be force to be installed by pip
    )

if __name__ == "__main__":
    main()
