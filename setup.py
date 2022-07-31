# -*- coding: utf-8 -*-
"""
| Author: Alexandre CARRE
| Created on: Jan 13, 2021
"""

from setuptools import setup, find_packages

console_scripts = ['qant-extractor=QAnT.extractor:main',
                   'qant-interface=QAnT.interface:main',
                   ]

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

args = dict(
    name='QAnT',
    version='0.0.1',
    description="image Quality Assessment aNd dicom Tags extraction",
    long_description=readme,
    author='Alexandre CARRE',
    url='https://github.com/Alxaline/QAnT',
    license=license,
    packages=find_packages(exclude=['docs']),
    package_data={'QAnT': ['schemas/param_schema.yaml', 'schemas/schema_funcs.py',
                           'example_parameters/default_parameters.yaml']},
    python_requires='>=3.6',
    entry_points={
        'console_scripts': console_scripts
    },
    keywords="Quality Assessment metrics DICOM metadata tags nifti",
)

setup(install_requires=['waitress',
                        'pydicom',
                        'pykwalify',
                        'numba',
                        'numpy',
                        'pandas',
                        'scikit-image',
                        'scipy',
                        'torch',
                        'six',
                        'dash',
                        'plotly',
                        'simpleitk',
                        'monai',
                        'scikit-learn',
                        'ComScan @ git+https://github.com/Alxaline/ComScan.git#egg=ComScan',
                        ], **args)
