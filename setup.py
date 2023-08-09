from setuptools import setup, find_packages

setup(
    name='timeseries_qmc',
    version='0.1.0',    
    description='A Python package for running the time series QMC algorithm',
    url='https://https://github.com/CQCL/timeseries-qmc',
    author='Quantinuum GmbH',
    author_email='khaldoon.ghanem@quantinuum.com',
    license='Apache License 2.0',
    packages=find_packages(),
    install_requires=['numpy',
                      'scipy',
                      'pytket',
                      'pytket-qiskit',
                      'physics-tenpy',                                 
                      ],
    dependency_links=['https://github.com/QuSpin/QuSpin/tree/master'],
    zip_safe=False,
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',  
        'Operating System :: POSIX :: Linux',  
        'Operating System :: MacOS :: MacOS X',       
        'Programming Language :: Python :: 3',
    ],
)