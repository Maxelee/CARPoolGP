from setuptools import setup

setup(
    name='CARPoolGP',
    version='0.1.0',    
    description='python package for correlated emulation',
    url='https://carpoolgp.readthedocs.io/en/latest/',
    author='Max E. Lee',
    author_email='max.e.lee@columbia.edu',
    license='BSD 2-clause',
    packages=['CARPoolGP'],
    install_requires=['jax',
                      'optax',
                      'scikit-learn',
                      'tinygp'
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.5',
    ],
)