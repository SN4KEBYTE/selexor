from setuptools import setup, find_packages

from selexor import __version__


def get_requirements():
    with open('requirements.txt') as inp:
        return [line for line in inp]


setup(
    name='selexor',
    version=__version__,

    description='Some useful algorithms for feature selection and extraction.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',

    url='https://github.com/SN4KEBYTE/selexor',
    author='Timur Kasimov',
    license='GNU General Public License v3.0',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Typing :: Typed',
    ],

    packages=find_packages(),
    install_requires=get_requirements(),
)
