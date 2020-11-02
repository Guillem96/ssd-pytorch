from setuptools.extension import Extension
from setuptools import setup, find_packages, Command
from os import path
from io import open
from distutils.command.build_ext import build_ext as DistUtilsBuildExt


here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open(path.join(here, 'requirements.txt')) as f:
    requires = [r for r in f.readlines() if not r.startswith('#')]

setup(
    name='ssd-pytorch',

    version='0.1',
    description='Modern PyTorch SSD implementation',
    long_description=long_description,
    long_description_content_type='text/markdown',

    url='https://github.com/Guillem96/ssd-pytorch',

    author='Guillem96 - Guillem Orellana Trullols',
    author_email='guillem.orellana@gmail.com', 

    classifiers=[
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Data Scientists - Deep Learning Engineers',
        'Topic :: Deep Learning',

        # Pick your license as you wish
        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3.6',
    ],

    keywords='object-detection ssd priorboxes vgg detection',

    packages=find_packages(exclude=['test', 'test.*']),
    python_requires='>=3.6',

    install_requires=requires,

    project_urls={ 
        'Source': 'https://github.com/Guillem96/ssd-pytorch',
    },
)