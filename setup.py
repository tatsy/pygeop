from setuptools import setup, find_packages
from setuptools.command.install import install

class Installer(install):
    def run(self):
        super(Installer, self).run()

setup(
    cmdclass={ 'install' : Installer },
    name='pygeop',
    version='0.1.0',
    author='tatsy',
    author_email='tatsy.mail@gmail.com',
    url='https://github.com/tatsy/hydra.git',
    description='Geometry processing library for Python',
    license='MIT',
    classfiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5'
    ],
    package=find_packages()
)
