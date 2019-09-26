from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='monkeyfrog',
    version='1.0.7',
    description='Package for processing and analyzing fiber photometry and behavioral data',
    url='http://github.com/dmbrady01/Monkey_frog',
    author='DM Brady <dmbrady01@gmail.com> ',
    packages=find_packages(),
    install_requires=required,
    test_suite='nose.collector',
    tests_require=['nose'],
    zip_safe=False)
