from setuptools import setup, find_packages

setup(
    name='utpgetools',
    version='0.1.3',
    packages=find_packages(),
    install_requires=['numpy','pandas','matplotlib','rich'],
    author='Brecken Runquist',
    description='A collection of utility tools for your UT PGE projects.',
    url='https://github.com/bRunquist/utpgetools',
    license='Custom Academic Use License',
    python_requires='>=3.7' 
)
