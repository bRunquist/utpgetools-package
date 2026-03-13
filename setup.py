from setuptools import setup, find_packages

setup(
    name='utpgetools',
    version='0.1.4',
    packages=find_packages(),
    install_requires=['numpy','pandas','matplotlib','rich','scipy','scikit-learn','seaborn'],
    author='Brecken Runquist',
    description='A collection of utility tools for your UT PGE projects.',
    url='https://github.com/bRunquist/utpgetools',
    license='Custom Academic Use License',
    python_requires='>=3.7' 
)
