from setuptools import setup, find_packages

setup(
    name='ml_eng',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'seaborn',
        'numpy',
        'pandas',
        'scikit-learn'
    ],
    test_suite='tests',
)
