import setuptools

with open("README.md", 'r') as fh:
    long_descrip = fh.read()

setuptools.setup(
    name="pyautomodel",
    version='0.13',
    description="Automatic Machine Learning Model training",
    long_description=long_descrip,
    long_description_content_type="text/markdown",
    author='Kai Mo',
    author_email='kam455@pitt.edu',
    maintainer='Kai Mo',
    maintainer_email='kam455@pitt.edu',
    url='https://github.com/kaimo455/Auto_model.git',
    download_url='https://github.com/kaimo455/Auto_model.git',
    packages=setuptools.find_packages(),
    #py_modules=['lightgbm_optimizer'],
    #scripts='placeholder',
    #ext_modules='placeholder',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
    ],
)
