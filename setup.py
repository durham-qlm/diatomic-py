import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="diatomic-py",
    version="1.0.2",
    author="Jake Blackmore",
    author_email="jacob.blackmore@physics.ox.ac.uk",
    description="A package for calculating rotational and hyperfine structure of singlet diatomic molecules.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JakeBlackmore/Diatomic-Py",
    packages= ['diatomic'],
    license = 'Boost Software License - Version 1.0',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy>=1.19','sympy>=1.4','scipy>=1.1'],
    python_requires='>=3.7',
)
