import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="diatom",
    version="1.0.1",
    author="Jake Blackmore",
    author_email="j.a.blackmore@durham.ac.uk",
    description="A package for calculating molecular hyperfine structure.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JakeBlackmore/Diatomic-Py",
    packages= ['diatom'],
    license = 'Boost Software License - Version 1.0',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy>=1.15','sympy>=1.4','scipy>=1.1'],
    python_requires='>=3.7',
)
