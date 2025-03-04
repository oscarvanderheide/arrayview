from setuptools import setup, find_packages

setup(
    name="arrayshow",
    version="0.1.0",
    description="A tool for visualizing multi-dimensional arrays in Python",
    author="Oscar van der Heide",
    author_email="oscarvanderheide@gmail.com",
    packages=find_packages(),
    install_requires=["numpy>=1.18.0", "matplotlib>=3.1.0", "pyqt5>=5.15.11"],
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
)
