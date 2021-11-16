import setuptools

setuptools.setup(
    name="PoreMapper",
    version="0.0.1",
    author="Andrew Tarzia",
    author_email="andrew.tarzia@gmail.com",
    description="Cavity size and shape evaluation by bead growth.",
    url="https://github.com/andrewtarzia/PoreMapper",
    packages=setuptools.find_packages(),
    install_requires=(
        'numpy',
        'scipy',
        'sklearn',
    ),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
