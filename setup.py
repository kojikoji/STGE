import setuptools


def _requires_from_file(filename):
    return open(filename).read().splitlines()


with open("README.md", "r") as fh:
    long_description = fh.read()

    setuptools.setup(
        name="setup-kojikoji",
        version="0.0.1",
        author="kojikoji",
        author_email="tiisaishima@gmail.com",
        description="spatiotemporal gene expression reconstruction",
        long_description=long_description,
        long_description_content_type="text/markdown",
        packages=setuptools.find_packages(),
        install_requires=_requires_from_file('requirements.txt'),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
        ],
        python_requires='>=3.6',
    )
