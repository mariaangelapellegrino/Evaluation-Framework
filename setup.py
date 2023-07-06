from setuptools import setup, find_packages

with open("pip_readme.md", "r") as f:
    long_description = f.read()

setup(
    name="evaluation_framework",
    packages=find_packages(),
    include_package_data=True,
    # packages_data={"": ["data/*.tsv", "data/*.csv", "data/*.txt"]},
    version="2.0",
    license="apache-2.0",
    description="Evaluation Framework for testing and comparing graph embedding techniques",
    long_description=long_description,
    author="Maria Angela Pellegrino",
    author_email="mariaangelapellegrino94@gmail.com",
    url="https://github.com/mariaangelapellegrino/Evaluation-Framework",
    download_url="https://github.com/mariaangelapellegrino/Evaluation-Framework/archive/v_2.0.tar.gz",
    keywords=[
        "evaluation",
        "graph-embedding",
        "python",
        "library",
        "benchmark-framework",
        "machine learning tasks",
        "semantic tasks",
    ],  # Keywords that define your package best
    install_requires=[
        "scikit-learn==0.23.1",
        "scipy==1.10.0",
        "h5py==2.10.0",
        "numpy==1.22.0",
        "pandas==1.0.5",
        "pathlib2",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.7",
    ],
)
