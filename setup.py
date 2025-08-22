from setuptools import setup, find_packages

setup(
    name="mlproject",
    version="0.0.1",
    description="end to end project",
    author="shayhari",
    author_email="shayharirv@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "jupyter"
    ],
    python_requires=">=3.11",
)