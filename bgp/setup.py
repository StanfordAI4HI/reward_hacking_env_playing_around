from setuptools import setup, find_packages

setup(
    name="simglucose",
    version="0.1.0",
    description="Blood glucose control environment with RL wrappers",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "gymnasium",
        "pandas",
    ],
    python_requires=">=3.9",
)
