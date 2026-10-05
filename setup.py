from setuptools import setup, find_packages

setup(
    name="gist-memory",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
    ],
    extras_require={
        "transformers": ["transformers>=4.36.0", "accelerate>=0.25.0"],
        "dev": ["pytest>=7.0.0", "transformers>=4.36.0", "accelerate>=0.25.0"],
    },
)
