from setuptools import setup, find_packages

setup(
    name="visual-anomaly-detection",
    version="0.1.0",
    author="Devarshi Dixit",
    author_email="dixitdevarshi16@gmail.com",
    description="Unsupervised visual anomaly detection using DINOv2 and PatchCore on MVTec AD",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
)