"""
Setup script for VLM Agent BASALT
"""

from setuptools import setup, find_packages

setup(
    name="vlm-agent-basalt",
    version="0.1.0",
    description="VLM Agent for MineRL BASALT tasks using QWEN VLM",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        # Add your dependencies here if needed
        # Note: You already have environment.yml for conda
    ],
)
