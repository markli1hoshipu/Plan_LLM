from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="plan_llm",
    version="0.1.0",
    author="Jinbang Huang, Zhiyuan Li, et al.",
    author_email="",
    description="Self-CriTeach: LLM Self-Teaching for PDDL Planning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/markli1hoshipu/Plan_LLM",
    project_urls={
        "Bug Tracker": "https://github.com/markli1hoshipu/Plan_LLM/issues",
        "Paper": "https://openreview.net/forum?id=8I0n20ufAy",
        "Models": "https://huggingface.co/self-criteach",
        "Dataset": "https://huggingface.co/datasets/self-criteach/pddl-planning-data",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
)
