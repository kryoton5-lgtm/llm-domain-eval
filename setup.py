from setuptools import setup, find_packages

setup(
    name="llm-domain-eval",
    version="0.1.0",
    description="Adversarial evaluation and PEFT fine-tuning for domain-specific LLM QA",
    author="Pranjal Sunder Kadam",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.38.0",
        "datasets>=2.16.0",
        "peft>=0.8.0",
        "trl>=0.7.0",
        "pyyaml>=6.0",
        "numpy>=1.24.0",
    ],
)
