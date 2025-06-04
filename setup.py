from setuptools import setup, find_packages

setup(
    name="vassoura",
    version="0.1",
    packages=find_packages(),
    install_requires=["pandas", "numpy", "matplotlib", "seaborn"],
    author="Seu Nome",
    description="Pacote para limpeza de multicolinearidade em conjuntos de dados",
)
