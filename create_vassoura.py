import os

# Estrutura de pastas e arquivos com conteúdo opcional
estrutura = {
    "vassoura": {
        "__init__.py": "",
        "correlacao.py": "# Funções para análise de correlação\n",
        "vif.py": "# Funções para cálculo de VIF\n",
        "limpeza.py": "# Funções para limpeza de colinearidade\n",
        "relatorio.py": "# Funções para geração de relatório\n",
        "utils.py": "# Funções auxiliares\n"
    },
    "tests": {
        "test_correlacao.py": "# Testes para correlacao.py\n",
        "test_vif.py": "# Testes para vif.py\n",
        "test_limpeza.py": "# Testes para limpeza.py\n"
    },
    "examples": {
        "exemplo_uso.ipynb": ""  # você pode abrir no Jupyter depois
    },
    ".": {
        "README.md": "# Projeto Vassoura\n\nBiblioteca para análise de correlação e multicolinearidade.",
        "setup.py": """from setuptools import setup, find_packages

setup(
    name="vassoura",
    version="0.1",
    packages=find_packages(),
    install_requires=["pandas", "numpy", "matplotlib", "seaborn"],
    author="Seu Nome",
    description="Pacote para limpeza de multicolinearidade em conjuntos de dados",
)
""",
        "pyproject.toml": """[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"
"""
    }
}

def criar_estrutura(estrutura, base_path="."):
    for pasta, arquivos in estrutura.items():
        dir_path = os.path.join(base_path, pasta)
        os.makedirs(dir_path, exist_ok=True)
        for nome_arquivo, conteudo in arquivos.items():
            caminho_arquivo = os.path.join(dir_path, nome_arquivo)
            with open(caminho_arquivo, "w", encoding="utf-8") as f:
                f.write(conteudo)

if __name__ == "__main__":
    criar_estrutura(estrutura)
    print("Estrutura do projeto 'vassoura' criada com sucesso!")
