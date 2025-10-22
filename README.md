
<h1 align="center">🔍 Delegacia Senac</h1>

<p align="center">
  <img alt="dashboard screenshot" src="https://github-production-user-asset-6210df.s3.amazonaws.com/108770641/503912896-1371d5b7-d445-4aa9-bca6-bce449193607.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVCODYLSA53PQK4ZA%2F20251021%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20251021T225859Z&X-Amz-Expires=300&X-Amz-Signature=2dc1161220b35f9322e418d0b3fd905a49e08cf535e67884beb4823872377753&X-Amz-SignedHeaders=host" width="100%">
</p>

---
**Acesse nossa interface [aqui](https://delegaciasenac.streamlit.app/)**

**Delegacia Senac** é um projeto desenvolvido por alunos do curso de **Análise e Desenvolvimento de Sistemas**, com foco em aplicar técnicas de **Ciência de Dados** e **Inteligência Artificial** em um contexto real de segurança pública.  

O sistema permite a **análise exploratória**, **clusterização**, **detecção de anomalias** e **modelagem supervisionada** de ocorrências criminais, além de gerar relatórios interativos em HTML diretamente pela interface Streamlit.



---

<p align="center">
  <a href="#-estrutura">Estrutura Principal</a>&nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="#-tecnologias">Tecnologias</a>&nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="#-requisitos">Requisitos</a>&nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="#-instalação">Instalação</a>&nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="#-execução">Execução</a>&nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="#-dicas">Dicas</a>&nbsp;&nbsp;|&nbsp;&nbsp;
  <a href="#memo-licença">Licença</a>
</p>

<p align="center">
  <img alt="License" src="https://img.shields.io/static/v1?label=license&message=MIT&color=49AA26&labelColor=000000">
</p>

---

## 🚧 Estrutura Principal

```perl
delegacia5senac/
│
├── app/
│   └── app.py                    # Aplicação principal do Streamlit
│
├── data/
│   └── file.csv                  # Arquivo CSV com dados de ocorrências criminais
│
├── outputs/                      # Relatórios HTML exportados
│
├── main.py                       # Script para pré-processamento e geração de DataFrame
│
├── requirements.txt              # Dependências do Python
├── README.md                     # Documentação do projeto
└── .gitignore                    # Arquivos ignorados pelo Git
```

## 🚀 Tecnologias

- **Frontend e Visualização:** Streamlit  
- **Análise e Processamento:** PySpark e Pandas  
- **Machine Learning:** Scikit-learn 
- **Ambiente Virtual:** venv  
- **Deploy:** Streamlit Cloud  
- **Versionamento:** Git e GitHub  


## 📚 Requisitos

1. **Python 3.10+**  
   Certifique-se de ter o Python instalado.  
2. **Ambiente virtual (`venv`)**  
   Recomendado para isolar as dependências.  
3. **Bibliotecas necessárias:**  
   Incluídas no `requirements.txt` (ex: `streamlit`, `pandas`, `pyspark`, `scikit-learn`).


## 🔧 Instalação

Clone o repositório:

```bash
git clone https://github.com/qnvitor/delegacia5senac.git
```

### Configurar o ambiente

1. Crie e ative o ambiente virtual:

```bash
python -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate       # Windows
```

2. Instale as dependências:

```bash
pip install -r requirements.txt
```


## 💻 Execução

### Executar o dashboard Streamlit

Execute:

```bash
streamlit run app/app.py
```

O Streamlit abrirá automaticamente no navegador (geralmente em `http://localhost:8501`).


## 💡 Dicas

- O **main.py** pode ser executado **antes** do comando `streamlit run app/app.py`, para garantir que os dados estejam prontos para visualização e a aplicação carregue mais rapidamente.  
- Os relatórios gerados em HTML são salvos na pasta `app/outputs/`.    
- O botão **“📤 Exportar Relatório (HTML)”** gera automaticamente um arquivo com o resumo estatístico dos dados e permite **download direto pela interface**.

## :memo: Licença

Esse projeto está sob a licença MIT.

---

Feito com ♥ by Abyssal Roll :wave: