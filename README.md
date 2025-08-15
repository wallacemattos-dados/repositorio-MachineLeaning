# repositorio-MachineLeaning
Repositório de ML para previsão de vendas (séries temporais) com os datasets M5 e Favorita. Arquitetura MLOps com Docker, MLflow, DVC, GitHub Actions e modelos avançados (LightGBM, Transformer).
# Projeto de Previsão de Vendas (Sales Forecasting)

![Banner]([https://i.imgur.com/vB9aK8E.png](https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExODZ1em0wY2hsdjFiMmE1bWdxMDViY281c2I0OWRidzQyYnoybTFjcCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/be8ktX65xLhlpHTpIE/giphy.gif))

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python" alt="Python Version">
  <img src="https://img.shields.io/badge/Docker-20.10%2B-blue?logo=docker" alt="Docker Version">
  <img src="https://img.shields.io/badge/MLflow-2.0%2B-green?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAEbklEQVR4Xu2bXWwcVRzHP+fs7uxu7G7VbU3qFG0DFpDWgGIJfCh4KUgM8EFPfBDxgQclEvGBAyFe8EEoUAKiCMQngAQt3VA8VIwJDbWJQEiMaSNNGjU2tT+2e9+d2Zk7F4vbtXbXm2I1nCRdZN7v/M7//OzM/B9HhL84Quw5oQzAVgAsB2AZgA8AaAagBIAyADsBaAPw3T8ArAEQBsA5AO0A/DqAqwD8bB6z+8iAUMy0A8BVAJ4F8OAApAMwB4BDADwHIHZuF2fM0AHAWQBuA7AFQAeASwCcAnBvJszqo9QYm9gBIA/AHQBWAvgNQBeAXwG4FMApADbN/jrvF8MABsIUAJMAfA3gZwA2AvA4gMMAbKTA+wI8COBMAHeMbX/n/F8QGAAkAGQBeBLAswC8Yg7TQQOA4wD8xTzm8J4iMHQAIArAhwB8C+B/AN4D8FMA/jQHv4U5TJrT0wA+BcDTAH4K4C4AN2f2d/YABgTqAPAEgL8A+CGAZwE8AuCLdZp3qR5gI6OAbwC8C+Du/XjKAEBKk/NTLcDvAVwI4FwApzZ5r9sT0gAGQlMApgL4IYC/AHgYwO1aHWLzBsi4dE0BeBTArbN/o7s1Y4AGoQzATADfAbgdwNcAfgLwm1YnKAErSgA43+yB7j6MBWig9AE4F8C9AL4K4A8AVqv1GwsdAGQpAPubPVTdBQwbaALAsQB+CeAvgLsBXG7Wz2gDRkQBWCrB291TMxYYNBcAWAnB292T8xgwYgSQBcBvARxP8NbuaYkBDQYzAKwE4O3uqQkDGhA+A+BIAHeL895v7s8Y0ECoA+BUAHeJ8/5v7s8YaADYK057t7tvYkADwNY57d3uvokBDQA7xbl3u/siBjQg3AGgr3ju/e6+iAENAD/Hef8391sMaEDYCcA/4rj/m/stBjQg/BDAn+K8/5v7LQY0IPwXwK9x3v/N/RYDGhC+CeBPcd7/zf0WAxoQfgngT3He/839FgMaEHoBfBTH/d/cbzGgAeE3AH6P4/5v7rcY0IDwLgB/iOP+b+63GNCAsBeAL+K4/5v7LQY0IBwF4As47v/mvosBDQivAfASjuu/ud9iQAPCLQA+gOP+b+63GNCAsBCAB3Dcf839FgMaEJYA8BCO+7+532JAA8JvATyG4/5v7rcY0ECoA+B0HA+b+y0GNCCsAfAEjsfM/RYDGhBWAbgKx+Nmn8eABoQVAJ7G8bLZp3F0QANCC8C/43T2gVw7pAENCD8A8A8cr5t9pNcOaEDYBODvcLxu9pFeOxpoQPgYwL9wvG72kV47GmhA+AHA7+J8hsyqHTU0IHwNwD/i+AyZVTsqaED4PIDfx/EZMqt2VNCAsBbA3+I8Q2bVDgUaED4DYD6P12cW7dihAeE6gP/Fc4bMuh0VNCCsBfB3OM+QWbWjgAaEtwP4JZy3P2MDBoR3AXwF5z1gYAMGhFcBvILzHjAwaAD+h+MRMLt2dNCAsAzAn+L4DJlVOypoQNjt83xW/w6kF8+D+wAAAABJRU5ErkJggg==" alt="MLflow">
  <img src="https://img.shields.io/badge/DVC-3.0%2B-green?logo=dvc" alt="DVC">
  <img src="https://img.shields.io/github/workflow/status/{SEU_USUARIO}/{SEU_REPOSITORIO}/CI?label=CI&logo=github" alt="CI Status">
</p>

## Tabela de Conteúdos
1. [Sobre o Projeto](#1-sobre-o-projeto)
2. [Arquitetura e Tecnologias](#2-arquitetura-e-tecnologias)
3. [Começando](#3-começando)
   - [Pré-requisitos](#pré-requisitos)
   - [Instalação](#instalação)
4. [Como Usar](#4-como-usar)
   - [Acessando os Serviços](#acessando-os-serviços)
   - [Executando o Pipeline de Dados](#executando-o-pipeline-de-dados)
   - [Executando os Testes](#executando-os-testes)
5. [Estrutura do Repositório](#5-estrutura-do-repositório)
6. [Como Contribuir](#6-como-contribuir)
7. [Contato](#7-contato)

---

### 1. Sobre o Projeto

Este projeto tem como objetivo desenvolver um sistema de previsão de vendas de ponta a ponta, utilizando dados públicos das competições [M5 Forecasting](https://www.kaggle.com/c/m5-forecasting-accuracy) e [Corporación Favorita Grocery Sales Forecasting](https://www.kaggle.com/c/favorita-grocery-sales-forecasting).

Nosso foco é construir uma base sólida e escalável de MLOps para suportar o ciclo de vida completo do modelo, desde a ingestão e versionamento dos dados até o treinamento, rastreamento de experimentos e, futuramente, o deploy do modelo como um serviço.

### 2. Arquitetura e Tecnologias

Utilizamos um conjunto de ferramentas modernas para garantir um ambiente de desenvolvimento robusto, reprodutível e automatizado.

* **Docker & Docker Compose:** Containeriza toda a aplicação e seus serviços (MLflow, MinIO), garantindo que o ambiente de desenvolvimento seja idêntico para todos os membros da equipe e possa ser iniciado com um único comando.
* **MLflow:** Para rastrear nossos experimentos de machine learning. Ele registra parâmetros, métricas, artefatos (modelos, gráficos) e o código associado a cada execução, permitindo total reprodutibilidade e comparação de resultados.
* **MinIO:** Um servidor de armazenamento de objetos de alta performance, compatível com a API do Amazon S3. Usamos como nosso Data Lake local para armazenar datasets brutos, processados e artefatos do MLflow.
* **DVC (Data Version Control):** "Git para dados". O DVC nos permite versionar grandes arquivos de dados e modelos sem sobrecarregar o repositório Git, conectando o código às diferentes versões dos dados.
* **Pytest:** Framework para escrever testes de unidade e integração, garantindo a qualidade e a confiabilidade das nossas funções de ETL e engenharia de atributos.
* **GitHub Actions:** Automação de CI/CD (Integração Contínua) diretamente no GitHub. Usamos para rodar os testes automaticamente a cada `push` ou `pull request`, garantindo que novo código não quebre a funcionalidade existente.

### 3. Começando

Siga estes passos para configurar e executar o ambiente de desenvolvimento em sua máquina local.

#### Pré-requisitos

Certifique-se de ter os seguintes softwares instalados:
* [Git](https://git-scm.com/downloads)
* [Docker](https://docs.docker.com/get-docker/)
* [Docker Compose](https://docs.docker.com/compose/install/) (geralmente já vem com o Docker Desktop)

#### Instalação

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/](https://github.com/){SEU_USUARIO}/{SEU_REPOSITORIO}.git
    cd {SEU_REPOSITORIO}
    ```

2.  **Configure as Variáveis de Ambiente:**
    Copie o arquivo de exemplo `.env.example` para um novo arquivo chamado `.env`. Este arquivo guardará as credenciais dos nossos serviços e não deve ser enviado para o Git.
    ```bash
    cp .env.example .env
    ```
    O arquivo `.env` já virá com valores padrão para o ambiente local. Você não precisa alterá-lo para começar.

    <details>
      <summary>Conteúdo do .env.example</summary>

      ```dotenv
      # Credenciais para o MinIO (Data Lake)
      MINIO_ROOT_USER=minioadmin
      MINIO_ROOT_PASSWORD=minioadmin
      
      # URI para o MLflow Tracking Server
      MLFLOW_TRACKING_URI=http://mlflow:5000
      
      # Configuração para o DVC se comunicar com o MinIO
      MLFLOW_S3_ENDPOINT_URL=http://minio:9000
      ```
    </details>

3.  **Inicie os serviços com Docker Compose:**
    Este comando irá construir as imagens Docker (se ainda não existirem) e iniciar os contêineres da API, MLflow e MinIO em segundo plano (`-d`).
    ```bash
    docker-compose up -d --build
    ```

Pronto! Seu ambiente de desenvolvimento está no ar.

### 4. Como Usar

#### Acessando os Serviços
Após iniciar os contêineres, você pode acessar as interfaces dos serviços no seu navegador:

* **MLflow UI:** [http://localhost:5000](http://localhost:5000)
    * Aqui você poderá ver todos os experimentos e suas execuções.
* **MinIO Console:** [http://localhost:9001](http://localhost:9001)
    * **Login:** `minioadmin`
    * **Senha:** `minioadmin`
    * Use o console para visualizar os "buckets" (pastas) e os arquivos do nosso Data Lake.

#### Executando o Pipeline de Dados
Para baixar os dados brutos, processá-los e gerar os datasets unificados e enriquecidos, use o DVC. O arquivo `dvc.yaml` define todos os estágios do pipeline.

```bash
# Entrar no container da API onde o DVC está configurado
docker-compose exec api bash

# Dentro do container, executar o pipeline
dvc repro
```
O comando `dvc repro` irá verificar as dependências e executar apenas os estágios que foram alterados (código ou dados), salvando o resultado final no MinIO.

#### Executando os Testes
Para garantir que tudo está funcionando como esperado, você pode rodar a suíte de testes com `pytest`.

```bash
# Entrar no container da API
docker-compose exec api bash

# Dentro do container, executar os testes
pytest
```

### 5. Estrutura do Repositório
```
.
├── .github/workflows/      # Arquivos de automação do GitHub Actions (CI)
├── data/                   # Dados brutos e processados (gerenciados pelo DVC)
├── notebooks/              # Jupyter Notebooks para análise exploratória e experimentação
├── src/                    # Código fonte do projeto
│   ├── etl/                # Scripts de Extração, Transformação e Carga
│   ├── features/           # Módulos de engenharia de atributos
│   └── training/           # Scripts de treinamento e avaliação de modelos
├── tests/                  # Testes de unidade e integração
├── .dockerignore           # Arquivos a serem ignorados pelo Docker
├── .env.example            # Arquivo de exemplo para variáveis de ambiente
├── .gitignore              # Arquivos a serem ignorados pelo Git
├── docker-compose.yml      # Orquestração dos serviços Docker
├── dvc.yaml                # Definição do pipeline do DVC
├── Dockerfile              # Definição da imagem Docker da nossa API/serviço
└── README.md               # Este arquivo :)
```

### 6. Como Contribuir

Agradecemos o interesse em contribuir! Siga nosso fluxo de trabalho para manter o projeto organizado:

1.  **Crie uma branch:** A partir da branch `develop`, crie uma nova branch para sua feature ou correção (ex: `feature/nome-da-feature` ou `fix/nome-do-bug`).
    ```bash
    git checkout develop
    git pull
    git checkout -b feature/minha-nova-feature
    ```
2.  **Faça suas alterações:** Implemente o código, adicione testes e documente o que for necessário.
3.  **Faça o commit:** Escreva mensagens de commit claras e descritivas.
4.  **Envie para o repositório:**
    ```bash
    git push origin feature/minha-nova-feature
    ```
5.  **Abra um Pull Request:** Vá para o GitHub e abra um Pull Request da sua branch para a `develop`. Aguarde a revisão do código e a passagem dos testes de CI.

### 7. Contato

**Engenheiro de ML:** {Wallace Mattos} - {wallacemattos5963@gmail.com}
