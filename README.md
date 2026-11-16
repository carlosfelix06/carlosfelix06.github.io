# Portfólio de Dados, Analytics & Risco de Crédito — Carlos Santos

[![CI - Testes e Validação Científica dos Estudos](https://github.com/carlosfelix06/carlosfelix06.github.io/actions/workflows/ci.yml/badge.svg)](https://github.com/carlosfelix06/carlosfelix06.github.io/actions/workflows/ci.yml)
[![Pytest](https://img.shields.io/badge/pytest-22%20passed%20(100%25)-success?logo=pytest)](https://github.com/carlosfelix06/carlosfelix06.github.io/tree/main/codigo)
[![Python](https://img.shields.io/badge/python-3.11-blue?logo=python)](https://www.python.org/)
[![GitHub Pages](https://img.shields.io/badge/pages-live%20production-success?logo=github)](https://carlosfelix06.github.io/)
[![Currículo Oficial PDF](https://img.shields.io/badge/Curr%C3%ADculo-PDF%20(441%20KB)-D91568?logo=adobeacrobatreader)](https://carlosfelix06.github.io/assets/curriculo-carlos-santos-dados-analytics.pdf)

> **Ambiente de Produção:** [https://carlosfelix06.github.io](https://carlosfelix06.github.io)  
> **Autor:** Carlos Alberto Felix Santos · Especialista em Dados & Analytics (São Paulo - SP)  
> **Contato:** [LinkedIn](https://www.linkedin.com/in/carlos-felix-santos-sp-br/) · [E-mail](mailto:carlosfelix06@hotmail.com)

---

## 1. Posicionamento Profissional & Trajetória Corporativa

Atuo na interseção entre **Dados, Analytics e Risco Financeiro**, traduzindo regras de negócio e bases transacionais em esteiras automatizadas de decisão, indicadores confiáveis e mitigação de perdas operacionais.

### Histórico Profissional em Grandes Instituições:
* **BTG Pactual (2025 – Atual) · Especialista em Dados**  
  Análise de dados de crédito, receita e produtos financeiros; diagnóstico de desvios em funis e KPIs operacionais; reconciliação e qualidade de dados com Python, SQL, Databricks e Apache Spark.
* **CredSystem Administradora de Cartões (2023 – 2024) · Cientista de Dados Sênior**  
  Modelagem e análise de risco de crédito; simulação de cenários de inadimplência e rentabilidade; relatórios analíticos em Power BI; preparação e análise com SAS, SQL, Databricks e Python (Random Forest).
* **Banco Bradesco (2022 – 2023) · Cientista de Dados Sênior**  
  Modelagem de risco de crédito; simulações de KPIs de carteira; gestão e otimização de rotinas em SAS; pipelines com Teradata, Hive, SQL e automações estatísticas em Python.
* **Ernst & Young — EY (2021 – 2022) · Consultor FSO Sênior (Cliente: Depto de Políticas de Crédito do Bradesco)**  
  Consultoria em modelagem e análise de risco; simulação contrafactual para políticas de crédito; apresentação em comitês executivos; rotinas em SAS, SQL e Python.
* **Marfrig Global Foods / Concentrix (SulAmérica) / Vermont (Carrefour) (2017 – 2021) · Especialista e Analista Sênior MIS/BI**  
  Modelagem dimensional de Data Warehouses, conciliação contábil-operacional, dimensionamento e forecast com SQL Server, MySQL, Power BI/DAX e VBA.

---

## 2. Laboratório de Pesquisa Aplicada & Benchmarks Autorais

Este repositório contém os códigos-fonte, datasets e pipelines de **5 estudos analíticos independentes** desenvolvidos com **bases públicas abertas** (BCB SGS, CVM, IBGE, UCI Repository e GiveMeSomeCredit). 

> **Aviso de Rigor Metodológico:** Os ganhos numéricos de simulação (como o P&L de +10,94% no Estudo 05 ou a redução de erro de R$ 7,39 bi/mês no Estudo 04) referem-se estritamente aos experimentos contrafactuais documentados nestes benchmarks públicos, não a dados sigilosos ou projetos proprietários das instituições empregadoras.

| Estudo | Domínio & Base | Principal Descoberta / Métrica | Algoritmo / Método | Código & Testes |
| :--- | :--- | :--- | :--- | :---: |
| **01 · Qualidade dos Dados** | UCI Credit Card (30k clientes) | Queda de -0,042 ROC-AUC sob ruído; recuperação simulada de NT$ 6,48M (97,6% do limpo). | HistGradientBoosting (NaN nativo) vs Imputação | [`codigo/estudo-01`](codigo/estudo-01-qualidade-dados) |
| **02 · MDM & Golden Record** | Dados Abertos CVM (7.905 registros) | 81,57% recall fim a fim; 33 de 42 conflitos de status divergentes entre recência e regras. | Record Linkage Híbrido (HGB) + Survivorship | [`codigo/estudo-02`](codigo/estudo-02-mdm-cvm) |
| **03 · Taxas & Renda Regional** | BCB SGS & IBGE PNADc (138 bancos) | Dispersão de 15,57x nos juros (Spearman ρ = 0,8656); 3 clusters macroeconômicos. | K-Means + Efeitos Fixos UF | [`codigo/estudo-03`](codigo/estudo-03-alta-renda-credito-pf) |
| **04 · Forecast de Crédito PF** | Banco Central SGS (180 meses) | MAE cai de R$ 13,63 bi para R$ 6,24 bi/mês (-54,2%); -R$ 1,23 bi/mês atribuído aos feriados ANBIMA. | Ridge Sazonal + Walk-Forward (33 origens) | [`codigo/estudo-04`](codigo/estudo-04-backtest-credito-pf) |
| **05 · Política de Crédito** | GiveMeSomeCredit (150k registros) | +10,94% P&L simulado (+5,05M u.m.); Bad Rate EAD de 1,054% (0,685% perda LGD 65%). | Esteira 2D (Score LP × RF CP) + Matriz 3×3 | [`codigo/estudo-05`](codigo/estudo-05-funil-politica-credito) |

---

## 3. Reprodutibilidade Científica & Execução Local

Todos os experimentos contam com baterias de testes unitários automatizados em `pytest`, validadas a cada commit via GitHub Actions.

```bash
# 1. Clonar o repositório
git clone https://github.com/carlosfelix06/carlosfelix06.github.io.git
cd carlosfelix06.github.io

# 2. Criar e ativar ambiente virtual Python (3.10 ou 3.11 recomendado)
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Instalar dependências científicas
pip install pytest numpy scipy pandas scikit-learn statsmodels

# 4. Executar toda a suíte de testes (22 testes auditáveis)
pytest codigo -v
```

---

## 4. Estrutura do Repositório

```text
carlosfelix06.github.io/
├── .github/workflows/ci.yml       # Automação de CI (executa pytest a cada push)
├── assets/                        # SVGs, PNGs (Open Graph 1200x627), CSS, JS e Currículo PDF
├── codigo/                        # Scripts e testes reproduzíveis dos 5 estudos
│   ├── estudo-01-qualidade-dados/
│   ├── estudo-02-mdm-cvm/
│   ├── estudo-03-alta-renda-credito-pf/
│   ├── estudo-04-backtest-credito-pf/
│   └── estudo-05-funil-politica-credito/
├── index.html                     # Homepage com Trajetória Corporativa e Laboratório
├── estudos.html                   # Vitrine consolidada dos 5 estudos
├── estudo-qualidade-dados-inadimplencia.html
├── estudo-mdm-golden-record.html
├── estudo-alta-renda-nichos.html
├── estudo-backtest-projecao-pedidos.html
└── estudo-funil-politica-credito.html
```

---

## 5. Licença & Direitos

Códigos e metodologias abertos para fins de demonstração técnica, pesquisa e auditoria sob licença MIT. Dados originais sob os termos de uso de suas respectivas fontes oficiais (BCB, CVM, IBGE, UCI e OpenML).
