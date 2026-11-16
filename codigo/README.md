# Código-Fonte e Dados Reproduzíveis dos Estudos Públicos

Este diretório reúne os scripts Python, testes automatizados e snapshots de bases públicas oficiais necessários para auditar e replicar localmente os **5 estudos analíticos** publicados no portfólio (`https://carlosfelix06.github.io/`).

| Estudo | Diretório Público | Fontes Públicas Utilizadas | Execução Principal |
| --- | --- | --- | --- |
| **01 · Qualidade de Dados e Cobrança** | [`estudo-01-qualidade-dados/`](estudo-01-qualidade-dados/) | UCI Credit Default (`30.000` clientes) | `python src/experimento_ausencias.py` |
| **02 · Arquitetura MDM &amp; Golden Record** | [`estudo-02-mdm-cvm/`](estudo-02-mdm-cvm/) | Dados Abertos CVM (`7.905` registros) | `python src/experimento_mdm_cvm.py` |
| **03 · Alta Renda (27 UFs) &amp; 138 Bancos** | [`estudo-03-alta-renda-credito-pf/`](estudo-03-alta-renda-credito-pf/) | IBGE PNAD (`7438`) + BCB SGS + BCB Olinda | `python src/estudo_alta_renda_publico.py` |
| **04 · Backtest Walk-Forward Crédito PF** | [`estudo-04-backtest-credito-pf/`](estudo-04-backtest-credito-pf/) | BCB SGS Concessões PF (`57` meses) | `python src/backtest_projecao_walkforward.py` |
| **05 · Funil de Crédito, Matriz 3×3 &amp; P&amp;L** | [`estudo-05-funil-politica-credito/`](estudo-05-funil-politica-credito/) | Give Me Some Credit (OpenML `46929`, `150.000` registros históricos) | `python src/funil_politica_credito.py` |
