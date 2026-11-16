# Estudo 03 · Estrutura Competitiva de Juros Bancários e Assimetria Regional PF

Auditoria quantitativa em **dois grãos oficiais separados** (sem falácia ecológica para CPFs):
1. **Bancos (BCB Olinda `11/2025` vs `10/2026`, `138` instituições, `370` pareamentos idênticos):** Clusterização `K-Means + PCA` (`74` bancos com `>=3` modalidades) e persistência de 11 meses (`Spearman ρ = 0,8656`, retenção `80,2%` no quartil barato e `82,6%` no mais caro). Coeficientes de silhueta: `k=2 (0,3033)`, `k=3 (0,2418)` e `k=4 (0,2556)`; a escolha de `k=3` prioriza a interpretabilidade estratégica de negócio (baixo spread, carteiras garantidas e alta extração).
2. **Painel Estadual (`108` UF-anos `2022–2025`, IBGE SIDRA 7438 × BCB SGS):** Efeitos Fixos de Ano (`R² = 0,550`), β-convergência `2022–2026` (`p = 0,0023`, com risco de artefato de regressão à média) e rejeição out-of-sample em `2025` do modelo puro de renda (`MAE 1,278` vs `0,953 p.p.` da inércia; avaliação retrospectiva com dados anuais consolidados do IBGE).

## Como Reproduzir

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s testes -v
python src/estudo_alta_renda_publico.py
# Para atualizar via APIs oficiais do IBGE e Banco Central:
python src/estudo_alta_renda_publico.py --refresh
```

Página publicada: https://carlosfelix06.github.io/estudo-alta-renda-nichos.html
