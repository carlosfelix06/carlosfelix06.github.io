# Estudo 04 · Backtest Walk-Forward e Piso de Ruído no Crédito PF (BCB SGS)

Validação walk-forward em janela expansiva de `33` origens out-of-sample (`01/2024–09/2026`) sobre `57` meses reais do Banco Central comparando 4 modelos (`Ingênuo t-1`, `Sazonal t-12`, `Holt-Winters ETS` e `Ridge Sazonal + Dias Úteis ANBIMA`), alcançando `wMAPE 2,79%` (`-38%` vs `4,52%` Ingênuo) e `MAE R$ 6,24 bi/mês`.

## Como Reproduzir

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s testes -v
python src/backtest_projecao_walkforward.py
# Para atualizar as séries ao vivo na API SGS do Banco Central:
python src/backtest_projecao_walkforward.py --refresh
```

Página publicada: https://carlosfelix06.github.io/estudo-backtest-projecao-pedidos.html
