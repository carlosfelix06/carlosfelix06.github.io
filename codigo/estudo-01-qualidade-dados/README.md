# Estudo 01 · Impacto da Qualidade dos Dados no Risco de Inadimplência

Replicação completa do experimento pareado (`MCAR` vs `Concentrada` em `10%`, `30%`, `50%`) sobre a base pública **UCI Default of Credit Card Clients** (`30.000` clientes, split estratificado 60/20/20).

## Como Reproduzir

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s testes
python src/experimento_ausencias.py
python src/mitigacao_financeira.py
```

Página publicada: https://carlosfelix06.github.io/estudo-qualidade-dados-inadimplencia.html
