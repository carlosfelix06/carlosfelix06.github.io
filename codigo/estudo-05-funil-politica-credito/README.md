# Estudo 05 — Funil de Política de Crédito (R1→R4), Matriz 3×3 e Otimização de P&L

- **Fonte pública:** Give Me Some Credit (`OpenML 46929`, `150.000` registros históricos; split estratificado `90.000 / 30.000 / 30.000`).
- **Principais resultados (simulação contrafactual OOS):** RF bruto nativo entrega `AUC 0,8044` (`ECE 0,00123`) e Combinação 2D soma `+0,0071` (`AUC 0,8116`, `p < 0,001`). Otimizada 2D gera `5.051.520 u.m.` (`4,14% P&L/EAD` em `121,98M` EAD vs `4.706.550` RF 1D @ mesmo EAD e `4.553.250` Rígida), vencendo também sob `117,32M` EAD (`4,62M` vs `4,59M`) e isolando `1.794` casos Early Warning (`6,35%` BR).

```bash
pip install -r requirements.txt
python src/funil_politica_credito.py
python -m unittest discover -s testes -v
```

Página publicada: https://carlosfelix06.github.io/estudo-funil-politica-credito.html
