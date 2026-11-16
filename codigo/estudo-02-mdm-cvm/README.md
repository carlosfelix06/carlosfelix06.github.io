# Estudo 02 · Arquitetura MDM & Golden Record na Base CVM

Cruzamento reproduzível de `7.905` registros públicos da CVM (`6.736` CNPJs únicos, `-14,8%`) comparando regras de sobrevivência (`Confiança Regulatória` vs `Recência`) e modos de sincronismo.

## Como Reproduzir

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s testes -v
python src/experimento_mdm_cvm.py
```

- Motor reutilizável relacionado: https://github.com/carlosfelix06/central-ia-pessoal/blob/main/central/mdm.py
- Página publicada: https://carlosfelix06.github.io/estudo-mdm-golden-record.html
