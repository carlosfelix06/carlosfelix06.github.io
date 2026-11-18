"""Simule três paradigmas de decisão de crédito PJ com dados sintéticos."""

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 61
TOTAL = 5_000
TICKETS = (5_000, 50_000, 250_000)
MARGEM = 0.12
PERDA = 0.55
LIMITE_RISCO = MARGEM / (MARGEM + PERDA)
PRECO_ENTRADA = 10.0
PRECO_SAIDA = 40.0
SAIDA = Path(__file__).resolve().parents[1] / "resultados" / "metricas_comparativo_ia_vs_ml.json"
logger = logging.getLogger(__name__)


def gerar_propostas():
    """Gere dados pré-decisão e sorteie a inadimplência posterior de cada PJ."""
    rng = np.random.default_rng(SEED)
    ticket = rng.choice(TICKETS, TOTAL)
    faturamento = rng.lognormal(np.log(180_000), 1.0, TOTAL)
    endividamento = rng.beta(3, 4, TOTAL)
    pontualidade = rng.beta(9, 2, TOTAL)
    restricoes = rng.poisson(0.3, TOTAL)
    fragilidade = rng.normal(0, 1, TOTAL)
    parecer = np.clip(70 - 17 * fragilidade + rng.normal(0, 10, TOTAL), 0, 100)
    exposicao = np.log1p(ticket / faturamento)
    log_odds = (
        -4.4 + 2.8 * endividamento + 3.5 * (1 - pontualidade)
        + 0.65 * restricoes + 0.9 * exposicao + 0.85 * fragilidade
    )
    return {
        "ticket": ticket,
        "faturamento": faturamento,
        "endividamento": endividamento,
        "pontualidade": pontualidade,
        "restricoes": restricoes,
        "parecer": parecer,
        "inadimplencia": rng.binomial(1, 1 / (1 + np.exp(-log_odds))),
    }


def pontuar_heuristica(propostas, teste):
    """Aplique regras ordenadas de risco, da faixa mais grave à mais leve."""
    riscos = []
    for i in teste:
        exposicao = propostas["ticket"][i] / propostas["faturamento"][i]
        divida = propostas["endividamento"][i]
        pontualidade = propostas["pontualidade"][i]
        restricoes = propostas["restricoes"][i]
        if restricoes >= 2 or exposicao > 2:
            risco = 0.45
        elif restricoes or divida > 0.65 or pontualidade < 0.70 or exposicao > 1:
            risco = 0.24
        elif divida > 0.45 or pontualidade < 0.85 or exposicao > 0.5:
            risco = 0.14
        else:
            risco = 0.06
        riscos.append(risco)
    return np.array(riscos)


def pontuar_modelos(propostas, treino, calibracao, teste):
    """Treine a regressão financeira e ajuste a combinação com o parecer."""
    financeiros = np.column_stack(
        (
            np.log(propostas["faturamento"]),
            np.log1p(propostas["ticket"] / propostas["faturamento"]),
            propostas["endividamento"],
            propostas["pontualidade"],
            propostas["restricoes"],
        )
    )
    alvo = propostas["inadimplencia"]
    modelo = make_pipeline(StandardScaler(), LogisticRegression(random_state=SEED))
    modelo.fit(financeiros[treino], alvo[treino])
    # O score de parecer é uma medida ruidosa da fragilidade pré-decisão.
    atributos_hibridos = np.column_stack(
        (modelo.decision_function(financeiros), propostas["parecer"])
    )
    hibrido = make_pipeline(StandardScaler(), LogisticRegression(random_state=SEED))
    hibrido.fit(atributos_hibridos[calibracao], alvo[calibracao])
    return (
        modelo.predict_proba(financeiros[teste])[:, 1],
        hibrido.predict_proba(atributos_hibridos[teste])[:, 1],
    )


def calcular_metricas(alvo, risco, custos, latencias, ticket):
    """Meça discriminação, aprovação, inadimplência, FinOps e retorno por ticket."""
    aprovadas = risco <= LIMITE_RISCO
    quantidade = int(aprovadas.sum())
    inadimplentes = int(alvo[aprovadas].sum())
    fpr, tpr, _ = roc_curve(alvo, risco)
    capital = quantidade * ticket
    custo_total = float(custos.sum())
    margem = (quantidade - inadimplentes) * ticket * MARGEM
    perda = inadimplentes * ticket * PERDA
    resultado = margem - perda - custo_total
    return {
        "propostas": len(alvo),
        "aprovadas": quantidade,
        "inadimplentes_aprovadas": inadimplentes,
        "ks": float(np.max(tpr - fpr)),
        "auc_roc": float(roc_auc_score(alvo, risco)),
        "taxa_aprovacao": quantidade / len(alvo),
        "bad_rate": inadimplentes / quantidade if quantidade else None,
        "custo_por_decisao_brl": float(custos.mean()),
        "latencia_media_ms": float(latencias.mean()),
        "capital_desembolsado_brl": capital,
        "custo_total_decisoes_brl": custo_total,
        "resultado_liquido_brl": resultado,
        "roi_liquido": resultado / capital if capital else None,
    }


def simular():
    """Compare as três políticas em teste comum e consolide as faixas de ticket."""
    propostas = gerar_propostas()
    alvo = propostas["inadimplencia"]
    desenvolvimento, teste = train_test_split(
        np.arange(TOTAL), test_size=0.20, random_state=SEED, stratify=alvo
    )
    treino, calibracao = train_test_split(
        desenvolvimento, test_size=0.25, random_state=SEED, stratify=alvo[desenvolvimento]
    )
    logger.info(
        "Propostas por amostra — treino %d, calibração %d e teste %d.",
        len(treino), len(calibracao), len(teste),
    )
    risco_ml, risco_hibrido = pontuar_modelos(propostas, treino, calibracao, teste)
    rng = np.random.default_rng(SEED + 1)
    entrada = rng.integers(400, 1_001, len(teste))
    saida = rng.integers(60, 201, len(teste))
    custo_ml, latencia_ml = 0.004, 8.0
    custo_tokens = (entrada * PRECO_ENTRADA + saida * PRECO_SAIDA) / 1_000_000
    latencia_hibrida = latencia_ml + 180 + entrada * 0.15 + saida * 12
    politicas = {
        "heuristica": (
            pontuar_heuristica(propostas, teste),
            np.full(len(teste), 0.001),
            np.full(len(teste), 2.0),
        ),
        "ml_tradicional": (
            risco_ml, np.full(len(teste), custo_ml), np.full(len(teste), latencia_ml)
        ),
        "ia_hibrida": (risco_hibrido, custo_ml + custo_tokens, latencia_hibrida),
    }

    resultados = []
    for nome, (risco, custos, latencias) in politicas.items():
        for ticket in TICKETS:
            faixa = propostas["ticket"][teste] == ticket
            metricas = calcular_metricas(
                alvo[teste][faixa], risco[faixa], custos[faixa], latencias[faixa], ticket
            )
            if nome == "ia_hibrida":
                metricas["tokens_entrada_medios"] = float(entrada[faixa].mean())
                metricas["tokens_saida_medios"] = float(saida[faixa].mean())
            resultados.append({"paradigma": nome, "ticket_brl": ticket, **metricas})

    return {
        "estudo": "KAN-61",
        "seed": SEED,
        "amostras": {
            "total": TOTAL, "treino": len(treino), "calibracao": len(calibracao),
            "teste": len(teste),
        },
        "premissas": {
            "dados": "Carteira sintética com desfecho simulado para todas as propostas.",
            "ia_hibrida": "Regressão financeira combinada com score de parecer sintético.",
            "finops": "Custos, preços de tokens e latências são premissas de simulação.",
            "margem_sobre_adimplentes": MARGEM,
            "perda_sobre_inadimplentes": PERDA,
            "limite_risco_aprovacao": LIMITE_RISCO,
            "custo_entrada_por_milhao_tokens_brl": PRECO_ENTRADA,
            "custo_saida_por_milhao_tokens_brl": PRECO_SAIDA,
            "roi": "Margem dos adimplentes menos perdas e custo de todas as decisões, "
                   "dividido pelo principal desembolsado.",
            "denominadores": "AUC e KS usam todas as propostas da faixa. Bad rate usa "
                             "as propostas aprovadas. Bad rate e ROI são nulos sem aprovações.",
            "unidades": "KS, AUC, taxas e ROI em proporção. Valores em reais e latência em ms.",
        },
        "resultados": resultados,
    }


def main():
    """Execute a simulação e exporte o comparativo consolidado em UTF-8."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    consolidado = simular()
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    conteudo = json.dumps(consolidado, ensure_ascii=False, indent=2, allow_nan=False)
    SAIDA.write_text(conteudo + "\n", encoding="utf-8")
    logger.info("Comparativo exportado para %s.", SAIDA)


if __name__ == "__main__":
    main()
