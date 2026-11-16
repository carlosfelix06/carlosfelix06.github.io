"""Estudo quantitativo reproduzivel: Estrutura Competitiva de Juros Bancarios (Olinda BCB)
e Assimetria Regional no Credito PF (Painel IBGE SIDRA 7438 x BCB SGS, 108 UF-anos).

Fontes Publicas Oficiais (grao instituicao financeira e UF — sem dados individuais de CPF):
1. IBGE SIDRA Tabela 7438 / Variavel 10769 (PNAD Continua 2022–2025): percentis P10..P99 em 27 UFs.
2. Banco Central SGS (15861..15887): inadimplencia PF (>90d) mensal por UF (2022–09/2026).
3. Banco Central SGS Nacional (2022–09/2026): series macro e saldo/juros/inadimplencia PF.
4. Banco Central Olinda (TaxasJurosDiariaPorInicioPeriodo): taxas por instituicao em 11/2025 e 10/2026.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import unicodedata
from urllib.parse import quote
import urllib.request

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, silhouette_score
from sklearn.preprocessing import StandardScaler

STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
RESULTS_DIR = STUDY_ROOT / "resultados"
SNAPSHOT_PATH = RESULTS_DIR / "snapshot_publico.json"
SITE_ASSETS_DIR = (
    REPO_ROOT
    / "estudos"
    / "202609 - Impacto da qualidade dos dados no risco de inadimplência"
    / "app"
    / "site_assets"
    / "assets"
)

BCB_UF_INAD_SERIES = [
    (15861, "AC", "Acre"),
    (15862, "AL", "Alagoas"),
    (15863, "AP", "Amapá"),
    (15864, "AM", "Amazonas"),
    (15865, "BA", "Bahia"),
    (15866, "CE", "Ceará"),
    (15867, "DF", "Distrito Federal"),
    (15868, "ES", "Espírito Santo"),
    (15869, "GO", "Goiás"),
    (15870, "MA", "Maranhão"),
    (15871, "MT", "Mato Grosso"),
    (15872, "MS", "Mato Grosso do Sul"),
    (15873, "MG", "Minas Gerais"),
    (15874, "PA", "Pará"),
    (15875, "PB", "Paraíba"),
    (15876, "PR", "Paraná"),
    (15877, "PE", "Pernambuco"),
    (15878, "PI", "Piauí"),
    (15879, "RJ", "Rio de Janeiro"),
    (15880, "RN", "Rio Grande do Norte"),
    (15881, "RS", "Rio Grande do Sul"),
    (15882, "RO", "Rondônia"),
    (15883, "RR", "Roraima"),
    (15884, "SC", "Santa Catarina"),
    (15885, "SP", "São Paulo"),
    (15886, "SE", "Sergipe"),
    (15887, "TO", "Tocantins"),
]

IBGE_YEARS = ("2022", "2023", "2024", "2025")
IBGE_UF_7438_URL = (
    "https://servicodados.ibge.gov.br/api/v3/agregados/7438/periodos/2022|2023|2024|2025/variaveis/10769"
    "?localidades=N1[1]|N3[all]&classificacao=1019[49260,49264,49266,49267,49268,49269,49270]"
)

BCB_NATIONAL_SERIES = {
    "endividamento_familias": {"id": 29037, "label": "Endividamento das famílias (%)"},
    "comprometimento_renda": {"id": 29034, "label": "Comprometimento de renda (%)"},
    "inad_direcionados": {"id": 21145, "label": "Direcionado PF"},
    "inad_consignado": {"id": 21119, "label": "Consignado Total"},
    "inad_veiculos": {"id": 21121, "label": "Veículos PF"},
    "inad_cartao_tot": {"id": 21129, "label": "Cartão Total PF"},
    "inad_nao_consig": {"id": 21114, "label": "Pessoal Não-Consignado"},
    "inad_cheque_esp": {"id": 21113, "label": "Cheque Especial"},
    "inad_cartao_rot": {"id": 21127, "label": "Cartão Rotativo"},
    "juros_consignado": {"id": 20746, "label": "Consignado Total"},
    "juros_veiculos": {"id": 20749, "label": "Veículos PF"},
    "juros_cartao_tot": {"id": 22024, "label": "Cartão Total PF"},
    "juros_nao_consig": {"id": 20742, "label": "Pessoal Não-Consignado"},
    "juros_cheque_esp": {"id": 20741, "label": "Cheque Especial"},
    "juros_cartao_rot": {"id": 22022, "label": "Cartão Rotativo"},
    "saldo_consignado": {"id": 20579, "label": "Consignado Total"},
    "saldo_veiculos": {"id": 20581, "label": "Veículos PF"},
    "saldo_cartao_tot": {"id": 20590, "label": "Cartão Total PF"},
    "saldo_nao_consig": {"id": 20574, "label": "Pessoal Não-Consignado"},
    "saldo_cheque_esp": {"id": 20573, "label": "Cheque Especial"},
    "saldo_total_livres": {"id": 20570, "label": "Total Livres PF"},
}


def _norm(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).strip().lower()


def _fetch_json(url: str, timeout: int = 30) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": "Central-IA-Portfolio-PublicResearch/3.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _fetch_bcb_sgs(sid: int, start: str = "01/01/2022", end: str = "31/12/2026") -> list[dict]:
    url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{sid}/dados?formato=json&dataInicial={start}&dataFinal={end}"
    return list(_fetch_json(url, timeout=25))


def _fetch_olinda_window(filter_expr: str) -> tuple[str, list[dict]]:
    filt = quote(filter_expr)
    url = (
        "https://olinda.bcb.gov.br/olinda/servico/taxaJuros/versao/v2/odata/"
        f"TaxasJurosDiariaPorInicioPeriodo?%24filter={filt}&%24top=5000&%24format=json"
    )
    rows = _fetch_json(url, timeout=30)["value"]
    latest_ini = max(x["InicioPeriodo"] for x in rows)
    latest_fim = max(x["FimPeriodo"] for x in rows if x["InicioPeriodo"] == latest_ini)
    cleaned = [
        {
            "modalidade": r["Modalidade"],
            "instituicao": r["InstituicaoFinanceira"],
            "taxa_mes": float(r["TaxaJurosAoMes"]),
            "taxa_ano": float(r["TaxaJurosAoAno"]),
            "posicao": int(r["Posicao"]),
        }
        for r in rows
        if r["InicioPeriodo"] == latest_ini and r.get("TaxaJurosAoAno") is not None
    ]
    return f"{latest_ini} a {latest_fim}", cleaned


def collect_snapshot() -> dict:
    # 1. IBGE 2022–2025 Brasil + 27 UFs
    ibge_raw = _fetch_json(IBGE_UF_7438_URL, timeout=35)
    cut_map = {"P10": "P10", "P50": "P50", "P70": "P70", "P80": "P80", "P90": "P90", "P95": "P95", "P99": "P99"}
    ibge_brasil_panel: dict[str, dict[str, float]] = {}
    ibge_ufs_panel: dict[str, dict[str, dict[str, float]]] = {}

    for res_item in ibge_raw[0]["resultados"]:
        cat_label = list(res_item["classificacoes"][0]["categoria"].values())[0]
        short = cut_map.get(cat_label.split()[0], cat_label)
        for ser in res_item["series"]:
            loc_name = ser["localidade"]["nome"]
            for yr in IBGE_YEARS:
                val_str = ser["serie"].get(yr)
                if not val_str or val_str in ("-", "...", "X"):
                    continue
                val = float(val_str)
                if ser["localidade"]["nivel"]["id"] == "N1":
                    ibge_brasil_panel.setdefault(yr, {})[short] = val
                else:
                    loc_entry = ibge_ufs_panel.setdefault(_norm(loc_name), {"uf_name": loc_name, "years": {}})
                    loc_entry["years"].setdefault(yr, {})[short] = val

    # 2. BCB SGS 27 UFs (medias anuais 2022..2025 + serie 09/2026)
    def _load_uf(item: tuple[int, str, str]) -> dict:
        sid, sig, nm = item
        recs = _fetch_bcb_sgs(sid, "01/01/2022", "31/12/2026")
        by_year: dict[str, list[float]] = {}
        for r in recs:
            yr = r["data"][-4:]
            by_year.setdefault(yr, []).append(float(r["valor"]))
        vals = [float(r["valor"]) for r in recs]
        return {
            "sgs_id": sid,
            "uf": sig,
            "uf_name": nm,
            "latest_date": recs[-1]["data"],
            "inad_latest": vals[-1],
            "inad_12m_avg": round(float(np.mean(vals[-12:])), 3),
            "inad_2022_01": vals[0],
            "delta_2022_2026_pp": round(vals[-1] - vals[0], 2),
            "annual_avg": {yr: round(float(np.mean(vs)), 3) for yr, vs in sorted(by_year.items()) if yr in IBGE_YEARS},
        }

    with ThreadPoolExecutor(max_workers=10) as ex:
        bcb_ufs = list(ex.map(_load_uf, BCB_UF_INAD_SERIES))

    # 3. BCB Olinda: janela atual (10/2026) vs janela 11 meses atras (11/2025)
    ol_curr_win, ol_curr = _fetch_olinda_window("Segmento eq 'PESSOA FÍSICA' and FimPeriodo ge '2026-10-15'")
    ol_prev_win, ol_prev = _fetch_olinda_window(
        "Segmento eq 'PESSOA FÍSICA' and InicioPeriodo ge '2025-11-03' and FimPeriodo le '2025-11-18'"
    )

    # 4. BCB SGS Series Nacionais (2022–09/2026)
    def _load_nat(kv: tuple[str, dict]) -> tuple[str, dict]:
        key, meta = kv
        recs = _fetch_bcb_sgs(meta["id"], "01/01/2022", "31/12/2026")
        return key, {
            "id": meta["id"],
            "label": meta["label"],
            "dates": [r["data"] for r in recs],
            "values": [float(r["valor"]) for r in recs],
        }

    with ThreadPoolExecutor(max_workers=10) as ex:
        bcb_nat = dict(ex.map(_load_nat, BCB_NATIONAL_SERIES.items()))

    snap = {
        "sources": {
            "ibge": IBGE_UF_7438_URL,
            "bcb_uf_sgs_range": "15861..15887 (api.bcb.gov.br)",
            "bcb_olinda_current": f"TaxasJurosDiariaPorInicioPeriodo ({ol_curr_win})",
            "bcb_olinda_previous": f"TaxasJurosDiariaPorInicioPeriodo ({ol_prev_win})",
            "bcb_national_sgs": "2022–09/2026 (api.bcb.gov.br)",
        },
        "ibge_brasil": ibge_brasil_panel.get("2025", {}),
        "ibge_brasil_panel": ibge_brasil_panel,
        "ibge_ufs_panel": ibge_ufs_panel,
        "bcb_ufs": bcb_ufs,
        "olinda_window": ol_curr_win,
        "olinda_pf": ol_curr,
        "olinda_prev_window": ol_prev_win,
        "olinda_pf_prev": ol_prev,
        "bcb_national": bcb_nat,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
    return snap


def load_snapshot() -> dict:
    if not SNAPSHOT_PATH.is_file():
        return collect_snapshot()
    snap = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    if "olinda_pf_prev" not in snap or "ibge_ufs_panel" not in snap:
        return collect_snapshot()
    return snap


def _is_secured_modality(mod: str) -> bool:
    low = mod.lower()
    return any(k in low for k in ("consignado", "veículos", "veiculos", "imobiliário", "imobiliario"))


def _analyze_bank_competitive_structure(ol_curr: list[dict], ol_prev: list[dict]) -> dict:
    """Cluster institutions across PF modalities and measure 11-month pricing persistence."""
    by_mod: dict[str, list[dict]] = {}
    for item in ol_curr:
        by_mod.setdefault(item["modalidade"], []).append(item)

    bank_dispersion = []
    mod_pct_map: dict[tuple[str, str], float] = {}
    for mod, items in sorted(by_mod.items()):
        if len(items) < 10:
            continue
        rates = np.array([x["taxa_ano"] for x in items])
        ranks = stats.rankdata(rates, method="average")
        pct_ranks = (ranks - 1.0) / max(len(ranks) - 1, 1)
        for item, pr in zip(items, pct_ranks):
            mod_pct_map[(item["instituicao"], mod)] = float(pr)

        q10, q25, q50, q75, q90 = np.percentile(rates, [10, 25, 50, 75, 90])
        cheapest = min(items, key=lambda x: x["taxa_ano"])
        priciest = max(items, key=lambda x: x["taxa_ano"])
        bank_dispersion.append(
            {
                "modalidade": mod.replace(" - Prefixado", ""),
                "n_bancos": len(items),
                "p10_aa": round(float(q10), 2),
                "mediana_aa": round(float(q50), 2),
                "p90_aa": round(float(q90), 2),
                "iqr_pp": round(float(q75 - q25), 2),
                "ratio_p90_p10": round(float(q90 / q10), 2) if q10 > 0 else 0.0,
                "min_inst": cheapest["instituicao"],
                "min_aa": round(cheapest["taxa_ano"], 2),
                "max_inst": priciest["instituicao"],
                "max_aa": round(priciest["taxa_ano"], 2),
            }
        )
    bank_dispersion.sort(key=lambda x: x["mediana_aa"])

    # Aggregate bank profile for institutions present in >= 3 core modalities
    by_bank: dict[str, list[tuple[str, float]]] = {}
    for (inst, mod), pr in mod_pct_map.items():
        by_bank.setdefault(inst, []).append((mod, pr))

    bank_profiles = []
    for inst, entries in sorted(by_bank.items()):
        if len(entries) < 3:
            continue
        pcts = np.array([e[1] for e in entries])
        sec_share = sum(1 for e in entries if _is_secured_modality(e[0])) / len(entries)
        bank_profiles.append(
            {
                "instituicao": inst,
                "n_modalidades": len(entries),
                "mean_pct_rank": float(pcts.mean()),
                "spread_std_pct": float(pcts.std(ddof=0)),
                "secured_share": float(sec_share),
            }
        )

    feat_mat = np.array(
        [[b["mean_pct_rank"], b["n_modalidades"], b["secured_share"], b["spread_std_pct"]] for b in bank_profiles]
    )
    Z = StandardScaler().fit_transform(feat_mat)
    sil_by_k = {}
    for k in (2, 3, 4):
        km_tmp = KMeans(n_clusters=k, n_init=25, random_state=42).fit(Z)
        sil_by_k[k] = round(float(silhouette_score(Z, km_tmp.labels_)), 4)

    km = KMeans(n_clusters=3, n_init=30, random_state=42).fit(Z)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(Z)

    # Sort clusters by mean_pct_rank ascending: 0 = Competitivos, 1 = Intermediarios/Universal, 2 = Alto Custo
    raw_means = {c: float(np.mean([bank_profiles[i]["mean_pct_rank"] for i in range(len(bank_profiles)) if km.labels_[i] == c])) for c in range(3)}
    cid_map = {old_c: new_c for new_c, (old_c, _) in enumerate(sorted(raw_means.items(), key=lambda kv: kv[1]))}
    archetype_labels = {
        0: "Arquétipo 1: Competidores de Baixo Custo",
        1: "Arquétipo 2: Bancos Universais / Portfólio Amplo",
        2: "Arquétipo 3: Segmento Caro / Alta Extração",
    }

    archetypes = []
    for cid in range(3):
        members = [i for i, b in enumerate(bank_profiles) if cid_map[int(km.labels_[i])] == cid]
        for idx in members:
            bank_profiles[idx]["cluster_id"] = cid
            bank_profiles[idx]["cluster_name"] = archetype_labels[cid]
            bank_profiles[idx]["pca_1"] = round(float(coords[idx, 0]), 3)
            bank_profiles[idx]["pca_2"] = round(float(coords[idx, 1]), 3)
        archetypes.append(
            {
                "cluster_id": cid,
                "name": archetype_labels[cid],
                "n_bancos": len(members),
                "avg_pct_rank": round(float(np.mean([bank_profiles[i]["mean_pct_rank"] for i in members])), 3),
                "avg_modalidades": round(float(np.mean([bank_profiles[i]["n_modalidades"] for i in members])), 1),
                "avg_secured_share": round(float(np.mean([bank_profiles[i]["secured_share"] for i in members])), 3),
                "sample_bancos": [bank_profiles[i]["instituicao"] for i in members[:3]],
            }
        )

    # 11-month price rank persistence (11/2025 vs 10/2026 across identical institution+modality quotes)
    prev_by_mod: dict[str, list[dict]] = {}
    for item in ol_prev:
        prev_by_mod.setdefault(item["modalidade"], []).append(item)
    prev_pct_map: dict[tuple[str, str], float] = {}
    prev_rate_map: dict[tuple[str, str], float] = {}
    for mod, items in prev_by_mod.items():
        if len(items) < 10:
            continue
        rates = np.array([x["taxa_ano"] for x in items])
        ranks = stats.rankdata(rates, method="average")
        pct_ranks = (ranks - 1.0) / max(len(ranks) - 1, 1)
        for item, pr in zip(items, pct_ranks):
            key = (item["instituicao"], mod)
            prev_pct_map[key] = float(pr)
            prev_rate_map[key] = float(item["taxa_ano"])

    curr_rate_map = {(item["instituicao"], item["modalidade"]): float(item["taxa_ano"]) for item in ol_curr}
    common_keys = sorted(set(mod_pct_map) & set(prev_pct_map))
    p_prev = np.array([prev_pct_map[k] for k in common_keys])
    p_curr = np.array([mod_pct_map[k] for k in common_keys])
    sp_pers = stats.spearmanr(p_prev, p_curr)
    deltas = np.array([curr_rate_map[k] - prev_rate_map[k] for k in common_keys])
    bottom_prev = p_prev <= 0.25
    top_prev = p_prev >= 0.75

    return {
        "unique_institutions": len(set(x["instituicao"] for x in ol_curr)),
        "total_quotes": len(ol_curr),
        "modalities": bank_dispersion,
        "clustering": {
            "clustered_banks_min3_mods": len(bank_profiles),
            "silhouette_by_k": sil_by_k,
            "silhouette_k3": sil_by_k[3],
            "pca_explained_variance": [round(float(v), 3) for v in pca.explained_variance_ratio_],
            "archetypes": archetypes,
            "bank_points": bank_profiles,
        },
        "persistence_11m": {
            "matched_bank_modality_pairs": len(common_keys),
            "spearman_rho": round(float(sp_pers.statistic), 4),
            "spearman_p": float(sp_pers.pvalue),
            "cheapest_quartile_retention_pct": round(float(np.mean(p_curr[bottom_prev] <= 0.25) * 100), 1),
            "priciest_quartile_retention_pct": round(float(np.mean(p_curr[top_prev] >= 0.75) * 100), 1),
            "median_delta_pp": round(float(np.median(deltas)), 2),
            "scatter_points": [
                {"prev": round(float(a), 3), "curr": round(float(b), 3)}
                for a, b in zip(p_prev, p_curr)
            ],
        },
    }


def _analyze_uf_panel(snap: dict) -> dict:
    """Analyze 4-year UF panel (2022–2025, N=108) + 09/2026 regional convergence."""
    uf_panel = []
    uf_latest = []
    for row in snap["bcb_ufs"]:
        ib_node = snap["ibge_ufs_panel"][_norm(row["uf_name"])]
        years_dict = ib_node["years"]
        for yr in IBGE_YEARS:
            if yr not in years_dict or yr not in row["annual_avg"]:
                continue
            ib = years_dict[yr]
            uf_panel.append(
                {
                    "uf": row["uf"],
                    "uf_name": row["uf_name"],
                    "year": int(yr),
                    "p50": ib["P50"],
                    "p90": ib["P90"],
                    "p99": ib["P99"],
                    "p90_p50_ratio": round(ib["P90"] / ib["P50"], 3),
                    "inad_yr": row["annual_avg"][yr],
                }
            )
        ib25 = years_dict["2025"]
        uf_latest.append(
            {
                "uf": row["uf"],
                "uf_name": row["uf_name"],
                "p50": ib25["P50"],
                "p90": ib25["P90"],
                "p95": ib25["P95"],
                "p99": ib25["P99"],
                "p90_p50_ratio": round(ib25["P90"] / ib25["P50"], 3),
                "inad_latest": row["inad_latest"],
                "inad_12m_avg": row["inad_12m_avg"],
                "inad_2022_01": row["inad_2022_01"],
                "delta_2022_2026_pp": row["delta_2022_2026_pp"],
            }
        )

    # Panel Year-Fixed-Effects OLS (N=108 UF-years) + Out-of-Sample Temporal Split (2022-2024 -> 2025)
    log_p90 = np.array([np.log(r["p90"]) for r in uf_panel])
    ratio = np.array([r["p90_p50_ratio"] for r in uf_panel])
    y_pan = np.array([r["inad_yr"] for r in uf_panel])
    yrs = np.array([r["year"] for r in uf_panel])

    yr_dummies = np.column_stack([(yrs == yr).astype(float) for yr in (2023, 2024, 2025)])
    X_pooled = np.column_stack([np.ones(len(uf_panel)), log_p90])
    beta_pool, _, _, _ = np.linalg.lstsq(X_pooled, y_pan, rcond=None)
    r2_pooled = float(r2_score(y_pan, X_pooled @ beta_pool))

    X_fe = np.column_stack([np.ones(len(uf_panel)), log_p90, ratio, yr_dummies])
    beta_fe, _, _, _ = np.linalg.lstsq(X_fe, y_pan, rcond=None)
    y_fe_hat = X_fe @ beta_fe
    r2_fe = float(r2_score(y_pan, y_fe_hat))

    # Out-of-sample prediction of 2025 from 2022-2024 model vs Naive Persistence (UF_2024)
    tr = yrs <= 2024
    te = yrs == 2025
    ridge = Ridge(alpha=1.0).fit(np.column_stack([log_p90[tr], ratio[tr]]), y_pan[tr])
    pred_2025 = ridge.predict(np.column_stack([log_p90[te], ratio[te]]))
    mae_ridge_2025 = float(mean_absolute_error(y_pan[te], pred_2025))
    r2_oos_2025 = float(r2_score(y_pan[te], pred_2025))

    inad_2025 = {r["uf"]: r["inad_yr"] for r in uf_panel if r["year"] == 2025}
    inad_2024 = {r["uf"]: r["inad_yr"] for r in uf_panel if r["year"] == 2024}
    mae_persist_2025 = float(mean_absolute_error(
        [inad_2025[u] for u in sorted(inad_2025)],
        [inad_2024[u] for u in sorted(inad_2025)],
    ))

    # Cross-section 2025 residuals (comparing observed 09/2026 vs Year-adjusted log-P90 model)
    p90_latest = np.array([r["p90"] for r in uf_latest])
    inad_latest = np.array([r["inad_latest"] for r in uf_latest])
    sp_p90 = stats.spearmanr(p90_latest, inad_latest)
    ols_26 = stats.linregress(np.log(p90_latest), inad_latest)
    for idx, r in enumerate(uf_latest):
        pred = ols_26.intercept + ols_26.slope * np.log(r["p90"])
        r["pred_inad"] = round(float(pred), 2)
        r["residual_pp"] = round(float(r["inad_latest"] - pred), 2)

    # Beta-convergence across 27 UFs (2022 baseline -> delta 2022..2026)
    init_22 = np.array([r["inad_2022_01"] for r in uf_latest])
    delta_26 = np.array([r["delta_2022_2026_pp"] for r in uf_latest])
    conv = stats.linregress(init_22, delta_26)

    return {
        "n_ufs": len(uf_latest),
        "n_panel_obs": len(uf_panel),
        "panel_years": list(IBGE_YEARS),
        "panel_pooled_r2": round(r2_pooled, 4),
        "panel_fe_r2": round(r2_fe, 4),
        "panel_beta_log_p90": round(float(beta_fe[1]), 4),
        "panel_beta_ratio_p90_p50": round(float(beta_fe[2]), 4),
        "oos_2025_ridge_r2": round(r2_oos_2025, 4),
        "oos_2025_ridge_mae_pp": round(mae_ridge_2025, 3),
        "oos_2025_persist_mae_pp": round(mae_persist_2025, 3),
        "spearman_2026_rho": round(float(sp_p90.statistic), 4),
        "spearman_2026_p": float(sp_p90.pvalue),
        "cross_2026_r2": round(float(ols_26.rvalue ** 2), 4),
        "convergence_2022_2026_slope": round(float(conv.slope), 4),
        "convergence_2022_2026_p": float(conv.pvalue),
        "lowest_residual_ufs": sorted(uf_latest, key=lambda r: r["residual_pp"])[:4],
        "highest_residual_ufs": sorted(uf_latest, key=lambda r: -r["residual_pp"])[:4],
        "panel_rows": uf_panel,
        "rows_2026": sorted(uf_latest, key=lambda r: -r["p90"]),
    }


def run_analysis(snap: dict) -> dict:
    bank_struct = _analyze_bank_competitive_structure(snap["olinda_pf"], snap["olinda_pf_prev"])
    uf_panel = _analyze_uf_panel(snap)
    nat = snap["bcb_national"]
    nat_modalities = []
    for suf, key_label in [
        ("consignado", "Consignado Total"),
        ("veiculos", "Veículos PF"),
        ("cartao_tot", "Cartão Total PF"),
        ("nao_consig", "Não-Consignado"),
        ("cheque_esp", "Cheque Especial"),
    ]:
        nat_modalities.append(
            {
                "label": key_label,
                "saldo_bi": round(nat[f"saldo_{suf}"]["values"][-1] / 1000.0, 1),
                "inad_pct": nat[f"inad_{suf}"]["values"][-1],
                "juros_aa_pct": nat[f"juros_{suf}"]["values"][-1],
            }
        )

    return {
        "window": {
            "ibge_panel_years": "2022–2025",
            "bcb_uf_latest": snap["bcb_ufs"][0]["latest_date"],
            "olinda_current": snap["olinda_window"],
            "olinda_previous": snap["olinda_prev_window"],
        },
        "bank_dispersion": bank_struct,
        "uf_analysis": uf_panel,
        "national_context": {
            "ibge_brasil": snap["ibge_brasil"],
            "endividamento_08_2026": nat["endividamento_familias"]["values"][-1],
            "comprometimento_08_2026": nat["comprometimento_renda"]["values"][-1],
            "saldo_total_livres_bi": round(nat["saldo_total_livres"]["values"][-1] / 1000.0, 1),
            "modalities": nat_modalities,
        },
    }


def _save_svg(fig, fig_dir: Path, filename: str) -> None:
    targets = [fig_dir / filename]
    if SITE_ASSETS_DIR.is_dir():
        targets.append(SITE_ASSETS_DIR / filename)
    for target in targets:
        fig.savefig(target, format="svg", facecolor="white")


def generate_figures(snap: dict, res: dict) -> None:
    fig_dir = RESULTS_DIR / "figuras"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})

    bd = res["bank_dispersion"]
    cl = bd["clustering"]
    pers = bd["persistence_11m"]
    uf = res["uf_analysis"]

    # Fig 1: PCA + K-Means (k=3) dos Arquétipos Competitivos Bancários (Olinda BCB)
    pts = cl["bank_points"]
    fig, ax = plt.subplots(figsize=(10.2, 5.8), layout="constrained")
    ax.set_facecolor("#F8FAFC")
    ax.grid(color="#E2E8F0", linewidth=0.8)
    colors = {0: "#10B981", 1: "#2878E8", 2: "#D91568"}
    for arch in cl["archetypes"]:
        cid = arch["cluster_id"]
        sub = [b for b in pts if b["cluster_id"] == cid]
        ax.scatter(
            [b["pca_1"] for b in sub],
            [b["pca_2"] for b in sub],
            s=[55 + b["n_modalidades"] * 8 for b in sub],
            color=colors[cid],
            alpha=0.86,
            edgecolors="white",
            linewidth=1.1,
            label=f"{arch['name']} ({arch['n_bancos']} bancos | rank médio {arch['avg_pct_rank']*100:.0f}º pct)",
            zorder=4,
        )
    # Annotate representative extreme institutions with non-overlapping directional offsets
    top_cheap = sorted(pts, key=lambda b: b["mean_pct_rank"])[:3]
    top_exp = sorted(pts, key=lambda b: -b["mean_pct_rank"])[:3]
    cheap_offsets = [(-18, -22), (16, -24), (-16, 22)]
    exp_offsets = [(-12, 18), (18, 22), (-14, -26)]
    for b, offset in zip(top_cheap, cheap_offsets):
        short = b["instituicao"].split("S.A")[0].strip()[:20]
        ax.annotate(
            short,
            (b["pca_1"], b["pca_2"]),
            xytext=offset,
            textcoords="offset points",
            ha="center",
            fontsize=7.6,
            fontweight="bold",
            color="#065F46",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#A7F3D0", alpha=0.92),
            arrowprops=dict(arrowstyle="-", color="#059669", lw=0.9),
            zorder=6,
        )
    for b, offset in zip(top_exp, exp_offsets):
        short = b["instituicao"].split("S.A")[0].strip()[:20]
        ax.annotate(
            short,
            (b["pca_1"], b["pca_2"]),
            xytext=offset,
            textcoords="offset points",
            ha="center",
            fontsize=7.6,
            fontweight="bold",
            color="#9D174D",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#FBCFE8", alpha=0.92),
            arrowprops=dict(arrowstyle="-", color="#D91568", lw=0.9),
            zorder=6,
        )

    ev = cl["pca_explained_variance"]
    ax.set_title(
        f"Arquétipos Competitivos Bancários (K-Means k=3 em {cl['clustered_banks_min3_mods']} bancos · Silhueta {cl['silhouette_k3']:.2f})",
        fontweight="bold",
        color="#0B2038",
        pad=12,
    )
    ax.set_xlabel(f"PCA 1 ({ev[0]*100:.0f}% var.: Mix Garantia + Amplitude de Portfólio)")
    ax.set_ylabel(f"PCA 2 ({ev[1]*100:.0f}% var.: Posicionamento de Preço & Dispersão)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        frameon=True,
        facecolor="white",
        edgecolor="#E2E8F0",
        fontsize=8.4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=1,
    )
    _save_svg(fig, fig_dir, "alta_renda_h1_h4.svg")
    plt.close(fig)

    # Fig 2: Dispersão P10-Mediana-P90 entre Instituições (Olinda BCB)
    mods = [m for m in bd["modalities"] if "Desconto" not in m["modalidade"] and "outros bens" not in m["modalidade"].lower()]
    fig, ax = plt.subplots(figsize=(10.4, 5.4), layout="constrained")
    ax.set_facecolor("#F8FAFC")
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.8)
    y = np.arange(len(mods))
    for i, m in enumerate(mods):
        ax.plot([m["p10_aa"], m["p90_aa"]], [i, i], color="#CBD5E1", linewidth=6, solid_capstyle="round", zorder=2)
        ax.scatter(m["p10_aa"], i, color="#10B981", s=70, zorder=4, label="P10 (bancos mais baratos)" if i == 0 else None)
        ax.scatter(m["mediana_aa"], i, color="#0B2038", s=75, marker="D", zorder=5, label="Mediana (P50)" if i == 0 else None)
        ax.scatter(m["p90_aa"], i, color="#D91568", s=70, zorder=4, label="P90 (bancos mais caros)" if i == 0 else None)
        ax.text(m["p90_aa"] * 1.13, i, f"{m['mediana_aa']:.0f}% ({m['ratio_p90_p10']:.1f}x P90/P10)", va="center", fontsize=8.5, fontweight="bold", color="#334155")
    ax.set_xscale("log")
    ax.set_xlim(12, 2600)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{m['modalidade']} ({m['n_bancos']} inst.)" for m in mods], fontsize=9)
    ax.set_title(f"Dispersão Intra-Produto entre {bd['unique_institutions']} Bancos (BCB Olinda {snap['olinda_window']})", fontweight="bold", fontsize=11, color="#0B2038", pad=26)
    ax.set_xlabel("Taxa de Juros Anual (% a.a., escala logarítmica)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3, fontsize=8.5)
    _save_svg(fig, fig_dir, "alta_renda_retornos_30d.svg")
    plt.close(fig)

    # Fig 3: Painel 2022–2025 (N=108 UF-Anos) com Efeitos Fixos Temporais
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 5.1), layout="constrained")
    yr_cols = {2022: "#94A3B8", 2023: "#2878E8", 2024: "#6A1B9A", 2025: "#D91568"}
    ax1.set_facecolor("#F8FAFC")
    ax1.grid(color="#E2E8F0", linewidth=0.8)
    for yr in (2022, 2023, 2024, 2025):
        sub = [r for r in uf["panel_rows"] if r["year"] == yr]
        ax1.scatter([r["p90"] for r in sub], [r["inad_yr"] for r in sub], color=yr_cols[yr], s=44, alpha=0.84, label=str(yr))
    ax1.set_title(f"Painel Estadual (N={uf['n_panel_obs']} UF-anos · R² FE {uf['panel_fe_r2']:.2f})", fontsize=10.5, fontweight="bold", color="#0B2038", pad=12)
    ax1.set_xlabel("Renda P90 Estadual (R$/mês · IBGE 2022–2025)")
    ax1.set_ylabel("Inadimplência PF Média Anual (%)")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0", fontsize=8.2, title="Ano IBGE", loc="upper right")

    # Right: 2026 residuals highlighting structural outliers beyond income
    sorted_res = sorted(uf["rows_2026"], key=lambda r: r["residual_pp"])
    pick = sorted_res[:5] + sorted_res[-5:]
    ax2.set_facecolor("#F8FAFC")
    ax2.grid(axis="x", color="#E2E8F0", linewidth=0.8)
    vals = [r["residual_pp"] for r in pick]
    y_p = np.arange(len(pick))
    ax2.barh(y_p, vals, color=["#10B981" if v < 0 else "#D91568" for v in vals], height=0.62)
    ax2.axvline(0, color="#0B2038", linewidth=1.2)
    ax2.set_yticks(y_p)
    ax2.set_yticklabels([f"{r['uf']} (P90 R$ {r['p90']:,.0f})".replace(",", ".") for r in pick], fontsize=8.4)
    for i, v in enumerate(vals):
        ax2.text(v + (0.10 if v >= 0 else -0.10), i, f"{v:+.2f} pp", va="center", ha="left" if v >= 0 else "right", fontsize=8.2, fontweight="bold", color="#1E293B")
    ax2.set_xlim(-3.4, 5.2)
    ax2.set_title("Resíduos Estruturais 09/2026\n(< 0 = risco menor que o esperado pela renda)", fontsize=10.0, fontweight="bold", color="#0B2038", pad=10)
    ax2.set_xlabel("Resíduo em p.p.")
    ax2.spines[["top", "right"]].set_visible(False)
    _save_svg(fig, fig_dir, "alta_renda_funil.svg")
    plt.close(fig)

    # Fig 4: Persistência Longitudinal de Preço Bancário (11 meses: 11/2025 vs 10/2026)
    fig, ax = plt.subplots(figsize=(9.4, 4.8), layout="constrained")
    ax.set_facecolor("#F8FAFC")
    ax.grid(color="#E2E8F0", linewidth=0.8)
    spts = pers["scatter_points"]
    px = np.array([s["prev"] * 100 for s in spts])
    py = np.array([s["curr"] * 100 for s in spts])
    ax.scatter(px, py, color="#2878E8", alpha=0.48, s=36, edgecolors="white", linewidth=0.5, label=f"{pers['matched_bank_modality_pairs']} pares Banco × Modalidade")
    ax.plot([0, 100], [0, 100], color="#D91568", linestyle="--", linewidth=1.8, label="Persistência perfeita (diagonal)")
    ax.axvspan(0, 25, color="#10B981", alpha=0.07)
    ax.axhspan(0, 25, color="#10B981", alpha=0.07)
    ax.text(
        4, 88,
        f"Spearman ρ = {pers['spearman_rho']:.3f} (p = {pers['spearman_p']:.1e})\n"
        f"Retenção quartil mais barato (≤P25): {pers['cheapest_quartile_retention_pct']:.1f}%\n"
        f"Retenção quartil mais caro (≥P75): {pers['priciest_quartile_retention_pct']:.1f}%",
        fontsize=9, fontweight="bold", color="#0B2038",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#CBD5E1"),
    )
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.set_xlabel(f"Percentil de Taxa do Banco na Modalidade em 11/2025 ({snap['olinda_prev_window'][:10]})")
    ax.set_ylabel(f"Percentil em 10/2026 ({snap['olinda_window'][:10]})")
    ax.set_title("Persistência Competitiva de Preços Bancários em 11 Meses (Olinda BCB)", fontweight="bold", color="#0B2038")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=True, facecolor="white", loc="lower right", fontsize=8.5)
    _save_svg(fig, fig_dir, "alta_renda_ranking.svg")
    plt.close(fig)

    # Fig 5: Vitrine Homepage SVG Real-Data Card (replaces static illustrative card cleanly!)
    if SITE_ASSETS_DIR.is_dir():
        top_mods = [m for m in bd["modalities"] if m["ratio_p90_p10"] >= 1.5][:5]
        fig, ax = plt.subplots(figsize=(6.4, 3.85), layout="constrained")
        ax.set_facecolor("#F8FAFC")
        ax.grid(axis="x", color="#E2E8F0", linewidth=0.8)
        yy = np.arange(len(top_mods))
        for i, m in enumerate(top_mods):
            ax.plot([m["p10_aa"], m["p90_aa"]], [i, i], color="#CBD5E1", linewidth=6.5, solid_capstyle="round", zorder=2)
            ax.scatter(m["p10_aa"], i, color="#10B981", s=68, zorder=4, label="P10 (barato)" if i == 0 else None)
            ax.scatter(m["mediana_aa"], i, color="#0B2038", s=66, marker="D", zorder=5, label="Mediana" if i == 0 else None)
            ax.scatter(m["p90_aa"], i, color="#D91568", s=68, zorder=4, label="P90 (caro)" if i == 0 else None)
            ax.text(m["p90_aa"] * 1.12, i, f"{m['ratio_p90_p10']:.1f}x", va="center", fontsize=8.8, fontweight="bold", color="#D91568")
        ax.set_xscale("log")
        ax.set_xlim(14, 1500)
        ax.set_yticks(yy)
        ax.set_yticklabels([m["modalidade"][:24] for m in top_mods], fontsize=8.5, fontweight="bold", color="#1E293B")
        ax.set_title(
            f"Dispersão P10–P90 ({bd['unique_institutions']} Bancos) · Persistência ρ={pers['spearman_rho']:.2f}",
            fontsize=9.8,
            fontweight="bold",
            color="#0B2038",
        )
        ax.set_xlabel("Juros (% a.a., escala log) · BCB Olinda & IBGE 2022–2025", fontsize=8.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8.0, loc="lower right")
        # vitrine_estudo_03_alta_renda.svg is maintained as standard dark navy executive card
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analise quantitativa BCB Olinda + Painel IBGE 27 UFs (2022–2025).")
    parser.add_argument("--refresh", action="store_true", help="Baixa dados atualizados das APIs do IBGE e Banco Central.")
    args = parser.parse_args()

    snap = collect_snapshot() if args.refresh else load_snapshot()
    results = run_analysis(snap)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "resultados.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    generate_figures(snap, results)
    print(json.dumps({
        "window": results["window"],
        "bank_clusters": {
            "clustered_banks": results["bank_dispersion"]["clustering"]["clustered_banks_min3_mods"],
            "silhouette_by_k": results["bank_dispersion"]["clustering"]["silhouette_by_k"],
            "archetypes": results["bank_dispersion"]["clustering"]["archetypes"],
        },
        "persistence_11m": {k: v for k, v in results["bank_dispersion"]["persistence_11m"].items() if k != "scatter_points"},
        "uf_panel": {k: v for k, v in results["uf_analysis"].items() if k not in ("panel_rows", "rows_2026", "lowest_residual_ufs", "highest_residual_ufs")},
        "top_mods": [{k: m[k] for k in ("modalidade", "n_bancos", "p10_aa", "mediana_aa", "p90_aa", "ratio_p90_p10")} for m in results["bank_dispersion"]["modalities"]],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
