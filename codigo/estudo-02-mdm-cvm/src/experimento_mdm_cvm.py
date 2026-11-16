"""Experimento reproduzivel de MDM e Resolucao de Entidades em cadastros publicos da CVM."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass
import difflib
import io
import json
from pathlib import Path
import re
import unicodedata
import urllib.request

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_score, recall_score, f1_score

STUDY_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STUDY_ROOT.parents[1]
SITE_ASSETS_DIR = (
    REPO_ROOT
    / "estudos"
    / "202609 - Impacto da qualidade dos dados no risco de inadimplência"
    / "app"
    / "site_assets"
    / "assets"
)

CVM_SOURCES = {
    "CVM_CIA_ABERTA": {
        "label": "Companhias Abertas",
        "url": "https://dados.cvm.gov.br/dados/CIA_ABERTA/CAD/DADOS/cad_cia_aberta.csv",
        "trust_rank": 1,
        "cnpj_col": "CNPJ_CIA",
        "name_col": "DENOM_SOCIAL",
        "trade_col": "DENOM_COMERC",
        "uf_col": "UF",
        "sit_col": "SIT",
        "date_col": "DT_INI_SIT",
    },
    "CVM_FI_ADMIN": {
        "label": "Administradores FI",
        "url": "https://dados.cvm.gov.br/dados/FI/CAD/DADOS/cad_fi.csv",
        "trust_rank": 2,
        "cnpj_col": "CNPJ_ADMIN",
        "name_col": "ADMIN",
        "trade_col": "",
        "uf_col": "",
        "sit_col": "SIT",
        "date_col": "DT_INI_SIT",
    },
    "CVM_ADM_FII": {
        "label": "Administradores FII",
        "url": "https://dados.cvm.gov.br/dados/ADM_FII/CAD/DADOS/cad_adm_fii.csv",
        "trust_rank": 3,
        "cnpj_col": "CNPJ",
        "name_col": "DENOM_SOCIAL",
        "trade_col": "DENOM_COMERC",
        "uf_col": "UF",
        "sit_col": "SIT",
        "date_col": "DT_INI_SIT",
    },
    "CVM_FI_CUSTOD": {
        "label": "Custodiantes FI",
        "url": "https://dados.cvm.gov.br/dados/FI/CAD/DADOS/cad_fi.csv",
        "trust_rank": 4,
        "cnpj_col": "CNPJ_CUSTODIANTE",
        "name_col": "CUSTODIANTE",
        "trade_col": "",
        "uf_col": "",
        "sit_col": "SIT",
        "date_col": "DT_INI_SIT",
    },
    "CVM_FI_GESTOR": {
        "label": "Gestores PJ FI",
        "url": "https://dados.cvm.gov.br/dados/FI/CAD/DADOS/cad_fi.csv",
        "trust_rank": 5,
        "cnpj_col": "CPF_CNPJ_GESTOR",
        "name_col": "GESTOR",
        "trade_col": "",
        "uf_col": "",
        "sit_col": "SIT",
        "date_col": "DT_INI_SIT",
    },
    "CVM_CIA_INCENT": {
        "label": "Cias. Incentivadas",
        "url": "https://dados.cvm.gov.br/dados/CIA_INCENT/CAD/DADOS/cad_cia_incent.csv",
        "trust_rank": 6,
        "cnpj_col": "CNPJ",
        "name_col": "DENOM_SOCIAL",
        "trade_col": "DENOM_COMERC",
        "uf_col": "UF",
        "sit_col": "SIT",
        "date_col": "DT_INI_SIT",
    },
}


@dataclass(frozen=True)
class CvmRecord:
    source: str
    trust_rank: int
    cnpj: str
    denom_social: str
    denom_comerc: str
    uf: str
    sit: str
    dt_ini_sit: str

    def completeness_score(self) -> float:
        fields = [
            bool(self.cnpj),
            bool(self.denom_social),
            bool(self.denom_comerc),
            bool(self.uf),
            bool(self.sit),
            bool(self.dt_ini_sit),
        ]
        return 100.0 * sum(fields) / len(fields)


def _clean_cnpj(raw: str | None) -> str:
    digits = "".join(ch for ch in (raw or "") if ch.isdigit())
    if len(digits) < 11 or len(digits) > 14:
        return ""
    return digits.zfill(14)


def _fetch_csv(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Central-IA-Portfolio/1.0"})
    with urllib.request.urlopen(req, timeout=45) as resp:
        return resp.read().decode("latin1", errors="replace")


def load_cvm_records() -> list[CvmRecord]:
    cache: dict[str, str] = {}
    records: list[CvmRecord] = []
    seen_fi_role: set[tuple[str, str]] = set()

    for src, cfg in CVM_SOURCES.items():
        url = str(cfg["url"])
        if url not in cache:
            cache[url] = _fetch_csv(url)
        reader = csv.DictReader(io.StringIO(cache[url]), delimiter=";")
        for row in reader:
            if src == "CVM_FI_GESTOR" and (row.get("PF_PJ_GESTOR") or "").strip() != "PJ":
                continue
            cnpj = _clean_cnpj(row.get(str(cfg["cnpj_col"])))
            if not cnpj:
                continue
            if src.startswith("CVM_FI_"):
                key = (src, cnpj)
                if key in seen_fi_role:
                    continue
                seen_fi_role.add(key)
            records.append(
                CvmRecord(
                    source=src,
                    trust_rank=int(cfg["trust_rank"]),
                    cnpj=cnpj,
                    denom_social=(row.get(str(cfg["name_col"])) or "").strip(),
                    denom_comerc=(row.get(str(cfg["trade_col"])) or "").strip() if cfg["trade_col"] else "",
                    uf=(row.get(str(cfg["uf_col"])) or "").strip() if cfg["uf_col"] else "",
                    sit=(row.get(str(cfg["sit_col"])) or "").strip(),
                    dt_ini_sit=(row.get(str(cfg["date_col"])) or "").strip(),
                )
            )
    return records


def analyze_cvm_mdm(records: list[CvmRecord]) -> dict:
    by_cnpj: dict[str, list[CvmRecord]] = defaultdict(list)
    by_source: dict[str, list[CvmRecord]] = defaultdict(list)
    for rec in records:
        by_cnpj[rec.cnpj].append(rec)
        by_source[rec.source].append(rec)

    source_stats = []
    for src, cfg in CVM_SOURCES.items():
        grp = by_source.get(src, [])
        scores = [r.completeness_score() for r in grp]
        missing_uf = sum(1 for r in grp if not r.uf)
        missing_trade = sum(1 for r in grp if not r.denom_comerc)
        source_stats.append(
            {
                "source": src,
                "label": cfg["label"],
                "records": len(grp),
                "avg_completeness": round(float(np.mean(scores)) if scores else 0.0, 1),
                "missing_uf": missing_uf,
                "missing_trade_name": missing_trade,
            }
        )

    multi = {k: v for k, v in by_cnpj.items() if len(v) >= 2}
    conflict_sit = 0
    conflict_date = 0
    conflict_name = 0
    trade_name_diffs = 0
    coexistence_fill_events = 0
    centralized_writebacks = 0

    trust_winner_counts: Counter[str] = Counter()
    recency_winner_counts: Counter[str] = Counter()
    flipped_sit_decisions = 0
    example_clusters = []

    for cnpj, grp in sorted(by_cnpj.items()):
        # Golden completeness across cluster
        sits = {r.sit.upper() for r in grp if r.sit}
        dates = {r.dt_ini_sit for r in grp if r.dt_ini_sit}
        names = {r.denom_social.upper() for r in grp if r.denom_social}
        has_trade_diff = any(r.denom_comerc and r.denom_social and r.denom_comerc.upper() != r.denom_social.upper() for r in grp)
        if has_trade_diff:
            trade_name_diffs += 1

        # Pick winners inside cluster
        trust_win = min(grp, key=lambda r: r.trust_rank)
        recency_win = max(grp, key=lambda r: (r.dt_ini_sit or "0000-00-00", -r.trust_rank))

        if len(grp) >= 2:
            if len(sits) > 1:
                conflict_sit += 1
            if len(dates) > 1:
                conflict_date += 1
            if len(names) > 1:
                conflict_name += 1
            trust_winner_counts[trust_win.source] += 1
            recency_winner_counts[recency_win.source] += 1
            if trust_win.sit.upper() != recency_win.sit.upper():
                flipped_sit_decisions += 1

            # Coexistencia vs Centralizado sobre atributos enriqueciveis no cluster
            golden_trade = next((r.denom_comerc for r in sorted(grp, key=lambda x: x.trust_rank) if r.denom_comerc), "")
            golden_uf = next((r.uf for r in sorted(grp, key=lambda x: x.trust_rank) if r.uf), "")
            for r in grp:
                if (not r.denom_comerc and golden_trade) or (not r.uf and golden_uf):
                    coexistence_fill_events += 1
                if r.sit != trust_win.sit or r.denom_social != trust_win.denom_social or r.denom_comerc != golden_trade or r.uf != golden_uf:
                    centralized_writebacks += 1

            if len(example_clusters) < 5 and len(sits) > 1 and any(r.uf for r in grp):
                example_clusters.append(
                    {
                        "cnpj_masked": f"{cnpj[:2]}.{cnpj[2:5]}.{cnpj[5:8]}/{cnpj[8:12]}-{cnpj[12:]}",
                        "records": [
                            {
                                "source": r.source,
                                "denom_social": r.denom_social,
                                "denom_comerc": r.denom_comerc,
                                "uf": r.uf,
                                "sit": r.sit,
                                "dt_ini_sit": r.dt_ini_sit,
                            }
                            for r in grp
                        ],
                        "winner_trust": trust_win.source,
                        "winner_recency": recency_win.source,
                    }
                )

    golden_scores = []
    for grp in by_cnpj.values():
        merged_fields = [
            True,  # cnpj
            any(r.denom_social for r in grp),
            any(r.denom_comerc for r in grp),
            any(r.uf for r in grp),
            any(r.sit for r in grp),
            any(r.dt_ini_sit for r in grp),
        ]
        golden_scores.append(100.0 * sum(merged_fields) / len(merged_fields))

    raw_count = len(records)
    golden_count = len(by_cnpj)
    dedup_pct = round((golden_count - raw_count) / raw_count * 100.0, 1)
    avg_raw_score = round(float(np.mean([r.completeness_score() for r in records])), 1)
    avg_golden_score = round(float(np.mean(golden_scores)), 1)

    return {
        "source_name": "CVM Dados Abertos (dados.cvm.gov.br)",
        "sources_count": len(CVM_SOURCES),
        "raw_records": raw_count,
        "golden_records": golden_count,
        "dedup_reduction_percent": dedup_pct,
        "redundant_records": raw_count - golden_count,
        "multi_source_entities": len(multi),
        "conflict_sit": conflict_sit,
        "conflict_date": conflict_date,
        "conflict_name": conflict_name,
        "trade_name_diff_entities": trade_name_diffs,
        "flipped_sit_trust_vs_recency": flipped_sit_decisions,
        "avg_raw_completeness": avg_raw_score,
        "avg_golden_completeness": avg_golden_score,
        "sync_modes": {
            "consolidacao_events": 0,
            "coexistencia_events": coexistence_fill_events,
            "centralizado_events": centralized_writebacks,
            "corrective_overwrites": centralized_writebacks - coexistence_fill_events,
        },
        "source_stats": source_stats,
        "trust_winner_counts": dict(trust_winner_counts),
        "recency_winner_counts": dict(recency_winner_counts),
        "example_clusters": example_clusters,
    }


def generate_figures(summary: dict, out_dirs: list[Path]) -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"

    # Fig 1: Completude cadastral media por base CVM vs Golden Record Consolidado
    labels = [s["label"] for s in summary["source_stats"]] + ["Golden Record\n(Hub MDM CVM)"]
    values = [s["avg_completeness"] for s in summary["source_stats"]] + [summary["avg_golden_completeness"]]
    colors = ["#2878E8", "#64748B", "#2878E8", "#64748B", "#64748B", "#D91568", "#6A1B9A"]

    fig, ax = plt.subplots(figsize=(9.6, 4.6), layout="constrained")
    ax.set_facecolor("#FFFFFF")
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.8, linestyle="--")
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color("#CBD5E1")
        s.set_linewidth(1.0)
    y = np.arange(len(labels))
    bars = ax.barh(y, values, color=colors, height=0.58)
    ax.set_yticks(y, labels=labels, fontsize=9.5, color="#1E293B")
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xlabel("Completude cadastral média (% de 6 atributos canônicos)", fontsize=10, color="#334155")
    ax.set_title(
        f"Completude nos 6 fluxos públicos CVM vs Golden Record ({summary['raw_records']:,} -> {summary['golden_records']:,} CNPJs)",
        fontsize=11.5,
        fontweight="bold",
        color="#0B2038",
    )
    for bar, val in zip(bars, values):
        ax.text(val + 1.2, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%", va="center", fontsize=9, fontweight="bold", color="#0B2038")
    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "mdm_fig1_qualidade_fontes.svg", facecolor="white")
    plt.close(fig)

    # Fig 2: Sobrevivencia (Confianca vs Recencia nos clusters multi-papel CVM) e Pacotes de Retorno
    multi_n = max(summary["multi_source_entities"], 1)
    top_sources = ["CVM_CIA_ABERTA", "CVM_FI_ADMIN", "CVM_ADM_FII", "CVM_FI_CUSTOD", "CVM_FI_GESTOR"]
    short_labels = ["Cia. Aberta", "Admin. FI", "Admin. FII", "Custod. FI", "Gestor FI"]
    trust_pct = [summary["trust_winner_counts"].get(s, 0) / multi_n * 100 for s in top_sources]
    recency_pct = [summary["recency_winner_counts"].get(s, 0) / multi_n * 100 for s in top_sources]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 5.2), layout="constrained")
    x = np.arange(len(short_labels))
    w = 0.36
    ax1.set_facecolor("#FFFFFF")
    ax1.grid(axis="y", color="#F1F5F9", linewidth=0.8, linestyle="--")
    for s in ax1.spines.values():
        s.set_visible(True)
        s.set_color("#CBD5E1")
        s.set_linewidth(1.0)
    b1 = ax1.bar(x - w / 2, trust_pct, width=w, color="#6A1B9A", label="Confiança Regulatória")
    b2 = ax1.bar(x + w / 2, recency_pct, width=w, color="#D91568", label="Recência (DT_INI_SIT)")
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            if h >= 1.0:
                ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.5, f"{h:.0f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#1E293B")
    ax1.set_xticks(x, short_labels, rotation=16, fontsize=8.8)
    ax1.set_ylim(0, max(*trust_pct, *recency_pct) * 1.32)
    ax1.set_ylabel("% dos clusters multi-papel CVM")
    ax1.set_title(f"Fonte vencedora em {summary['multi_source_entities']} CNPJs multi-origem", fontsize=10.5, fontweight="bold", color="#0B2038", pad=36)
    ax1.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", framealpha=0.98, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, borderpad=0.4)

    sync = summary["sync_modes"]
    sync_labels = ["Consolidação\n(Somente Hub)", "Coexistência\n(Preenche vazios)", "Centralizado\n(Padronização total)"]
    sync_vals = [sync["consolidacao_events"], sync["coexistencia_events"], sync["centralizado_events"]]
    ax2.set_facecolor("#FFFFFF")
    ax2.grid(axis="y", color="#F1F5F9", linewidth=0.8, linestyle="--")
    for s in ax2.spines.values():
        s.set_visible(True)
        s.set_color("#CBD5E1")
        s.set_linewidth(1.0)
    b2 = ax2.bar(sync_labels, sync_vals, color=["#64748B", "#2878E8", "#D91568"], width=0.52)
    ax2.set_ylim(0, max(sync_vals) * 1.25)
    ax2.set_ylabel("Registros/atributos sincronizados")
    ax2.set_title("Impacto de retorno às origens nos CNPJs multi-papel", fontsize=10.5, fontweight="bold", color="#0B2038", pad=36)
    for bar, val in zip(b2, sync_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + max(sync_vals) * 0.02, f"{val:,}", ha="center", fontsize=9.5, fontweight="bold", color="#0B2038")

    for d in out_dirs:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / "mdm_fig2_survivorship_sync.svg", facecolor="white")
    plt.close(fig)

    rl = summary.get("record_linkage")
    if rl and rl.get("regimes"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 5.1), layout="constrained")
        reg_legal = rl["regimes"]["legal_name"]["models"]
        reg_comm = rl["regimes"]["commercial_drift"]["models"]

        methods = ["Regra Exata\nNormalizada", "TF-IDF Char\n(3-4 gramas)", "Matcher ML Híbrido\n(Features + UF)"]
        keys = ["exact_normalized", "tfidf_cosine", "hybrid_logistic"]
        x = np.arange(len(keys))
        w = 0.34

        f1_legal = [reg_legal[k]["f1"] * 100 for k in keys]
        f1_comm = [reg_comm[k]["f1"] * 100 for k in keys]

        ax1.set_facecolor("#FFFFFF")
        ax1.grid(axis="y", color="#F1F5F9", linewidth=0.8, linestyle="--")
        for s in ax1.spines.values():
            s.set_visible(True)
            s.set_color("#CBD5E1")
            s.set_linewidth(1.0)
        b1 = ax1.bar(x - w / 2, f1_legal, width=w, color="#2878E8", label="Razão Social formal (RFB/CVM)")
        b2 = ax1.bar(x + w / 2, f1_comm, width=w, color="#D91568", label="Nome Comercial CRM vs Social (Drift)")
        ax1.set_xticks(x, methods, fontsize=8.8)
        ax1.set_ylim(0, 122)
        ax1.set_ylabel("F1-Score fora da amostra (%)")
        ax1.set_title(
            "Pareamento nominal sem CNPJ: Formal vs Comercial",
            fontsize=10.5,
            fontweight="bold",
            color="#0B2038",
            pad=42,
        )
        ax1.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", framealpha=0.98, fontsize=8.2, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=1, borderpad=0.4)
        for bars in (b1, b2):
            for bar in bars:
                h = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + 1.5,
                    f"{h:.1f}%",
                    ha="center",
                    fontsize=8.2,
                    fontweight="bold",
                    color="#0B2038",
                )

        ax2.set_facecolor("#FFFFFF")
        ax2.grid(True, color="#F1F5F9", linewidth=0.8, linestyle="--")
        for s in ax2.spines.values():
            s.set_visible(True)
            s.set_color("#CBD5E1")
            s.set_linewidth(1.0)
        for key, label, color, ls in [
            ("hybrid_logistic", f"ML Híbrido (PR-AUC {reg_comm['hybrid_logistic']['pr_auc']:.3f})", "#D91568", "-"),
            ("tfidf_cosine", f"TF-IDF Char (PR-AUC {reg_comm['tfidf_cosine']['pr_auc']:.3f})", "#2878E8", "--"),
        ]:
            curve = reg_comm[key]["pr_curve"]
            ax2.plot(curve["recall"], curve["precision"], color=color, linestyle=ls, linewidth=2.2, label=label)
        ex = reg_comm["exact_normalized"]
        ax2.scatter(
            [ex["recall"]],
            [ex["precision"]],
            color="#6A1B9A",
            s=80,
            zorder=5,
            label=f"Regra Exata (Rec {ex['recall']*100:.1f}%, F1 {ex['f1']*100:.1f}%)",
        )
        ax2.set_xlim(0, 1.03)
        ax2.set_ylim(0, 1.06)
        ax2.set_xlabel("Revocação (pares verdadeiros recuperados)")
        ax2.set_ylabel("Precisão (pares corretos entre aceitos)")
        ax2.set_title("Curva PR no cenário CRM/Comercial (5 folds disjuntos)", fontsize=10.5, fontweight="bold", color="#0B2038", pad=34)
        ax2.legend(frameon=True, facecolor="white", edgecolor="#E2E8F0", fontsize=8.3, loc="lower left")

        for d in out_dirs:
            d.mkdir(parents=True, exist_ok=True)
            fig.savefig(d / "mdm_fig3_record_linkage_cv.svg", facecolor="white")
        plt.close(fig)


_STOPWORDS = {
    "SA", "S", "A", "LTDA", "CIA", "COMPANHIA", "FUNDO", "BANCO", "DTVM", "CTVM",
    "DISTRIBUIDORA", "CORRETORA", "GESTORA", "GESTAO", "ADMINISTRADORA", "ADM",
    "INVESTIMENTOS", "INVESTIMENTO", "PARTICIPACOES", "PARTICIPACAO", "CAPITAL",
    "ASSET", "MANAGEMENT", "BRASIL", "SOCIEDADE", "ANONIMA", "LIMITADA", "TITULOS",
    "VALORES", "MOBILIARIOS", "CREDITO", "FINANCIAMENTO", "DE", "DA", "DO", "DOS",
    "DAS", "E", "EM", "NO", "NA", "SERVICOS", "CONSULTORIA", "RECURSOS",
}
_LEGAL_SUFFIX_RE = re.compile(
    r"\b(S\s+A|S/?A|LTDA|EIRELI|ME|EPP|CIA|COMPANHIA|DTVM|CTVM)\b", flags=re.IGNORECASE
)


def _normalize_text(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text or "")
    ascii_txt = "".join(ch for ch in nfkd if not unicodedata.combining(ch)).upper()
    ascii_txt = re.sub(r"[^A-Z0-9\s]", " ", ascii_txt)
    return re.sub(r"\s+", " ", ascii_txt).strip()


def _normalize_core(text: str) -> str:
    norm = _normalize_text(text)
    core = _LEGAL_SUFFIX_RE.sub(" ", norm)
    return re.sub(r"\s+", " ", core).strip()


def _informative_tokens(norm_text: str) -> list[str]:
    return [t for t in norm_text.split() if len(t) >= 3 and t not in _STOPWORDS]


def _best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    prec, rec, thresh = precision_recall_curve(y_true, scores)
    if len(thresh) == 0:
        return 0.5
    f1s = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    best_idx = int(np.argmax(f1s))
    return float(thresh[best_idx])


def _assign_disjoint_entity_folds(cnpjs_a: list[str], cnpjs_b: list[str], n_folds: int = 5) -> np.ndarray:
    """Assign each pair to a CV fold such that connected CNPJ components never cross folds."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for a, b in zip(cnpjs_a, cnpjs_b):
        union(a, b)

    comp_indices: dict[str, list[int]] = defaultdict(list)
    for idx, a in enumerate(cnpjs_a):
        comp_indices[find(a)].append(idx)

    fold_loads = [0] * n_folds
    fold_ids = np.zeros(len(cnpjs_a), dtype=int)
    for _, idxs in sorted(comp_indices.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        target_fold = int(np.argmin(fold_loads))
        fold_ids[idxs] = target_fold
        fold_loads[target_fold] += len(idxs)
    return fold_ids


def _evaluate_linkage_regime(
    entities: list[CvmRecord],
    names: list[str],
    all_true_cross_pairs: int,
) -> dict:
    norm_full = [_normalize_text(n) for n in names]
    norm_core = [_normalize_core(n) for n in names]
    tok_lists = [_informative_tokens(nf) for nf in norm_full]
    tok_sets = [set(tl) for tl in tok_lists]
    ufs = [(r.uf or "").strip().upper() for r in entities]

    # Nominal blocking (strictly blind to CNPJ): informative tokens (DF <= 120), 6-char core prefix & 4-char core prefix
    inverted: dict[str, list[int]] = defaultdict(list)
    for idx, tokens in enumerate(tok_sets):
        for tok in tokens:
            inverted[f"T:{tok}"].append(idx)
        if len(norm_core[idx]) >= 5:
            inverted[f"P:{norm_core[idx][:6]}"].append(idx)
        if len(norm_core[idx]) >= 4:
            inverted[f"Q:{norm_core[idx][:4]}"].append(idx)

    cand_set: set[tuple[int, int]] = set()
    for block_key, idxs in inverted.items():
        max_len = 120 if block_key.startswith("T:") else (90 if block_key.startswith("P:") else 45)
        if len(idxs) < 2 or len(idxs) > max_len:
            continue
        for pos_a in range(len(idxs)):
            i = idxs[pos_a]
            src_i = entities[i].source
            for pos_b in range(pos_a + 1, len(idxs)):
                j = idxs[pos_b]
                if entities[j].source != src_i:
                    a, b = (i, j) if i < j else (j, i)
                    cand_set.add((a, b))

    pairs = sorted(cand_set)
    idx_a = [p[0] for p in pairs]
    idx_b = [p[1] for p in pairs]
    y = np.array([1 if entities[i].cnpj == entities[j].cnpj else 0 for i, j in pairs], dtype=int)
    block_true = int(y.sum())
    block_recall = block_true / max(all_true_cross_pairs, 1)

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=2, sublinear_tf=True)
    tfidf_mat = vec.fit_transform([nc or nf for nc, nf in zip(norm_core, norm_full)])

    n_pairs = len(pairs)
    tfidf_cos = np.zeros(n_pairs, dtype=float)
    seq_ratio = np.zeros(n_pairs, dtype=float)
    tok_jaccard = np.zeros(n_pairs, dtype=float)
    tok_overlap_min = np.zeros(n_pairs, dtype=float)
    substring_hit = np.zeros(n_pairs, dtype=float)
    uf_conflict = np.zeros(n_pairs, dtype=float)
    exact_norm = np.zeros(n_pairs, dtype=int)

    chunk = 4000
    for start in range(0, n_pairs, chunk):
        end = min(start + chunk, n_pairs)
        rows_a = tfidf_mat[idx_a[start:end]]
        rows_b = tfidf_mat[idx_b[start:end]]
        tfidf_cos[start:end] = np.asarray(rows_a.multiply(rows_b).sum(axis=1)).ravel()

    for k, (i, j) in enumerate(pairs):
        ca = norm_core[i] or norm_full[i]
        cb = norm_core[j] or norm_full[j]
        exact_norm[k] = 1 if ca == cb and ca != "" else 0
        seq_ratio[k] = difflib.SequenceMatcher(None, ca, cb).ratio()
        sa, sb = tok_sets[i], tok_sets[j]
        inter = len(sa & sb)
        union_len = len(sa | sb)
        min_len = min(len(sa), len(sb))
        tok_jaccard[k] = (inter / union_len) if union_len else 0.0
        tok_overlap_min[k] = (inter / min_len) if min_len else 0.0
        shorter, longer = (ca, cb) if len(ca) <= len(cb) else (cb, ca)
        substring_hit[k] = 1.0 if (len(shorter) >= 5 and shorter in longer) else 0.0
        uf_conflict[k] = 1.0 if (ufs[i] and ufs[j] and ufs[i] != ufs[j]) else 0.0

    X = np.column_stack([tfidf_cos, seq_ratio, tok_jaccard, tok_overlap_min, substring_hit, uf_conflict])
    folds = _assign_disjoint_entity_folds(
        [entities[i].cnpj for i in idx_a],
        [entities[j].cnpj for j in idx_b],
        n_folds=5,
    )

    oof_tfidf_pred = np.zeros(n_pairs, dtype=int)
    oof_ml_score = np.zeros(n_pairs, dtype=float)
    oof_ml_pred = np.zeros(n_pairs, dtype=int)
    feature_coefs: list[np.ndarray] = []

    for f in range(5):
        val_mask = folds == f
        tr_mask = ~val_mask

        th_tfidf = _best_f1_threshold(y[tr_mask], tfidf_cos[tr_mask])
        oof_tfidf_pred[val_mask] = (tfidf_cos[val_mask] >= th_tfidf).astype(int)

        clf = HistGradientBoostingClassifier(max_depth=3, min_samples_leaf=15, learning_rate=0.06, max_iter=150, random_state=42)
        clf.fit(X[tr_mask], y[tr_mask])
        val_scores = clf.predict_proba(X[val_mask])[:, 1]
        oof_ml_score[val_mask] = val_scores
        oof_ml_pred[val_mask] = (val_scores >= 0.50).astype(int)
        feature_coefs.append(np.array([np.corrcoef(X[val_mask, col], val_scores)[0, 1] if np.std(X[val_mask, col]) > 0 else 0.0 for col in range(X.shape[1])]))

    def _summarize_model(preds: np.ndarray, scores: np.ndarray | None) -> dict:
        prec = float(precision_score(y, preds, zero_division=0))
        rec = float(recall_score(y, preds, zero_division=0))
        f1 = float(f1_score(y, preds, zero_division=0))
        out = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "end_to_end_recall": round(rec * block_recall, 4),
            "f1": round(f1, 4),
            "pr_auc": round(float(average_precision_score(y, scores if scores is not None else preds)), 4),
            "predicted_positives": int(preds.sum()),
        }
        if scores is not None:
            p_curve, r_curve, _ = precision_recall_curve(y, scores)
            step = max(len(p_curve) // 60, 1)
            out["pr_curve"] = {
                "precision": [round(float(v), 4) for v in p_curve[::step]] + [round(float(p_curve[-1]), 4)],
                "recall": [round(float(v), 4) for v in r_curve[::step]] + [round(float(r_curve[-1]), 4)],
            }
        return out

    recovered_examples = []
    for k in np.argsort(-oof_ml_score):
        if y[k] == 1 and exact_norm[k] == 0 and oof_ml_pred[k] == 1:
            i, j = pairs[k]
            recovered_examples.append(
                {
                    "cnpj": entities[i].cnpj,
                    "source_a": entities[i].source,
                    "name_a": names[i],
                    "source_b": entities[j].source,
                    "name_b": names[j],
                    "ml_score": round(float(oof_ml_score[k]), 3),
                    "tfidf_cos": round(float(tfidf_cos[k]), 3),
                }
            )
            if len(recovered_examples) >= 4:
                break

    mean_coefs = np.mean(feature_coefs, axis=0)
    feat_names = ["tfidf_char_cos", "seq_ratio", "token_jaccard", "token_overlap_min", "substring_hit", "uf_conflict"]
    return {
        "candidate_pairs": int(n_pairs),
        "candidate_positives": block_true,
        "candidate_negatives": int(n_pairs - block_true),
        "blocking_recall": round(float(block_recall), 4),
        "classifier_type": "HistGradientBoostingClassifier",
        "models": {
            "exact_normalized": _summarize_model(exact_norm, None),
            "tfidf_cosine": _summarize_model(oof_tfidf_pred, tfidf_cos),
            "hybrid_hgb": _summarize_model(oof_ml_pred, oof_ml_score),
            "hybrid_logistic": _summarize_model(oof_ml_pred, oof_ml_score),
        },
        "feature_coefficients": {name: round(float(val), 4) for name, val in zip(feat_names, mean_coefs)},
        "recovered_nominal_drift_examples": recovered_examples,
    }


def run_blind_record_linkage(records: list[CvmRecord]) -> dict:
    """Compare blind nominal linkage on formal corporate name vs real CRM/commercial trade-name drift."""
    dedup_map: dict[tuple[str, str], CvmRecord] = {}
    for rec in records:
        if not rec.denom_social:
            continue
        key = (rec.source, rec.cnpj)
        if key not in dedup_map:
            dedup_map[key] = rec
    entities = sorted(dedup_map.values(), key=lambda r: (r.source, r.cnpj))

    by_cnpj_idx: dict[str, list[int]] = defaultdict(list)
    for idx, r in enumerate(entities):
        by_cnpj_idx[r.cnpj].append(idx)
    all_true_cross_pairs = sum(len(idxs) * (len(idxs) - 1) // 2 for idxs in by_cnpj_idx.values() if len(idxs) > 1)

    legal_names = [r.denom_social for r in entities]
    commercial_names = [
        (r.denom_comerc if (r.denom_comerc and not r.source.startswith("CVM_FI_")) else r.denom_social)
        for r in entities
    ]

    reg_legal = _evaluate_linkage_regime(entities, legal_names, all_true_cross_pairs)
    reg_comm = _evaluate_linkage_regime(entities, commercial_names, all_true_cross_pairs)
    return {
        "unique_source_entity_rows": len(entities),
        "all_true_cross_pairs": int(all_true_cross_pairs),
        "cv_folds_entity_disjoint": 5,
        "regimes": {
            "legal_name": reg_legal,
            "commercial_drift": reg_comm,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa experimento MDM em cadastros publicos da CVM.")
    parser.add_argument("--refresh", action="store_true", help="Baixa novamente os CSVs oficiais de dados.cvm.gov.br.")
    args = parser.parse_args()

    out_json = STUDY_ROOT / "resultados" / "resultados.json"
    snap_json = STUDY_ROOT / "resultados" / "cvm_snapshot.json"
    cached = json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {}
    if out_json.exists() and not args.refresh and cached.get("record_linkage", {}).get("regimes"):
        summary = cached
    else:
        if snap_json.exists() and not args.refresh:
            raw_rows = json.loads(snap_json.read_text(encoding="utf-8"))
            records = [CvmRecord(**row) for row in raw_rows]
        else:
            records = load_cvm_records()
            snap_json.parent.mkdir(parents=True, exist_ok=True)
            snap_json.write_text(json.dumps([asdict(r) for r in records], ensure_ascii=False), encoding="utf-8")

        summary = analyze_cvm_mdm(records)
        summary["record_linkage"] = run_blind_record_linkage(records)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    fig_dirs = [STUDY_ROOT / "resultados" / "figuras"]
    if SITE_ASSETS_DIR.is_dir():
        fig_dirs.append(SITE_ASSETS_DIR)
    generate_figures(summary, fig_dirs)
    rl = summary.get("record_linkage", {})
    print(json.dumps({k: v for k, v in summary.items() if k not in ("source_stats", "example_clusters", "record_linkage")}, ensure_ascii=False, indent=2))
    for rname, rdata in rl.get("regimes", {}).items():
        print(f"=== REGIME: {rname} (cand={rdata['candidate_pairs']}, pos={rdata['candidate_positives']}, block_rec={rdata['blocking_recall']}) ===")
        print(json.dumps({k: {mk: mv for mk, mv in v.items() if mk != "pr_curve"} for k, v in rdata["models"].items()}, ensure_ascii=False, indent=2))
        if rdata.get("recovered_nominal_drift_examples"):
            print("Examples recovered by ML:", json.dumps(rdata["recovered_nominal_drift_examples"][:3], ensure_ascii=False))


if __name__ == "__main__":
    main()
