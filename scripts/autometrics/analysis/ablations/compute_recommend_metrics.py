#!/usr/bin/env python3
# analysis/ablations/compute_recommend_metrics.py
"""
Compute NDCG@K + Recall@K for every task/axis/recommender,
then emit detailed CSVs, an aggregate CSV (mean ± std), and
a LaTeX table with best numbers bold-faced.

Recall uses only the top-5 ground-truth metric classes.
"""
from __future__ import annotations
import argparse, logging
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
import re

# ───────────────────────── SETTINGS ───────────────────────── #
CORR_ROOT = Path("outputs/correlation")
REC_ROOT  = Path("outputs/recommendation")
OUT_ROOT  = Path("results/ablations/recommend")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

SYNONYMS  = {"LENSMetric": "LENS"}
MODEL_MAP = {"qwen3": "llm_litellm_proxy_Qwen_Qwen3-32B",
             "gpt4o-mini": "llm_openai_gpt-4o-mini"}
TOP_M_RELEVANT = 5          # relevant set size for Recall

# ───────── prettier names + bold-best ─────────────────── #
DISPLAY_MAP = {
    "PipelinedRec(BM25→LLMRec)"  : "BM25→LLMRec",
    "PipelinedRec(ColBERT→LLMRec)": "ColBERT→LLMRec",
    "PipelinedRec(Faiss→LLMRec)" : "Faiss→LLMRec",
    # add more if you introduce new pipelines
}
def prettify(name: str) -> str:
    return DISPLAY_MAP.get(name, name).replace("_", r"\_")  # escape _ for LaTeX

def bold_best(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # numeric part for comparison
    nums = df.applymap(lambda s: float(s.split()[0]))
    for col in df.columns:
        best = nums[col].max()
        out[col] = [
            rf"\textbf{{{v}}}" if np.isclose(m, best) else v
            for v, m in zip(out[col], nums[col])
        ]
    return out

# ───────────────────────── CLI ──────────────────────────── #
def parse_args():
    p = argparse.ArgumentParser(description="Compute recommend-metrics + LaTeX")
    p.add_argument("--corr_type", default="kendall")
    p.add_argument("--model", required=True)
    p.add_argument("--k", type=int, nargs="+", default=[1,5,10,20])
    p.add_argument("--debug", action="store_true")
    return p.parse_args()

# ───────────────────────── UTILITIES ────────────────────── #
def load_ground_truth(csv_path: Path) -> Dict[str, float]:
    df = (pd.read_csv(csv_path)
            .assign(Metric_Class=lambda d: d["Metric_Class"].replace(SYNONYMS))
            .sort_values("Correlation", ascending=False)
            .drop_duplicates("Metric_Class", keep="first"))
    df["rank"] = np.arange(1, len(df)+1)
    df["rel"]  = 1 / df["rank"]
    return dict(zip(df["Metric_Class"], df["rel"]))

def load_recs(csv: Path)->Dict[str,List[str]]:
    df = pd.read_csv(csv, index_col=0)
    out={}
    for rec,row in df.iterrows():
        seen,items=set(),[]
        for m in row.values.tolist():
            if isinstance(m,str) and m:
                m=SYNONYMS.get(m,m)
                if m not in seen: items.append(m); seen.add(m)
        out[rec]=items
    return out

def ndcg_at_k(gt:Dict[str,float],pred:List[str],k:int)->float:
    items=list(gt)
    y_true=np.array([[gt[x] for x in items]])
    scores={m:(len(pred)-i) for i,m in enumerate(pred)}
    y_pred=np.array([[scores.get(x,0) for x in items]])
    return ndcg_score(y_true,y_pred,k=k)

def recall_at_k(gt:Dict[str,float],pred:List[str],k:int)->float:
    rel=set(list(gt)[:TOP_M_RELEVANT])
    return len(rel&set(pred[:k]))/len(rel) if rel else 0.0

def order_cols(cols:List[str])->List[str]:
    ndcg=sorted([c for c in cols if c.startswith("ndcg@")], key=lambda s:int(s.split("@")[1]))
    rec =sorted([c for c in cols if c.startswith("recall@")], key=lambda s:int(s.split("@")[1]))
    return ["recommender"]+ndcg+rec

# ──────────  LaTeX helpers (fixed header) ──────────────── #
def bold_best(df:pd.DataFrame)->pd.DataFrame:
    out=df.copy()
    # numeric part before we insert bold
    vals=df.applymap(lambda s: float(s.split()[0]))
    for col in df.columns:
        best=vals[col].max()
        mask=vals[col]==best
        out.loc[mask,col]=out.loc[mask,col].apply(lambda s:r"\textbf{"+s+r"}")
    return out

def make_latex(wide: pd.DataFrame, corr: str, model_alias: str) -> str:
    ndcg_cols = [c for c in wide.columns if c.startswith("ndcg@")]
    rec_cols  = [c for c in wide.columns if c.startswith("recall@")]
    head_ndcg = " & ".join(f"@{c.split('@')[1]}" for c in ndcg_cols)
    head_rec  = " & ".join(f"@{c.split('@')[1]}" for c in rec_cols)

    col_spec = "l" + "c"*len(ndcg_cols) + "|" + "c"*len(rec_cols)
    body = "\n".join(
        "    " + " & ".join(row.values) + r"\\"
        for _, row in wide.iterrows()
    )

    caption = (rf"Average performance (± std) across all tasks/axes using "
               rf"{corr.title()} ground truth "
               rf"(recommendations from \texttt{{{model_alias}}}).")

    # ----------  NEW: create LaTeX-safe label  ----------
    raw_label = f"rec_metrics_{corr}_{model_alias}"
    safe_label = re.sub(r"[^A-Za-z0-9]+", "", raw_label)   # keep only letters & digits
    # ----------------------------------------------------

    return rf"""\begin{{table*}}[h]
  \centering
  \resizebox{{\textwidth}}{{!}}{{%
  \begin{{tabular}}{{{col_spec}}}
    \toprule
    \rowcolor[gray]{{0.85}}
    & \multicolumn{{{len(ndcg_cols)}}}{{c|}}{{\textbf{{NDCG}}}} &
      \multicolumn{{{len(rec_cols)}}}{{c}}{{\textbf{{Recall}}}}\\
    \rowcolor[gray]{{0.85}}
    Method & {head_ndcg} & {head_rec}\\
    \midrule
{body}
    \bottomrule
  \end{{tabular}}}}
  \caption{{{caption}}}
  \label{{tab:{safe_label}}}
\end{{table*}}"""

# ─────────────────────────── MAIN ──────────────────────── #
def main():
    args=parse_args()
    ks=sorted(set(args.k))
    model_folder=MODEL_MAP.get(args.model.lower(),args.model)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(levelname)s | %(message)s")

    ndcg_rows=[]; recall_rows=[]
    for task in CORR_ROOT.iterdir():
        cdir=task/args.corr_type
        if not cdir.is_dir(): continue
        for corr_file in cdir.glob("correlation_*.csv"):
            axis=corr_file.stem.replace("correlation_","")
            gt=load_ground_truth(corr_file)
            rec_path=REC_ROOT/task.name/model_folder/axis/"recommendations.csv"
            if not rec_path.exists(): continue
            recs=load_recs(rec_path)
            for rec_name,preds in recs.items():
                if args.debug:
                    logging.debug(f"{task.name}/{axis} | {rec_name} | GTtop5={list(gt)[:TOP_M_RELEVANT]} | Predtop5={preds[:5]}")
                for k in ks:
                    ndcg_rows.append(dict(task=task.name,axis=axis,recommender=rec_name,
                                          k=k,ndcg=ndcg_at_k(gt,preds,k)))
                    recall_rows.append(dict(task=task.name,axis=axis,recommender=rec_name,
                                            k=k,recall=recall_at_k(gt,preds,k)))

    # Detailed CSVs
    ndcg_df=pd.DataFrame(ndcg_rows)
    recall_df=pd.DataFrame(recall_rows)
    ndcg_df.to_csv(OUT_ROOT/f"ndcg_{args.corr_type}_{model_folder}.csv",index=False)
    recall_df.to_csv(OUT_ROOT/f"recall_{args.corr_type}_{model_folder}.csv",index=False)

    # Aggregate mean±std
    long=pd.concat([
        ndcg_df.assign(metric=lambda d:"ndcg@"+d.k.astype(str),
                       value=ndcg_df.ndcg)[["recommender","metric","value"]],
        recall_df.assign(metric=lambda d:"recall@"+d.k.astype(str),
                         value=recall_df.recall)[["recommender","metric","value"]]
    ])
    stats=(long.groupby(["recommender","metric"])["value"]
                .agg(["mean","std"]).reset_index())
    stats["formatted"]=stats["mean"].round(3).astype(str)+ \
                       " {\\scriptsize $\\pm$ "+stats["std"].round(3).astype(str)+"}"
    wide=(stats.pivot(index="recommender",columns="metric",values="formatted")
               .reset_index())
    wide=wide[order_cols(list(wide.columns))]
    # escape underscores in names for LaTeX
    wide["recommender"]=wide["recommender"].str.replace("_",r"\_",regex=False)
    wide_bold=bold_best(wide.set_index("recommender")).reset_index()

    # ───── Write aggregate CSV + LaTeX ─────────────────── #
    agg_path = OUT_ROOT / f"aggregate_{args.corr_type}_{model_folder}.csv"

    # prettify row names for display
    wide_disp = wide.copy()
    wide_disp["recommender"] = wide_disp["recommender"].apply(prettify)

    # bold the best numbers
    wide_bold = bold_best(wide_disp.set_index("recommender")).reset_index()
    wide_bold.to_csv(agg_path, index=False)

    tex_path  = OUT_ROOT / f"table_{args.corr_type}_{model_folder}.tex"
    tex_path.write_text(make_latex(wide_bold, args.corr_type, args.model))

    logging.info(f"Aggregate CSV → {agg_path.name}")
    logging.info(f"LaTeX table  → {tex_path.name}")

if __name__ == "__main__":
    main()

# Example usage:
# evaluate Qwen3 recommendations, default K list
# python analysis/ablations/compute_recommend_metrics.py --corr_type kendall --model qwen3

# evaluate GPT-4o-mini recommendations, default K list
# python analysis/ablations/compute_recommend_metrics.py --corr_type kendall --model gpt4o-mini