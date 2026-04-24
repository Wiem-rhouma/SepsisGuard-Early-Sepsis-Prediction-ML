import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    cross_val_score, cross_val_predict, learning_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    ConfusionMatrixDisplay, matthews_corrcoef,
    make_scorer, brier_score_loss
)

try:
    import xgboost as xgb
    HAS_XGB = True
    print(" XGBoost disponible")
except ImportError:
    HAS_XGB = False
    print("XGBoost absent → pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LGB = True
    print(" LightGBM disponible")
except ImportError:
    HAS_LGB = False

warnings.filterwarnings("ignore")

# ─── helpers ─────────────────────────────────────────────────
def savefig(path, dpi=130):
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f" Figure sauvegardée : {path}")

def best_f1_threshold(y_true, y_proba):
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    f1  = 2 * prec * rec / (prec + rec + 1e-8)
    idx = f1[:-1].argmax()
    return float(thr[idx])

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.figsize": (12, 6),
                     "axes.titlesize": 14, "axes.labelsize": 11})

# ============================================================
# CONFIG
# ============================================================
TARGET            = "SepsisLabel"
DATA_PATH         = r"C:\Users\wiwir\Downloads\db\Dataset.csv"
UNDERSAMPLE_RATIO = "1:2"
RATIO_MAP         = {"1:1": 1, "1:2": 2, "1:3": 3}
PROTECTED         = ["Hour","ICULOS","HR","Resp","MAP","SBP","Temp","O2Sat"]
VITALS            = ["HR","Resp","MAP","SBP","Temp","O2Sat","DBP"]
SEED              = 42

# ============================================================
# CHARGEMENT
# ============================================================
print("=" * 65)
print("  SEPSIS — PIPELINE ")
print("=" * 65)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Fichier introuvable : {DATA_PATH}")

with open(DATA_PATH) as f:
    first = f.readline()
sep_c = "|" if "|" in first else ";" if ";" in first else ","
df_raw = pd.read_csv(DATA_PATH, sep=sep_c)
if df_raw.shape[1] == 1:
    df_raw = pd.read_csv(DATA_PATH, sep=",")

df_raw = df_raw.drop(columns=[c for c in df_raw.columns
                               if str(c).strip()=="" or "Unnamed" in str(c)],
                     errors="ignore")
if "Patient_ID" in df_raw.columns:
    df_raw = df_raw.drop(columns=["Patient_ID"])
if TARGET not in df_raw.columns:
    raise KeyError(f"'{TARGET}' introuvable.")

df_raw  = df_raw.select_dtypes(include=[np.number])
n_pos   = int(df_raw[TARGET].sum())
n_neg   = len(df_raw) - n_pos
prev    = df_raw[TARGET].mean()

print(f"\nDataset brut : {len(df_raw):,} lignes × {df_raw.shape[1]} colonnes")
print(f"Septiques    : {n_pos:,}  ({prev*100:.2f}%)")
print(f"Non-sep.     : {n_neg:,}")

# colonnes catégories pour visualisation
vitals_avail = [c for c in ["HR","O2Sat","Temp","SBP","MAP","DBP","Resp","EtCO2"]
                if c in df_raw.columns]
labs_avail   = [c for c in ["Lactate","WBC","Creatinine","Glucose","Platelets",
                             "Bilirubin_total","pH","HCO3","Hgb","Hct"]
                if c in df_raw.columns]
demo_avail   = [c for c in ["Age","Gender","HospAdmTime","ICULOS","Hour"]
                if c in df_raw.columns]

# ============================================================
# ÉTAPE 0 — UNDERSAMPLING
# ============================================================
print("\n" + "="*65)
print("ÉTAPE 0 — UNDERSAMPLING (garde TOUS les septiques)")
print("="*65)

r          = RATIO_MAP[UNDERSAMPLE_RATIO]
n_neg_keep = n_pos * r
df_pos     = df_raw[df_raw[TARGET] == 1]
df_neg     = df_raw[df_raw[TARGET] == 0].sample(n=int(n_neg_keep), random_state=SEED)
df_bal     = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=SEED).reset_index(drop=True)
prev_bal   = df_bal[TARGET].mean()

print(f"Ratio         : {UNDERSAMPLE_RATIO}")
print(f"Dataset final : {len(df_bal):,} lignes | Prévalence : {prev_bal*100:.1f}%")
assert int(df_bal[TARGET].sum()) == n_pos
print(f"{n_pos:,} septiques conservés")

# ============================================================
# ÉTAPE 1 — FEATURES TEMPORELLES
# ============================================================
print("\n" + "="*65)
print("ÉTAPE 1 — FEATURES TEMPORELLES")
print("="*65)

def add_temporal(df):
    df = df.copy()
    for v in [v for v in VITALS if v in df.columns]:
        df[f"{v}_d1"] = df[v].diff(1)
        df[f"{v}_d6"] = df[v].diff(6)
        df[f"{v}_s6"] = df[v].rolling(6, min_periods=2).std()
    if "HR" in df.columns and "SBP" in df.columns:
        df["shock_idx"]   = df["HR"] / (df["SBP"].replace(0, np.nan) + 1e-5)
    if "SBP" in df.columns and "DBP" in df.columns:
        df["pulse_press"] = df["SBP"] - df["DBP"]
    return df.replace([np.inf, -np.inf], np.nan)

df_feat  = add_temporal(df_bal)
new_cols = [c for c in df_feat.columns if c not in df_bal.columns]
print(f"  +{len(new_cols)} features temporelles : {new_cols}")

# ============================================================
# ÉTAPE 2 — SPLIT 60 / 20 / 20
# ============================================================
print("\n" + "="*65)
print("ÉTAPE 2 — SPLIT TRAIN / VAL / TEST (60/20/20)")
print("="*65)

X_all = df_feat.drop(columns=[TARGET])
y_all = df_feat[TARGET].values

X_tmp,  X_te_raw, y_tmp, y_te = train_test_split(
    X_all, y_all, test_size=0.20, random_state=SEED, stratify=y_all)
X_tr_raw, X_va_raw, y_tr, y_va = train_test_split(
    X_tmp, y_tmp, test_size=0.25, random_state=SEED, stratify=y_tmp)

print(f"  Train : {len(y_tr):,}  | Sepsis : {y_tr.mean()*100:.1f}%")
print(f"  Val   : {len(y_va):,}  | Sepsis : {y_va.mean()*100:.1f}%")
print(f"  Test  : {len(y_te):,}  | Sepsis : {y_te.mean()*100:.1f}%")

# ============================================================
# ÉTAPE 3 — IMPUTATION + INDICATEURS DE MANQUANTS
# ============================================================
print("\n" + "="*65)
print("ÉTAPE 3 — IMPUTATION ")
print("="*65)

orig_cols = [c for c in X_tr_raw.columns
             if not any(x in c for x in ["_d1","_d6","_s6","shock","pulse"])]
miss_rate = X_tr_raw[orig_cols].isnull().mean()
high_miss = miss_rate[miss_rate > 0.05].index.tolist()
print(f"  Colonnes >5% manquants : {len(high_miss)} → indicateurs _miss")

def add_miss_ind(X, cols):
    X = X.copy()
    for c in cols:
        if c in X.columns:
            X[f"{c}_miss"] = X[c].isnull().astype(int)
    return X

X_tr_fl = add_miss_ind(X_tr_raw, high_miss)
X_va_fl = add_miss_ind(X_va_raw, high_miss)
X_te_fl = add_miss_ind(X_te_raw, high_miss)

imp      = SimpleImputer(strategy="median")
cols_imp = X_tr_fl.columns.tolist()
X_tr_imp = pd.DataFrame(imp.fit_transform(X_tr_fl), columns=cols_imp)
X_va_imp = pd.DataFrame(imp.transform(X_va_fl),     columns=cols_imp)
X_te_imp = pd.DataFrame(imp.transform(X_te_fl),     columns=cols_imp)

const = [c for c in X_tr_imp.columns
         if X_tr_imp[c].std() == 0 and c not in PROTECTED]
if const:
    X_tr_imp = X_tr_imp.drop(columns=const)
    X_va_imp = X_va_imp.drop(columns=[c for c in const if c in X_va_imp.columns])
    X_te_imp = X_te_imp.drop(columns=[c for c in const if c in X_te_imp.columns])

print(f"  Features totales : {X_tr_imp.shape[1]}")
print(f"  NaN résiduels   : {X_tr_imp.isnull().sum().sum()}")

# ============================================================
# ÉTAPE 4 — VISUALISATION
# ============================================================
print("\n" + "="*65)
print("ÉTAPE 4 — VISUALISATION")
print("="*65)

df_viz           = X_tr_imp.copy()
df_viz[TARGET]   = y_tr
df_viz["Statut"] = pd.Series(y_tr).map({0:"Non-Septique", 1:"Septique"}).values

# ── A — Distribution des classes ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
vc = pd.Series(y_tr).value_counts().sort_index()
axes[0].bar(["Non-Septique","Sepsis"], vc.values,
            color=["#2ECC71","#E74C3C"], width=0.5, edgecolor="white")
for bar, val in zip(axes[0].patches, vc.values):
    pct = val / vc.sum() * 100
    axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height(),
                 f"{val:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=11)
axes[0].set_title("Distribution des classes (train set)")
axes[0].set_ylabel("Nombre de lignes")
axes[1].pie(vc.values, labels=["Non-Septique","Sepsis"],
            autopct="%1.2f%%", colors=["#2ECC71","#E74C3C"],
            explode=(0, 0.10), shadow=True, startangle=90)
axes[1].set_title(f"Après undersampling {UNDERSAMPLE_RATIO} ({prev_bal*100:.1f}% sepsis)")
plt.suptitle("A — Distribution des Classes", fontsize=14, fontweight="bold")
plt.tight_layout(); savefig("A_class_distribution.png")

# ── B — Histogrammes signes vitaux ───────────────────────────
if vitals_avail:
    vp_present = [c for c in vitals_avail if c in df_viz.columns]
    ncols = 4; nrows = (len(vp_present)+ncols-1)//ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4*nrows))
    axes = axes.flatten()
    colors_h = ["#3498DB","#E74C3C","#F39C12","#2ECC71","#9B59B6","#1ABC9C","#E67E22","#34495E"]
    for i, col in enumerate(vp_present):
        axes[i].hist(df_viz[col].dropna(), bins=40, color=colors_h[i%len(colors_h)],
                     edgecolor="white", alpha=0.85)
        axes[i].set_title(col, fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Valeur"); axes[i].set_ylabel("Fréquence")
        mv = df_viz[col].mean()
        axes[i].axvline(mv, color="black", linestyle="--", lw=1.5, label=f"Moy={mv:.1f}")
        axes[i].legend(fontsize=8)
    for j in range(len(vp_present), len(axes)): axes[j].set_visible(False)
    plt.suptitle("B — Distribution des Signes Vitaux", fontsize=14, fontweight="bold")
    plt.tight_layout(); savefig("B_vitals_histograms.png")

# ── C — Boxplots signes vitaux par classe ────────────────────
if vitals_avail:
    vp_present = [c for c in vitals_avail if c in df_viz.columns]
    samp = df_viz.sample(min(8000, len(df_viz)), random_state=SEED)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8)); axes = axes.flatten()
    for i, col in enumerate(vp_present[:8]):
        sns.boxplot(data=samp, x="Statut", y=col, hue="Statut",
                    palette={"Non-Septique":"#AED6F1","Septique":"#F1948A"},
                    ax=axes[i], order=["Non-Septique","Septique"], legend=False)
        axes[i].set_title(col, fontsize=11, fontweight="bold"); axes[i].set_xlabel("")
    for j in range(len(vp_present[:8]), 8): axes[j].set_visible(False)
    plt.suptitle("C — Signes Vitaux : Septique vs Non-Septique", fontsize=14, fontweight="bold")
    plt.tight_layout(); savefig("C_vitals_boxplots.png")

# ── D — Boxplots analyses biologiques par classe ─────────────
if labs_avail:
    labs_present = [c for c in labs_avail if c in df_viz.columns]
    samp = df_viz.sample(min(8000, len(df_viz)), random_state=SEED)
    ncols = 5; nrows = (len(labs_present)+ncols-1)//ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5*nrows)); axes = axes.flatten()
    for i, col in enumerate(labs_present):
        sns.boxplot(data=samp, x="Statut", y=col, hue="Statut",
                    palette={"Non-Septique":"#AED6F1","Septique":"#F1948A"},
                    ax=axes[i], order=["Non-Septique","Septique"], legend=False)
        axes[i].set_title(col, fontsize=10, fontweight="bold"); axes[i].set_xlabel("")
    for j in range(len(labs_present), len(axes)): axes[j].set_visible(False)
    plt.suptitle("D — Analyses Biologiques : Septique vs Non-Septique", fontsize=14, fontweight="bold")
    plt.tight_layout(); savefig("D_labs_boxplots.png")

# ── E — Matrice de corrélation ───────────────────────────────
key_corr_cols = ([c for c in vitals_avail if c in df_viz.columns] +
                 [c for c in ["Lactate","WBC","Creatinine","Glucose","pH","Age","ICULOS"]
                  if c in df_viz.columns] + [TARGET])
plt.figure(figsize=(14, 10))
corr_mat = df_viz[key_corr_cols].corr()
mask = np.triu(np.ones_like(corr_mat, dtype=bool))
sns.heatmap(corr_mat, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.4, square=True,
            annot_kws={"size":8}, vmin=-1, vmax=1)
plt.title("E — Matrice de Corrélation (train set)", fontsize=14, fontweight="bold", pad=20)
plt.tight_layout(); savefig("E_correlation_heatmap.png")

# ── F — Violinplots features clés ────────────────────────────
key_violin = [c for c in ["HR","Resp","Temp","Lactate","ICULOS","Hour"] if c in df_viz.columns]
if key_violin:
    fig, axes = plt.subplots(1, len(key_violin), figsize=(4*len(key_violin), 6))
    if len(key_violin) == 1: axes = [axes]
    samp_v = df_viz.sample(min(5000, len(df_viz)), random_state=SEED)
    for i, col in enumerate(key_violin):
        sns.violinplot(data=samp_v.dropna(subset=[col]),
                       x="Statut", y=col, hue="Statut",
                       palette={"Non-Septique":"#AED6F1","Septique":"#F1948A"},
                       ax=axes[i], inner="quartile",
                       order=["Non-Septique","Septique"], legend=False)
        axes[i].set_title(col, fontsize=12, fontweight="bold"); axes[i].set_xlabel("")
    plt.suptitle("F — Violinplots des Variables Clés", fontsize=14, fontweight="bold")
    plt.tight_layout(); savefig("F_violinplots.png")

df_viz = df_viz.drop(columns=["Statut"])

# ============================================================
# ÉTAPE 5 — OUTLIERS
# ============================================================
print("\n" + "="*65)
print("ÉTAPE 5 — DÉTECTION DES VALEURS ABERRANTES")
print("="*65)

numeric_cols = [c for c in X_tr_imp.select_dtypes(include=np.number).columns
                if not any(x in c for x in ["_d1","_d6","_s6","shock","pulse","_miss"])]

# ── G — Vue globale normalisée ────────────────────────────────
if vitals_avail:
    vp_norm = [c for c in vitals_avail if c in X_tr_imp.columns]
    plt.figure(figsize=(16, 5))
    df_norm = X_tr_imp[vp_norm].copy()
    df_norm = (df_norm - df_norm.mean()) / (df_norm.std() + 1e-8)
    sns.boxplot(data=df_norm, palette="Set2")
    plt.axhline( 3, color="red", linestyle="--", lw=1.5, label="Seuil ±3σ")
    plt.axhline(-3, color="red", linestyle="--", lw=1.5)
    plt.title("G — Vue Globale Outliers (Signes Vitaux normalisés)", fontsize=13, fontweight="bold")
    plt.xticks(rotation=30); plt.legend()
    plt.tight_layout(); savefig("G_outliers_global.png")

# Rapport IQR
outlier_report = []
for col in numeric_cols:
    if col not in X_tr_imp.columns: continue
    Q1, Q3 = X_tr_imp[col].quantile(0.25), X_tr_imp[col].quantile(0.75)
    IQR = Q3 - Q1
    if IQR == 0: continue
    lo, hi = Q1-1.5*IQR, Q3+1.5*IQR
    n_out = ((X_tr_imp[col] < lo) | (X_tr_imp[col] > hi)).sum()
    pct   = round(n_out / X_tr_imp[col].notna().sum() * 100, 2)
    outlier_report.append({"Variable":col, "Q1":round(Q1,2), "Q3":round(Q3,2),
                            "Borne_inf":round(lo,2), "Borne_sup":round(hi,2),
                            "N_outliers":n_out, "%":pct})

out_df  = pd.DataFrame(outlier_report).sort_values("N_outliers", ascending=False)
top_out = out_df[out_df["N_outliers"] > 0].head(15)
print("\n  Top 15 variables avec outliers :")
print(top_out.to_string(index=False))
print(f"\n  Total outliers détectés : {out_df['N_outliers'].sum():,}")

# ── H — Rapport outliers barh ─────────────────────────────────
if len(top_out) > 0:
    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.barh(top_out["Variable"], top_out["N_outliers"],
                   color="#E74C3C", alpha=0.8, edgecolor="white")
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=9)
    ax.set_xlabel("Nombre d'outliers (IQR)")
    ax.set_title("H — Outliers par Variable (IQR, train set)", fontsize=13, fontweight="bold")
    plt.tight_layout(); savefig("H_outliers_report.png")

# ============================================================
# ÉTAPE 6 — PRÉTRAITEMENT (Clipping IQR + StandardScaler)
# ============================================================
print("\n" + "="*65)
print("ÉTAPE 6 — CLIPPING IQR + STANDARDSCALER")
print("="*65)

clip_cols = [c for c in X_tr_imp.columns
             if not any(x in c for x in ["_d1","_d6","_s6","shock","pulse","_miss"])]

X_tr_c, X_va_c, X_te_c = X_tr_imp.copy(), X_va_imp.copy(), X_te_imp.copy()
for col in clip_cols:
    if col not in X_tr_c.columns: continue
    Q1, Q3 = X_tr_c[col].quantile(0.25), X_tr_c[col].quantile(0.75)
    IQR = Q3 - Q1
    if IQR == 0: continue
    lo, hi = Q1-1.5*IQR, Q3+1.5*IQR
    for df_ in [X_tr_c, X_va_c, X_te_c]:
        if col in df_.columns: df_[col] = df_[col].clip(lo, hi)

sc      = StandardScaler()
cols_sc = X_tr_c.columns.tolist()
X_tr_sc = pd.DataFrame(sc.fit_transform(X_tr_c), columns=cols_sc)
X_va_sc = pd.DataFrame(sc.transform(X_va_c),     columns=cols_sc)
X_te_sc = pd.DataFrame(sc.transform(X_te_c),     columns=cols_sc)

print(f"  Shape train : {X_tr_sc.shape}")
print(f"  Moy ≈ {X_tr_sc.mean().mean():.5f}  |  Std ≈ {X_tr_sc.std().mean():.4f}")

# ── I — Effet normalisation avant/après ───────────────────────
show_c = [c for c in ["HR","SBP","MAP","Resp","ICULOS"] if c in X_tr_imp.columns]
if show_c:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    X_tr_imp[show_c].plot(kind="box", ax=axes[0], patch_artist=True,
                           boxprops=dict(facecolor="#AED6F1"))
    axes[0].set_title("Avant normalisation", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Valeur brute")
    X_tr_sc[show_c].plot(kind="box", ax=axes[1], patch_artist=True,
                          boxprops=dict(facecolor="#A9DFBF"))
    axes[1].set_title("Après normalisation (StandardScaler)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("z-score")
    plt.suptitle("I — Effet de la Normalisation", fontsize=13, fontweight="bold")
    plt.tight_layout(); savefig("I_normalization_effect.png")

# ============================================================
# ÉTAPE 7 — SÉLECTION DES FEATURES (2 méthodes)
# ============================================================
print("\n" + "="*65)
print("ÉTAPE 7 — SÉLECTION DES FEATURES (2 méthodes, consensus)")
print("="*65)

# Méthode 1 — Corrélation de Pearson
corr_t = X_tr_sc.corrwith(
    pd.Series(y_tr, index=X_tr_sc.index)
).abs().sort_values(ascending=False).dropna()
print("\n  Top 15 (Corrélation) :"); print(corr_t.head(15).round(4).to_string())

# ── J — Corrélation Pearson bar ───────────────────────────────
plt.figure(figsize=(12, 6))
colors_corr = ["#E74C3C" if v > 0.05 else "#95A5A6" for v in corr_t.head(20).values]
corr_t.head(20).plot(kind="bar", color=colors_corr, edgecolor="white")
plt.axhline(0.05, color="red", linestyle="--", lw=1.5, label="Seuil 0.05")
plt.title("J — Corrélation Absolue avec SepsisLabel", fontsize=13, fontweight="bold")
plt.xlabel("Variable"); plt.ylabel("Corrélation absolue")
plt.legend(); plt.xticks(rotation=45, ha="right")
plt.tight_layout(); savefig("J_correlation_target.png")

# Méthode 2 — SelectKBest ANOVA
sk = SelectKBest(f_classif, k="all"); sk.fit(X_tr_sc, y_tr)
kbest = pd.Series(sk.scores_, index=X_tr_sc.columns).sort_values(ascending=False).dropna()
print("\n  Top 15 (F-score ANOVA) :"); print(kbest.head(15).round(2).to_string())

# ── K — F-scores bar ──────────────────────────────────────────
plt.figure(figsize=(12, 6))
colors_kb = ["#E74C3C" if i < 10 else "#95A5A6" for i in range(len(kbest.head(20)))]
kbest.head(20).plot(kind="bar", color=colors_kb, edgecolor="white")
plt.title("K — SelectKBest : F-scores ANOVA", fontsize=13, fontweight="bold")
plt.xlabel("Variable"); plt.ylabel("F-score"); plt.xticks(rotation=45, ha="right")
plt.tight_layout(); savefig("K_selectkbest.png")

# Consensus ≥ 2/2 méthodes (présent dans les deux top-20)
top_c = set(corr_t.head(20).index)
top_k = set(kbest.head(20).index)
sel   = [f for f in X_tr_sc.columns if f in top_c and f in top_k]
if len(sel) < 10: sel = list(top_c | top_k)

# Suppression redondants
cm_s  = X_tr_sc[sel].corr().abs()
up    = cm_s.where(np.triu(np.ones_like(cm_s, dtype=bool), k=1))
red   = [c for c in up.columns if any(up[c] > 0.95) and c not in PROTECTED]
if red:
    print(f"  Redondants supprimés : {red}")
    sel = [f for f in sel if f not in red]
for pf in PROTECTED:
    if pf in X_tr_sc.columns and pf not in sel:
        sel.append(pf); print(f"  Protégée forcée : {pf}")
sel = [f for f in kbest.index if f in sel]

print(f"\n  Features retenues ({len(sel)}) :")
for f in sel:
    tag = " [Δ]" if any(x in f for x in ["_d1","_d6","_s6","shock","pulse"]) else ""
    print(f"  • {f:<28}  corr={corr_t.get(f,0):.3f}  F={kbest.get(f,0):.1f}{tag}")

# ── M — Comparaison 2 méthodes grouped bar ────────────────────
top_show = sel[:12]
fig, ax = plt.subplots(figsize=(16, 6))
xp = np.arange(len(top_show)); w = 0.30
ax.bar(xp - w/2, [corr_t.get(f,0) for f in top_show],
       w, label="Corrélation", color="#3498DB", alpha=0.85)
ax.bar(xp + w/2, [kbest.get(f,0)/max(kbest.max(),1) for f in top_show],
       w, label="F-score (normalisé)", color="#E74C3C", alpha=0.85)
ax.set_xticks(xp); ax.set_xticklabels(top_show, rotation=35, ha="right")
ax.set_ylabel("Score normalisé")
ax.set_title("M — Comparaison des 2 Méthodes de Sélection", fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout(); savefig("M_feature_selection_comparison.png")

sv  = [f for f in sel if f in X_va_sc.columns]
st  = [f for f in sel if f in X_te_sc.columns]
X_tr = X_tr_sc[sel].values
X_va = X_va_sc[sv].values
X_te = X_te_sc[st].values
print(f"\n  train:{X_tr.shape} | val:{X_va.shape} | test:{X_te.shape}")

# ============================================================
# ÉTAPE 8 — MODÈLES + HYPERPARAMETER TUNING (GridSearchCV CV=4)
# ============================================================
print("\n" + "="*65)
print("ÉTAPE 8 — TUNING DES HYPERPARAMÈTRES (GridSearchCV, CV=4)")
print("="*65)

from sklearn.model_selection import GridSearchCV

cv4 = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)

# ── Logistic Regression ──────────────────────────────────────
print("\n Logistic Regression...")
param_lr = {
    "C":       [0.01, 0.1, 1.0, 10.0],
    "solver":  ["lbfgs", "saga"],
    "max_iter":[1000]
}
gs_lr = GridSearchCV(
    LogisticRegression(random_state=SEED, n_jobs=1),
    param_lr, cv=cv4, scoring="roc_auc", n_jobs=1, verbose=0
)
gs_lr.fit(X_tr, y_tr)
print(f"  Meilleurs params : {gs_lr.best_params_}")
print(f"  ROC-AUC CV=4    : {gs_lr.best_score_:.4f}")

# ── Random Forest ────────────────────────────────────────────
print("\n Random Forest...")
param_rf = {
    "n_estimators":    [200, 300],
    "max_depth":       [10, 15, None],
    "min_samples_leaf":[5, 10],
    "max_features":    ["sqrt"]
}
gs_rf = GridSearchCV(
    RandomForestClassifier(random_state=SEED, n_jobs=4),
    param_rf, cv=cv4, scoring="roc_auc", n_jobs=1, verbose=0
)
gs_rf.fit(X_tr, y_tr)
print(f"  Meilleurs params : {gs_rf.best_params_}")
print(f"  ROC-AUC CV=4    : {gs_rf.best_score_:.4f}")

# ── HistGradientBoosting ──────────────────────────────────────
print("\n HistGradientBoosting...")
param_hgb = {
    "max_iter":        [200, 300],
    "learning_rate":   [0.05, 0.1],
    "max_depth":       [4, 6],
    "min_samples_leaf":[20, 50]
}
gs_hgb = GridSearchCV(
    HistGradientBoostingClassifier(random_state=SEED),
    param_hgb, cv=cv4, scoring="roc_auc", n_jobs=1, verbose=0
)
gs_hgb.fit(X_tr, y_tr)
print(f"  Meilleurs params : {gs_hgb.best_params_}")
print(f"  ROC-AUC CV=4    : {gs_hgb.best_score_:.4f}")

# ── XGBoost ───────────────────────────────────────────────────
if HAS_XGB:
    print("\n XGBoost...")
    param_xgb = {
        "n_estimators":      [200, 300],
        "learning_rate":     [0.05, 0.1],
        "max_depth":         [4, 6],
        "min_child_weight":  [20, 30],
        "subsample":         [0.8],
        "colsample_bytree":  [0.7]
    }
    gs_xgb = GridSearchCV(
        xgb.XGBClassifier(
            eval_metric="logloss", random_state=SEED,
            n_jobs=4, verbosity=0
        ),
        param_xgb, cv=cv4, scoring="roc_auc", n_jobs=1, verbose=0
    )
    gs_xgb.fit(X_tr, y_tr)
    print(f"  Meilleurs params : {gs_xgb.best_params_}")
    print(f"  ROC-AUC CV=4    : {gs_xgb.best_score_:.4f}")

# ── Tableau récap tuning ──────────────────────────────────────
tuning_rows = [
    {"Modèle": "Logistic Regression",    "Meilleurs paramètres": str(gs_lr.best_params_),  "ROC-AUC CV=4": round(gs_lr.best_score_, 4)},
    {"Modèle": "Random Forest",          "Meilleurs paramètres": str(gs_rf.best_params_),  "ROC-AUC CV=4": round(gs_rf.best_score_, 4)},
    {"Modèle": "HistGradientBoosting",   "Meilleurs paramètres": str(gs_hgb.best_params_), "ROC-AUC CV=4": round(gs_hgb.best_score_, 4)},
]
if HAS_XGB:
    tuning_rows.append(
        {"Modèle": "XGBoost", "Meilleurs paramètres": str(gs_xgb.best_params_), "ROC-AUC CV=4": round(gs_xgb.best_score_, 4)}
    )
tuning_df = pd.DataFrame(tuning_rows).set_index("Modèle")
print("\n" + tuning_df.to_string())
tuning_df.to_csv("sepsis_tuning_results.csv")
print("\n  Tableau sauvegardé : sepsis_tuning_results.csv")

# ── V — Heatmap résultats GridSearchCV ────────────────────────
fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(tuning_df[["ROC-AUC CV=4"]].T, annot=True, fmt=".4f",
            cmap="YlOrRd", linewidths=0.5, ax=ax, vmin=0.5, vmax=1.0,
            annot_kws={"size": 13, "weight": "bold"})
ax.set_title("V — Résultats GridSearchCV (ROC-AUC, CV=4)",
             fontsize=13, fontweight="bold")
ax.set_ylabel(""); ax.set_xlabel("Modèle")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
savefig("V_gridsearch_results.png")

# ── Reconstruction des modèles avec meilleurs params ─────────
models = {}
models["Logistic Regression"]  = gs_lr.best_estimator_
models["Random Forest"]        = gs_rf.best_estimator_
models["HistGradientBoosting"] = gs_hgb.best_estimator_
if HAS_XGB:
    models["XGBoost"]          = gs_xgb.best_estimator_

print("\n  Modèles optimaux prêts pour l'évaluation.")

# ============================================================
# ÉTAPE 9 — VALIDATION CROISÉE (4-Fold, avec modèles optimaux)
# ============================================================
print("\n" + "="*65)
print("ÉTAPE 9 — VALIDATION CROISÉE (4-Fold, modèles optimaux)")
print("="*65)

cv        = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
ap_scorer = make_scorer(average_precision_score, needs_proba=True)
cv_res    = {}

for name, model in models.items():
    print(f"\n ┌─ {name}"); t0 = time.time()
    roc_sc = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=1)
    ap_sc  = cross_val_score(model, X_tr, y_tr, cv=cv, scoring=ap_scorer, n_jobs=1)
    rec_sc = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="recall",  n_jobs=1)
    try:
        oof = cross_val_predict(model, X_tr, y_tr, cv=cv,
                                method="predict_proba", n_jobs=1)[:,1]
        pr, rc, th = precision_recall_curve(y_tr, oof)
        f1v = 2*pr*rc/(pr+rc+1e-8); bt = th[f1v[:-1].argmax()]
        rep = classification_report(y_tr, (oof>=bt).astype(int), output_dict=True)
        r_oof, f1_oof = rep["1"]["recall"], rep["1"]["f1-score"]
    except Exception as e:
        r_oof = f1_oof = float("nan"); bt = 0.5; print(f"  OOF: {e}")
    print(f" │ ROC-AUC : {roc_sc.mean():.4f} ± {roc_sc.std():.4f}")
    print(f" │ PR-AUC  : {ap_sc.mean():.4f} ± {ap_sc.std():.4f}")
    print(f" │ Recall  : {rec_sc.mean():.4f} ± {rec_sc.std():.4f}")
    print(f" │ OOF → Recall={r_oof:.4f}  F1={f1_oof:.4f}  seuil={bt:.3f}")
    print(f" └─ {time.time()-t0:.1f}s")
    cv_res[name] = {"ROC-AUC": roc_sc, "PR-AUC": ap_sc, "Recall": rec_sc,
                    "r_oof": r_oof, "f1_oof": f1_oof}

# ── N — Boxplot ROC-AUC CV ────────────────────────────────────
clrs = ["#3498DB","#E74C3C","#2ECC71","#F39C12","#9B59B6"]
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, metric, title in zip(axes,
    ["ROC-AUC","PR-AUC","Recall"],
    ["N1 — ROC-AUC 4-Fold CV","N2 — PR-AUC 4-Fold CV","N3 — Recall 4-Fold CV"]):
    db = [cv_res[n][metric] for n in models]
    bp = ax.boxplot(db, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", lw=2))
    for p, c in zip(bp["boxes"], clrs): p.set_facecolor(c); p.set_alpha(0.7)
    ax.set_xticklabels(list(models.keys()), rotation=20, ha="right", fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold"); ax.set_ylabel(metric)
    ax.axhline(0.5, color="grey", ls="--", lw=0.8); ax.set_ylim(0.3, 1.05)
plt.suptitle("N — Validation Croisée 4-Fold (modèles optimaux)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
savefig("N_cv_boxplots.png")
# ============================================================
# ÉTAPE 10 — LEARNING CURVES
# ============================================================
print("\n" + "="*65)
print("ÉTAPE 10 — LEARNING CURVES")
print("="*65)

lc_list = [(n, m) for n, m in models.items()
           if n in ["Random Forest","Logistic Regression","XGBoost","HistGradientBoosting"]][:3]
palette = [("#E74C3C","#F1948A"),("#3498DB","#AED6F1"),("#9B59B6","#D7BDE2")]
cv_lc   = StratifiedKFold(n_splits=4, shuffle=True, random_state=SEED)
fig, axes = plt.subplots(1, len(lc_list), figsize=(6*len(lc_list), 6))
if len(lc_list) == 1: axes = [axes]

for ax, (name, model), (c1, c2) in zip(axes, lc_list, palette):
    print(f"  {name} ...", end=" ", flush=True); t0 = time.time()
    sz, ts_, vs_ = learning_curve(model, X_tr, y_tr,
                                   train_sizes=np.linspace(0.1, 1.0, 8),
                                   cv=cv_lc, scoring="roc_auc",
                                   n_jobs=1, shuffle=True, random_state=SEED)
    print(f"{time.time()-t0:.1f}s")
    tm, ts = ts_.mean(1), ts_.std(1); vm, vs = vs_.mean(1), vs_.std(1)
    gap  = tm[-1] - vm[-1]
    diag = " Surapprentissage" if gap > 0.08 else "⚠ Sous-apprentissage" if vm[-1] < 0.70 else "✔ Bonne généralisation"
    ax.plot(sz, tm, "o-", color=c1, lw=2, label="Train")
    ax.fill_between(sz, tm-ts, tm+ts, alpha=0.15, color=c1)
    ax.plot(sz, vm, "s--", color=c2, lw=2, label="Validation (CV)")
    ax.fill_between(sz, vm-vs, vm+vs, alpha=0.15, color=c2)
    ax.set_title(f"{name}\n{diag} (gap={gap:.3f})", fontsize=10, fontweight="bold")
    ax.set_xlabel("Taille du train set"); ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.4, 1.05); ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
plt.suptitle("O — Learning Curves (ROC-AUC, 5-Fold CV)", fontsize=14, fontweight="bold")
plt.tight_layout(); savefig("O_learning_curves.png")

# ============================================================
# ÉTAPE 11 — ÉVALUATION TEST SET
# ============================================================
print("\n" + "="*65)
print("ÉTAPE 11 — ÉVALUATION TEST SET")
print("="*65)
print("  Calibration : IsotonicRegression sur val set")
print("  Seuil       : F1-max sur val set → appliqué au test")

test_res = {}

for name, model in models.items():
    print(f"\n ┌─ {name}"); t0 = time.time()

    if name == "XGBoost":
        model_fit = xgb.XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            min_child_weight=30, subsample=0.8, colsample_bytree=0.7,
            eval_metric="logloss", early_stopping_rounds=30,
            random_state=SEED, n_jobs=4, verbosity=0)
        model_fit.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    else:
        model_fit = model
        model_fit.fit(X_tr, y_tr)

    raw_va = model_fit.predict_proba(X_va)[:, 1]
    iso    = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_va, y_va)
    cal_va = iso.predict(raw_va)
    thr    = best_f1_threshold(y_va, cal_va)

    raw_te = model_fit.predict_proba(X_te)[:, 1]
    cal_te = iso.predict(raw_te)
    y_pred = (cal_te >= thr).astype(int)

    roc   = roc_auc_score(y_te, cal_te)
    ap    = average_precision_score(y_te, cal_te)
    rep   = classification_report(y_te, y_pred, output_dict=True)
    cm    = confusion_matrix(y_te, y_pred)
    mcc   = matthews_corrcoef(y_te, y_pred)
    brier = brier_score_loss(y_te, cal_te)
    tn, fp, fn, tp = cm.ravel()

    print(f" │ Seuil (val, F1-max) : {thr:.4f}")
    print(f" │ ROC-AUC={roc:.4f}  PR-AUC={ap:.4f}  MCC={mcc:.4f}  Brier={brier:.4f}")
    print(f" │ Recall={rep['1']['recall']:.4f}  Precision={rep['1']['precision']:.4f}  F1={rep['1']['f1-score']:.4f}")
    print(f" │ TP={tp:,}  TN={tn:,}  FP={fp:,}  FN={fn:,}")
    print(f" │ Sepsis détectés : {tp:,}/{int(y_te.sum()):,} ({tp/y_te.sum()*100:.1f}%)")
    print(f" └─ {time.time()-t0:.1f}s")

    test_res[name] = {
        "model_fit": model_fit, "iso": iso,
        "y_pred": y_pred, "y_prob": cal_te,
        "roc": roc, "ap": ap, "mcc": mcc, "brier": brier,
        "rep": rep, "cm": cm, "thr": thr,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)
    }

# ── P — Matrices de confusion ─────────────────────────────────
n_m = len(test_res)
fig, axes = plt.subplots(1, n_m, figsize=(5*n_m, 5))
if n_m == 1: axes = [axes]
for ax, (nm, rs) in zip(axes, test_res.items()):
    ConfusionMatrixDisplay(confusion_matrix=rs["cm"],
                           display_labels=["Non-Septique","Septique"]
                           ).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{nm}\n(seuil={rs['thr']:.3f})", fontsize=9, fontweight="bold")
    ax.set_xlabel("Prédit"); ax.set_ylabel("Réel")
plt.suptitle("P — Matrices de Confusion (seuil optimal sur val set)",
             fontsize=14, fontweight="bold")
plt.tight_layout(); savefig("P_confusion_matrices.png")

# ── Q — Courbes ROC ───────────────────────────────────────────
plt.figure(figsize=(10, 7))
for (nm, rs), cl in zip(test_res.items(), clrs):
    fpr_r, tpr_r, _ = roc_curve(y_te, rs["y_prob"])
    plt.plot(fpr_r, tpr_r, color=cl, lw=2.5, label=f"{nm} (AUC={rs['roc']:.3f})")
plt.plot([0,1],[0,1],"k--", lw=1.2, label="Aléatoire (AUC=0.500)")
plt.fill_between([0,1],[0,1], alpha=0.04, color="grey")
plt.xlabel("Taux de Faux Positifs (FPR)", fontsize=12)
plt.ylabel("Taux de Vrais Positifs (Recall)", fontsize=12)
plt.title("Q — Courbes ROC (Test Set)", fontsize=14, fontweight="bold")
plt.legend(fontsize=11, loc="lower right"); plt.grid(True, alpha=0.3)
plt.tight_layout(); savefig("Q_roc_curves.png")

# ── R — Courbes Precision-Recall ──────────────────────────────
plt.figure(figsize=(10, 7))
baseline_pr = y_te.mean()
for (nm, rs), cl in zip(test_res.items(), clrs):
    pr_r, rc_r, _ = precision_recall_curve(y_te, rs["y_prob"])
    plt.plot(rc_r, pr_r, color=cl, lw=2.5, label=f"{nm} (AP={rs['ap']:.3f})")
plt.axhline(baseline_pr, color="grey", ls="--", lw=1.5,
            label=f"Baseline ({baseline_pr:.3f})")
plt.xlabel("Recall", fontsize=12); plt.ylabel("Precision", fontsize=12)
plt.title(f"R — Courbes Precision-Recall (Test Set)\nBaseline = {baseline_pr:.3f}",
          fontsize=13, fontweight="bold")
plt.legend(fontsize=11); plt.grid(True, alpha=0.3)
plt.tight_layout(); savefig("R_pr_curves.png")

# ── S — Courbes de calibration ────────────────────────────────
plt.figure(figsize=(10, 7))
plt.plot([0,1],[0,1],"k--", lw=1.5, label="Calibration parfaite")
for (nm, rs), cl in zip(test_res.items(), clrs):
    try:
        fp_c, mp_c = calibration_curve(y_te, rs["y_prob"], n_bins=15)
        plt.plot(mp_c, fp_c, "o-", color=cl, lw=2, label=nm, markersize=5, alpha=0.85)
    except Exception: pass
plt.xlabel("Probabilité prédite moyenne", fontsize=12)
plt.ylabel("Fraction de positifs réels", fontsize=12)
plt.title("S — Courbes de Calibration (IsotonicRegression sur val)",
          fontsize=13, fontweight="bold")
plt.legend(fontsize=10); plt.grid(True, alpha=0.3)
plt.tight_layout(); savefig("S_calibration_curves.png")

# ============================================================
# ÉTAPE 12 — TABLEAU COMPARATIF FINAL
# ============================================================
print("\n" + "="*65)
print("ÉTAPE 12 — TABLEAU COMPARATIF FINAL")
print("="*65)

rows = []
for nm, rs in test_res.items():
    r = rs["rep"]
    rows.append({
        "Modèle":        nm,
        "Accuracy":      round(r["accuracy"], 4),
        "Precision (1)": round(r["1"]["precision"], 4),
        "Recall (1)":    round(r["1"]["recall"], 4),
        "F1 (1)":        round(r["1"]["f1-score"], 4),
        "MCC":           round(rs["mcc"], 4),
        "ROC-AUC":       round(rs["roc"], 4),
        "Avg Precision": round(rs["ap"], 4),
        "Brier Score":   round(rs["brier"], 4),
        "FN (manqués)":  rs["fn"],
        "FP (alarmes)":  rs["fp"],
        "Seuil":         round(rs["thr"], 4),
    })

summary = pd.DataFrame(rows).set_index("Modèle")
print("\n" + summary.to_string())
summary.to_csv("sepsis_v6_results.csv")
print("\n   Tableau sauvegardé : sepsis_v6_results.csv")

# ── T — Heatmap comparaison finale ───────────────────────────
plot_c = ["Precision (1)","Recall (1)","F1 (1)","MCC","ROC-AUC","Avg Precision"]
fig, ax = plt.subplots(figsize=(16, 6))
sns.heatmap(summary[plot_c].T, annot=True, fmt=".3f", cmap="YlOrRd",
            linewidths=0.5, ax=ax, vmin=0, vmax=1,
            annot_kws={"size":11, "weight":"bold"})
ax.set_title("T — Comparaison des Modèles (Test Set, seuils optimaux)",
             fontsize=14, fontweight="bold")
ax.set_ylabel("Métrique"); ax.set_xlabel("Modèle")
plt.xticks(rotation=20, ha="right")
plt.tight_layout(); savefig("T_model_comparison_heatmap.png")

# ── U — Feature Importance du meilleur modèle arboré ─────────
best_tree = next((n for n in ["LightGBM","XGBoost","Random Forest","HistGradientBoosting"]
                  if n in test_res), None)
if best_tree:
    try:
        src = (test_res[best_tree]["model_fit"]
               if best_tree == "XGBoost" else models[best_tree])
        fi  = pd.Series(src.feature_importances_,
                        index=sel[:len(src.feature_importances_)]
                        ).sort_values(ascending=False)
        plt.figure(figsize=(12, 6))
        is_d   = [any(x in f for x in ["_d1","_d6","_s6","shock","pulse"]) for f in fi.index]
        c_fi   = ["#9B59B6" if d else "#E74C3C" if i<5 else "#AED6F1" for i,d in enumerate(is_d)]
        fi.plot(kind="bar", color=c_fi, edgecolor="white")
        plt.title(f"U — Feature Importance ({best_tree}) — violet=delta",
                  fontsize=13, fontweight="bold")
        plt.xlabel("Feature"); plt.ylabel("Importance (Gini)")
        plt.xticks(rotation=45, ha="right")
        from matplotlib.patches import Patch
        plt.legend(handles=[Patch(facecolor="#9B59B6", label="Delta features"),
                             Patch(facecolor="#E74C3C", label="Top-5"),
                             Patch(facecolor="#AED6F1", label="Autres")])
        plt.tight_layout(); savefig("U_feature_importance.png")
    except Exception as e:
        print(f"  Feature importance : {e}")
