"""Démo Streamlit de scoring du churn bancaire.

Application autonome : elle charge directement le pipeline entraîné (pas besoin que
l'API tourne), ce qui permet un déploiement en un clic sur Streamlit Community Cloud.

Lancement local :
    streamlit run app/ui/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Rendre le package `churn` importable (src/ layout).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from churn import config  # noqa: E402
from churn import predict as predict_mod  # noqa: E402

st.set_page_config(page_title="Churn Bank — Scoring", page_icon="🏦", layout="wide")


@st.cache_data
def load_metrics() -> dict:
    if config.METRICS_PATH.exists():
        return json.loads(config.METRICS_PATH.read_text())
    return {}


@st.cache_data
def load_importance() -> dict:
    if config.FEATURE_IMPORTANCE_PATH.exists():
        return json.loads(config.FEATURE_IMPORTANCE_PATH.read_text())
    return {}


@st.cache_resource
def get_model():
    return predict_mod.load_model()


metrics = load_metrics()
threshold = predict_mod.load_threshold()

# ---- En-tête -----------------------------------------------------------------
st.title("🏦 Churn Bank — Prédiction de l'attrition client")
st.markdown(
    "Estime la **probabilité qu'un client quitte la banque** pour cibler les actions de "
    "rétention. Acquérir un nouveau client coûte plusieurs fois plus cher que d'en "
    "retenir un : mieux vaut agir avant le départ."
)

# ---- Barre latérale : performance du modèle ----------------------------------
with st.sidebar:
    st.header(" Performance du modèle")
    test = metrics.get("test", {})
    if test:
        st.metric("ROC-AUC", f"{test.get('roc_auc', 0):.3f}")
        st.metric("F1-score", f"{test.get('f1', 0):.3f}")
        st.metric("Précision", f"{test.get('precision', 0):.3f}")
        st.metric("Rappel", f"{test.get('recall', 0):.3f}")
        st.caption(
            f"Modèle : **{metrics.get('model', 'n/a')}** · seuil de décision "
            f"{threshold:.2f} · évalué sur {metrics.get('n_test', 0):,} clients held-out."
        )
    else:
        st.info("Aucune métrique trouvée. Lance `python -m churn.train`.")

    importance = load_importance()
    if importance:
        st.subheader("Top facteurs de churn")
        top = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:8])
        st.bar_chart(pd.Series(top), horizontal=True)

tab_single, tab_batch = st.tabs([" Scorer un client", "📂 Scorer un fichier CSV"])

# ---- Onglet 1 : client unique ------------------------------------------------
with tab_single:
    st.subheader("Profil du client")
    c1, c2, c3 = st.columns(3)
    with c1:
        credit_score = st.slider("Score de crédit", 350, 850, 650)
        age = st.slider("Âge", 18, 92, 40)
        tenure = st.slider("Ancienneté (années)", 0, 10, 5)
    with c2:
        balance = st.number_input("Solde du compte (€)", 0.0, 260000.0, 60000.0, step=1000.0)
        salary = st.number_input("Salaire estimé (€)", 0.0, 200000.0, 100000.0, step=1000.0)
        num_products = st.selectbox("Nombre de produits", [1, 2, 3, 4], index=0)
    with c3:
        geography = st.selectbox("Pays", ["France", "Germany", "Spain"])
        gender = st.selectbox("Genre", ["Female", "Male"])
        has_card = st.checkbox("Possède une carte de crédit", value=True)
        is_active = st.checkbox("Membre actif", value=True)

    if st.button("Prédire le risque de départ", type="primary"):
        customer = pd.DataFrame(
            [
                {
                    "CreditScore": credit_score,
                    "Geography": geography,
                    "Gender": gender,
                    "Age": age,
                    "Tenure": tenure,
                    "Balance": balance,
                    "NumOfProducts": num_products,
                    "HasCrCard": int(has_card),
                    "IsActiveMember": int(is_active),
                    "EstimatedSalary": salary,
                }
            ]
        )
        proba = predict_mod.predict_proba(customer)[0]
        is_churn = proba >= threshold

        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.metric("Probabilité de churn", f"{proba:.1%}")
        with col_b:
            st.progress(min(proba, 1.0))
            if is_churn:
                st.error(
                    f"⚠️ Client à risque (≥ seuil {threshold:.0%}) — "
                    "action de rétention conseillée."
                )
            else:
                st.success(f"✅ Client peu à risque (< seuil {threshold:.0%}).")

        # Petits signaux explicatifs (alignés sur l'EDA).
        signals = []
        if num_products >= 3:
            signals.append("détient 3+ produits (fort signal de départ)")
        if not is_active:
            signals.append("membre inactif")
        if balance == 0:
            signals.append("solde nul")
        if age >= 50:
            signals.append("client plus âgé")
        if signals:
            st.caption("Facteurs de risque détectés : " + ", ".join(signals) + ".")

# ---- Onglet 2 : batch CSV ----------------------------------------------------
with tab_batch:
    st.subheader("Scorer un portefeuille de clients")
    st.caption("Le CSV doit contenir les colonnes : " + ", ".join(config.INPUT_FEATURES))
    uploaded = st.file_uploader("Fichier CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        missing = [c for c in config.INPUT_FEATURES if c not in df.columns]
        if missing:
            st.error(f"Colonnes manquantes : {missing}")
        else:
            scored = predict_mod.predict(df)
            out = df.copy()
            out["churn_probability"] = scored["probability"].round(4)
            out["churn_prediction"] = scored["churn"]
            st.success(
                f"{int(scored['churn'].sum())} clients à risque sur {len(out)} "
                f"({scored['churn'].mean():.1%})."
            )
            st.dataframe(out, use_container_width=True)
            st.download_button(
                "⬇️ Télécharger les scores",
                out.to_csv(index=False).encode("utf-8"),
                "churn_scores.csv",
                "text/csv",
            )
