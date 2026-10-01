import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, Lipinski
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

st.set_page_config(page_title="약물 용해도 예측기", layout="wide")

st.markdown("## Solubility Predictor for Poorly Soluble Drugs")
st.markdown("##### Lab. of Drug Delivery, Dongguk University")

st.markdown(
    "<hr style='border: 3px solid #E87722; margin-top: 5px; margin-bottom: 25px;'>",
    unsafe_allow_html=True
)

# ---------- 공통 함수 ----------
def get_features(mol):
    return {
        "MW": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "RotB": Lipinski.NumRotatableBonds(mol),
    }

def calc_esol(mol):
    logp = Descriptors.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    rb = Lipinski.NumRotatableBonds(mol)
    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    heavy_atoms = mol.GetNumHeavyAtoms()
    ap = aromatic_atoms / heavy_atoms if heavy_atoms > 0 else 0
    return 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rb - 0.74 * ap

def solubility_comment(logS):
    if logS >= -2:
        return "높음 (잘 녹음)"
    elif logS >= -4:
        return "보통"
    else:
        return "낮음 (난용성)"

# ============================================================
# 1. 단일 약물 용해도 예측 (ESOL 방정식)
# ============================================================
with st.container(border=True):
    st.subheader("1️⃣ 분자 구조 입력 (단일 약물 용해도 예측)")
    smiles = st.text_input("SMILES를 입력하세요", placeholder="예: CC(=O)OC1=CC=CC=C1C(=O)O")

if smiles:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("올바른 SMILES가 아니에요.")
    else:
        with st.container(border=True):
            st.subheader("예측 결과")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("분자량 (MW)", f"{Descriptors.MolWt(mol):.2f}")
            with col2:
                st.metric("LogP (지용성 지표)", f"{Descriptors.MolLogP(mol):.2f}")
            with col3:
                logS = calc_esol(mol)
                st.metric("예측 logS (수용해도)", f"{logS:.2f}")

            st.caption(f"수용해도 등급: **{solubility_comment(logS)}** (ESOL 방정식 기반 추정치)")

            img = Draw.MolToImage(mol, size=(200, 200))
            st.image(img, caption="분자 구조")

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# 2. 첨가제(폴리머) 효과 예측 — 초안 (예시 데이터 기반)
# ============================================================
with st.container(border=True):
    st.subheader("2️⃣ 첨가제 효과 예측 — 초안")
    st.warning(
        "⚠️ 이 섹션은 랩실 실제 실험 데이터가 아직 들어가지 않은 **구조 초안**입니다. "
        "아래 결과는 코드 구조 확인용 예시 데이터로 학습된 것이며, 실제 과학적 정확도를 갖고 있지 않습니다. "
        "실제 실험 데이터를 넣으면 이 모델이 진짜 예측값과 신뢰할 수 있는 정확도를 보여주게 됩니다."
    )

    colA, colB = st.columns(2)
    with colA:
        drug_smiles = st.text_input("약물 SMILES", key="drug_smiles2",
                                     placeholder="예: CC(=O)OC1=CC=CC=C1C(=O)O")
    with colB:
        add_smiles = st.text_input("첨가제 SMILES", key="add_smiles2",
                                    placeholder="예: 폴리머 단량체 SMILES")

    colC, colD = st.columns(2)
    with colC:
        weight_frac = st.slider("첨가제 비율 (W, 전체 중 질량비)", 0.0, 1.0, 0.3, 0.05)
    with colD:
        temperature = st.slider("온도 (K)", 280, 380, 310, 1)

    predict_btn = st.button("첨가제 효과 예측하기")

# ---- 예시(placeholder) 학습 데이터 ----
# TODO: 랩실 실제 실험 데이터를 받으면 이 리스트를 실제 데이터로 교체하세요.
# 각 항목: 약물 SMILES, 첨가제 SMILES, W(첨가제 비율), T(온도 K), logS(실험으로 측정된 값)
EXAMPLE_DATA = [
    {"drug": "CC(=O)OC1=CC=CC=C1C(=O)O", "additive": "CC(=O)N(C)C", "W": 0.1, "T": 298, "logS": -2.1},
    {"drug": "CC(=O)OC1=CC=CC=C1C(=O)O", "additive": "CC(=O)N(C)C", "W": 0.3, "T": 298, "logS": -1.6},
    {"drug": "CC(=O)OC1=CC=CC=C1C(=O)O", "additive": "CC(=O)N(C)C", "W": 0.5, "T": 313, "logS": -1.1},
    {"drug": "CC(=O)NC1=CC=C(O)C=C1", "additive": "CC(=O)N(C)C", "W": 0.2, "T": 298, "logS": -0.9},
    {"drug": "CC(=O)NC1=CC=C(O)C=C1", "additive": "CC(=O)N(C)C", "W": 0.4, "T": 313, "logS": -0.4},
    {"drug": "CC(=O)NC1=CC=C(O)C=C1", "additive": "CC(=O)N(C)C", "W": 0.6, "T": 313, "logS": 0.1},
]

@st.cache_resource
def train_example_model():
    rows = []
    for d in EXAMPLE_DATA:
        drug_mol = Chem.MolFromSmiles(d["drug"])
        add_mol = Chem.MolFromSmiles(d["additive"])
        if drug_mol is None or add_mol is None:
            continue
        row = {f"drug_{k}": v for k, v in get_features(drug_mol).items()}
        row.update({f"add_{k}": v for k, v in get_features(add_mol).items()})
        row["W"] = d["W"]
        row["T"] = d["T"]
        row["logS"] = d["logS"]
        rows.append(row)

    df = pd.DataFrame(rows)
    X = df.drop(columns=["logS"])
    y = df["logS"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    pred = model.predict(X)
    r2 = r2_score(y, pred)
    mae = mean_absolute_error(y, pred)
    return model, list(X.columns), r2, mae

if predict_btn:
    if not drug_smiles or not add_smiles:
        st.error("약물과 첨가제 SMILES를 모두 입력해주세요.")
    else:
        drug_mol = Chem.MolFromSmiles(drug_smiles)
        add_mol = Chem.MolFromSmiles(add_smiles)
        if drug_mol is None or add_mol is None:
            st.error("올바른 SMILES가 아니에요.")
        else:
            model, feature_names, r2, mae = train_example_model()

            row = {f"drug_{k}": v for k, v in get_features(drug_mol).items()}
            row.update({f"add_{k}": v for k, v in get_features(add_mol).items()})
            row["W"] = weight_frac
            row["T"] = temperature

            X_new = pd.DataFrame([row])[feature_names]
            pred_logS = model.predict(X_new)[0]

            row_base = dict(row)
            row_base["W"] = 0.0
            X_base = pd.DataFrame([row_base])[feature_names]
            base_logS = model.predict(X_base)[0]
            delta = pred_logS - base_logS

            with st.container(border=True):
                st.subheader("예측 결과 (초안 모델)")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("첨가제 적용 전 logS", f"{base_logS:.2f}")
                with col2:
                    st.metric("첨가제 적용 후 logS", f"{pred_logS:.2f}")
                with col3:
                    direction = "증가 ⬆️" if delta > 0 else ("감소 ⬇️" if delta < 0 else "변화 없음")
                    st.metric("변화량 (Δ logS)", f"{delta:+.2f}", delta=direction)

                st.caption(
                    f"모델 참고 정확도 (예시 데이터 기준, 신뢰도 낮음): R² = {r2:.2f}, MAE = {mae:.2f}. "
                    "실제 실험 데이터로 재학습하면 이 수치가 신뢰할 수 있는 값으로 바뀝니다."
                )

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    "<div style='background-color:#E87722; padding:10px; border-radius:5px; color:white; font-weight:bold; text-align:center;'>Supervised by Prof. Sung-Giu Jin</div>",
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    if os.path.exists("logo/Lab_logo.png"):
        st.image("logo/Lab_logo.png", width=220)
with col3:
    if os.path.exists("logo/dongguk_logo.png"):
        st.image("logo/dongguk_logo.png", width=220)
