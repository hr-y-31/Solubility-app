import os

import pandas as pd
import py3Dmol
import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw, Lipinski
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="약물 용해도 예측기", layout="wide")

st.markdown("## Solubility Predictor for Poorly Soluble Drugs")
st.markdown("##### Lab. of Drug Delivery, Dongguk University")

st.markdown(
    "<hr style='border: 3px solid #E87722; margin-top: 5px; margin-bottom: 25px;'>",
    unsafe_allow_html=True,
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
    """ESOL (Delaney) 방정식 기반 logS 추정."""
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


def make_3d_html(mol):
    """3D 구조 HTML 생성. 실패하면 None 반환."""
    mol_3d = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol_3d, randomSeed=42) != 0:
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol_3d)
    except Exception:
        pass  # 최적화 실패해도 임베딩된 구조는 사용
    viewer = py3Dmol.view(width=350, height=350)
    viewer.addModel(Chem.MolToMolBlock(mol_3d), "mol")
    viewer.setStyle({"stick": {}})
    viewer.zoomTo()
    return viewer._make_html()


# ============================================================
# 1. 단일 약물 용해도 예측 (ESOL 방정식)
# ============================================================
with st.container(border=True):
    st.subheader("1️⃣ 분자 구조 입력 (단일 약물 용해도 예측)")
    smiles = st.text_input(
        "SMILES를 입력하세요", placeholder="예: CC(=O)OC1=CC=CC=C1C(=O)O"
    ).strip()

if smiles:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("올바른 SMILES가 아니에요.")
    else:
        with st.container(border=True):
            st.subheader("예측 결과")
            logS = calc_esol(mol)

            col1, col2, col3 = st.columns(3)
            col1.metric("분자량 (MW)", f"{Descriptors.MolWt(mol):.2f}")
            col2.metric("LogP (지용성 지표)", f"{Descriptors.MolLogP(mol):.2f}")
            col3.metric("예측 logS (수용해도)", f"{logS:.2f}")

            st.caption(
                f"수용해도 등급: **{solubility_comment(logS)}** (ESOL 방정식 기반 추정치)"
            )

            col_img, col_3d = st.columns(2)
            with col_img:
                st.image(Draw.MolToImage(mol, size=(250, 250)), caption="2D 구조")
            with col_3d:
                st.caption("3D 구조 (마우스로 드래그해서 회전 가능)")
                try:
                    html = make_3d_html(mol)
                    if html is None:
                        st.warning("이 분자는 3D 구조를 생성하지 못했어요.")
                    else:
                        components.html(html, height=370, width=370)
                except Exception as e:
                    st.error(f"3D 구조 생성 실패: {e}")

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
        drug_smiles = st.text_input(
            "약물 SMILES", key="drug_smiles2",
            placeholder="예: CC(=O)OC1=CC=CC=C1C(=O)O",
        ).strip()
    with colB:
        add_smiles = st.text_input(
            "첨가제 SMILES", key="add_smiles2",
            placeholder="예: 폴리머 단량체 SMILES",
        ).strip()

    colC, colD = st.columns(2)
    with colC:
        weight_frac = st.slider("첨가제 비율 (W, 전체 중 질량비)", 0.0, 1.0, 0.3, 0.05)
    with colD:
        temperature = st.slider("온도 (K)", 280, 380, 310, 1)

    predict_btn = st.button("첨가제 효과 예측하기")

# ---- 예시(placeholder) 학습 데이터 ----
# TODO: 랩실 실제 실험 데이터를 받으면 이 리스트를 실제 데이터로 교체하세요.
# 각 항목: 약물 SMILES, 첨가제 SMILES, W(첨가제 비율), T(온도 K), logS(실험 측정값)
EXAMPLE_DATA = [
    {"drug": "CC(=O)OC1=CC=CC=C1C(=O)O", "additive": "CC(=O)N(C)C", "W": 0.1, "T": 298, "logS": -2.1},
    {"drug": "CC(=O)OC1=CC=CC=C1C(=O)O", "additive": "CC(=O)N(C)C", "W": 0.3, "T": 298, "logS": -1.6},
    {"drug": "CC(=O)OC1=CC=CC=C1C(=O)O", "additive": "CC(=O)N(C)C", "W": 0.5, "T": 313, "logS": -1.1},
    {"drug": "CC(=O)NC1=CC=C(O)C=C1", "additive": "CC(=O)N(C)C", "W": 0.2, "T": 298, "logS": -0.9},
    {"drug": "CC(=O)NC1=CC=C(O)C=C1", "additive": "CC(=O)N(C)C", "W": 0.4, "T": 313, "logS": -0.4},
    {"drug": "CC(=O)NC1=CC=C(O)C=C1", "additive": "CC(=O)N(C)C", "W": 0.6, "T": 313, "logS": 0.1},
]


def build_row(drug_mol, add_mol, W, T):
    row = {f"drug_{k}": v for k, v in get_features(drug_mol).items()}
    row.update({f"add_{k}": v for k, v in get_features(add_mol).items()})
    row["W"] = W
    row["T"] = T
    return row


@st.cache_resource
def train_example_model():
    rows = []
    for d in EXAMPLE_DATA:
        drug_mol = Chem.MolFromSmiles(d["drug"])
        add_mol = Chem.MolFromSmiles(d["additive"])
        if drug_mol is None or add_mol is None:
            continue
        row = build_row(drug_mol, add_mol, d["W"], d["T"])
        row["logS"] = d["logS"]
        rows.append(row)

    df = pd.DataFrame(rows)
    X = df.drop(columns=["logS"])
    y = df["logS"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    pred = model.predict(X)
    return model, list(X.columns), r2_score(y, pred), mean_absolute_error(y, pred)


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

            row = build_row(drug_mol, add_mol, weight_frac, temperature)
            pred_logS = model.predict(pd.DataFrame([row])[feature_names])[0]

            row_base = dict(row, W=0.0)
            base_logS = model.predict(pd.DataFrame([row_base])[feature_names])[0]
            delta = pred_logS - base_logS

            with st.container(border=True):
                st.subheader("예측 결과 (초안 모델)")
                col1, col2, col3 = st.columns(3)
                col1.metric("첨가제 적용 전 logS", f"{base_logS:.2f}")
                col2.metric("첨가제 적용 후 logS", f"{pred_logS:.2f}")
                col3.metric("변화량 (Δ logS)", f"{delta:+.2f}", delta=f"{delta:+.2f}")

                st.caption(
                    f"모델 참고 정확도 (학습 데이터 기준, 신뢰도 낮음): R² = {r2:.2f}, MAE = {mae:.2f}. "
                    "예시 데이터에는 W=0(첨가제 없음) 샘플이 없어 '적용 전' 값은 외삽 추정치입니다. "
                    "실제 실험 데이터로 재학습하면 신뢰할 수 있는 값으로 바뀝니다."
                )

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    "<div style='background-color:#E87722; padding:10px; border-radius:5px; "
    "color:white; font-weight:bold; text-align:center;'>Supervised by Prof. Sung-Giu Jin</div>",
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

LAB_LOGO = "logo/Lab.logo.png"
DONGGUK_LOGO = "logo/dongguk.logo.png"  # 실제 파일명에 맞게 확인하세요

col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    if os.path.exists(Lab.lOGO):
        st.image(Lab.lOGO, width=220)
with col3:
    if os.path.exists(dongguk_lOGO):
        st.image(dongguk_lOGO, width=320)
