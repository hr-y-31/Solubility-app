import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, Lipinski
from openai import OpenAI
import os

st.set_page_config(page_title="난용성 약물 용해도 예측기", layout="wide")

st.markdown("## Solubility Predictor for Poorly Soluble Drugs")
st.markdown("##### Lab. of Drug Delivery, Dongguk University")

st.markdown(
    "<hr style='border: 3px solid #E87722; margin-top: 5px; margin-bottom: 25px;'>",
    unsafe_allow_html=True
)

with st.container(border=True):
    st.subheader("분자 구조 입력")
    smiles = st.text_input("SMILES를 입력하세요", placeholder="예: CC(=O)OC1=CC=CC=C1C(=O)O")

def calc_esol(mol):
    logp = Descriptors.MolLogP(mol)
    mw = Descriptors.MolWt(mol)
    rb = Lipinski.NumRotatableBonds(mol)
    aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
    heavy_atoms = mol.GetNumHeavyAtoms()
    ap = aromatic_atoms / heavy_atoms if heavy_atoms > 0 else 0
    logS = 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rb - 0.74 * ap
    return logS

def solubility_comment(logS):
    if logS >= -2:
        return "높음 (잘 녹음)"
    elif logS >= -4:
        return "보통"
    else:
        return "낮음 (난용성)"

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

            img = Draw.MolToImage(mol, size=(300, 300))
            st.image(img, caption="분자 구조")

        with st.container(border=True):
            st.subheader("AI 용해도 개선 제안")
            ask_ai = st.button("DeepSeek-R1에게 개선 방법 물어보기")
            if ask_ai:
                with st.spinner("AI가 생각 중이에요..."):
                    prompt = (
                        "다음은 난용성 약물 분자 정보입니다.\n"
                        f"SMILES: {smiles}\n"
                        f"분자량: {Descriptors.MolWt(mol):.2f}\n"
                        f"LogP: {Descriptors.MolLogP(mol):.2f}\n"
                        f"예측 logS (ESOL 방정식): {logS:.2f}\n\n"
                        "이 분자의 수용해도를 개선할 수 있는 화학적 방법을 3가지 제안해주세요."
                    )
                    client = OpenAI(api_key=st.secrets["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
                    response = client.chat.completions.create(
                        model="deepseek-reasoner",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    result = response.choices[0].message.content
                    st.write(result)

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    "<div style='background-color:#E87722; padding:10px; border-radius:5px; color:white; font-weight:bold; text-align:center;'>Supervised by Prof. Sung-Giu Jin</div>",
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    if os.path.exists("logo/lab_logo.png"):
        st.image("logo/lab_logo.png", width=120)
with col3:
    if os.path.exists("logo/dongguk_logo.png"):
        st.image("logo/dongguk_logo.png", width=120)
