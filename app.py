import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
import requests
import os

st.set_page_config(page_title="난용성 약물 용해도 예측기", layout="wide")

# ── 상단 제목 영역 ──────────────────────────
st.markdown("## Solubility Predictor for Poorly Soluble Drugs")
st.markdown("##### Lab. of Drug Delivery, Dongguk University")

st.markdown(
    "<hr style='border: 3px solid #E87722; margin-top: 5px; margin-bottom: 25px;'>",
    unsafe_allow_html=True
)

# ── 입력 영역 ──────────────────────────────────
with st.container(border=True):
    st.subheader("🧪 분자 구조 입력")
    smiles = st.text_input("SMILES를 입력하세요", placeholder="예: CC(=O)OC1=CC=CC=C1C(=O)O")

# ── 결과 영역 ──────────────────────────────────
if smiles:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("올바른 SMILES가 아니에요.")
    else:
        with st.container(border=True):
            st.subheader("📊 예측 결과")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("분자량 (MW)", f"{Descriptors.MolWt(mol):.2f}")
            with col2:
                st.metric("LogP (지용성 지표)", f"{Descriptors.MolLogP(mol):.2f}")

            img = Draw.MolToImage(mol, size=(300, 300))
            st.image(img, caption="분자 구조")

        # ── DeepSeek-R1 개선 제안 ──────────────────
        with st.container(border=True):
            st.subheader("🤖 AI 용해도 개선 제안")
            if st.button("DeepSeek-R1에게 개선 방법 물어보기"):
                with st.spinner("AI가 생각 중이에요..."):
                    prompt = f"""
다음은 난용성 약물 분자 정보입니다.
SMILES: {smiles}
분자량: {Descriptors.MolWt(mol):.2f}
LogP: {Descriptors.MolLogP(mol):.2f}

이 분자의 수용해도를 개선할 수 있는 화학적 방법을 3가지 제안해주세요.
"""
        try:
                from openai import OpenAI
    client = OpenAI(api_key=st.secrets["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[{"role": "user", "content": prompt}]
    )
    result = response.choices[0].message.content
    st.write(result)
except Exception as e:
    st.error(f"DeepSeek 연결 실패: {e}")

# ── 하단 푸터 ──────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(
    "<div style='background-color:#E87722; padding:10px; border-radius:5px; "
    "color:white; font-weight:bold; text-align:center;'>Supervised by Prof. Sung-Giu Jin</div>",
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 3, 1])
with col1:
    if os.path.exists("logo/Lab.logo.png"):
        st.image("logo/lab.logo.png", width=200)
with col3:
    if os.path.exists("logo/dongguk.logo.png"):
        st.image("logo/dongguk.logo.png", width=300)
