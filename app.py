import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="Modelo Preditivo", layout="centered")

st.title("🏠 Previsão de Imóveis")
st.write("Faça upload de um arquivo CSV com as 30 features para obter as previsões.")

# ------------ Carregar modelo ------------
@st.cache_resource
def load_model():
    return joblib.load("model.joblib")

model = load_model()

# ------------ Upload do CSV ------------
uploaded_file = st.file_uploader("Envie o CSV aqui", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("Primeiras linhas do arquivo enviado:")
        st.write(df.head())

        # Garantir que valores infinitos não quebrem o modelo
        df = df.replace([np.inf, -np.inf], np.nan)

        # Previsão
        with st.spinner("Gerando previsões..."):
            preds = model.predict(df)

        df_resultado = df.copy()
        df_resultado["predicao"] = preds

        st.success("Previsões concluídas!")
        st.subheader("Resultados:")
        st.write(df_resultado.head())

        # Download do resultado
        csv_download = df_resultado.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Baixar CSV com previsões",
            data=csv_download,
            file_name="predicoes.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")

else:
    st.info("Envie um arquivo CSV para começar.")
