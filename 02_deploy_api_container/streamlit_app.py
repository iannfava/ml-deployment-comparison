#streamlit_app.py

import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000") + "/predict"

st.set_page_config(page_title="Shopping Preference", layout="wide")
st.title("Shopping Preference")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Perfil")
    age = st.number_input("Idade", 18, 79, 45)
    gender_label = st.selectbox("Gênero", ["Feminino", "Masculino", "Outro"])
    gender = {"Feminino": "Female", "Masculino": "Male", "Outro": "Other"}[gender_label]
    daily_internet_hours = st.slider("Horas de internet por dia", 1.0, 12.0, 6.0)
    monthly_income = st.number_input("Renda mensal", 1200, 25000, 5000)
    smartphone_usage_years = st.number_input("Anos usando smartphone", 1, 15, 5)
    social_media_hours = st.slider("Horas em redes sociais", 0.0, 6.0, 2.5)
    online_payment_trust_score = st.slider("Confiança em pagamento online", 1, 10, 5)
    tech_savvy_score = st.slider("Afinidade com tecnologia", 1, 10, 5)

with col2:
    st.subheader("Consumo")
    monthly_online_orders = st.number_input("Pedidos online por mês", 0, 50, 5)
    monthly_store_visits = st.number_input("Visitas à loja por mês", 0, 20, 5)
    avg_online_spend = st.number_input("Gasto médio online", 523, 149997, 5000)
    avg_store_spend = st.number_input("Gasto médio em loja", 542, 149973, 5000)
    discount_sensitivity = st.slider("Sensibilidade a desconto", 1, 10, 5)
    return_frequency = st.number_input("Frequência de devolução", 0, 10, 1)
    avg_delivery_days = st.number_input("Dias médios de entrega", 1, 8, 3)
    delivery_fee_sensitivity = st.slider("Sensibilidade a frete", 1, 10, 5)

with col3:
    st.subheader("Preferências")
    free_return_importance = st.slider("Importância de devolução grátis", 1, 10, 5)
    product_availability_online = st.slider("Disponibilidade online", 1, 10, 5)
    impulse_buying_score = st.slider("Compra por impulso", 1, 10, 5)
    need_touch_feel_score = st.slider("Necessidade de tocar produto", 1, 10, 5)
    brand_loyalty_score = st.slider("Fidelidade à marca", 1, 10, 5)
    environmental_awareness = st.slider("Consciência ambiental", 1, 10, 5)
    time_pressure_level = st.slider("Pressão de tempo", 1, 10, 5)
    city_tier = st.selectbox("Nível da cidade", ["Tier 1", "Tier 2", "Tier 3"])

if st.button("Realizar previsão", type="primary"):
    payload = {
        "age": age,
        "monthly_income": monthly_income,
        "daily_internet_hours": daily_internet_hours,
        "smartphone_usage_years": smartphone_usage_years,
        "social_media_hours": social_media_hours,
        "online_payment_trust_score": online_payment_trust_score,
        "tech_savvy_score": tech_savvy_score,
        "monthly_online_orders": monthly_online_orders,
        "monthly_store_visits": monthly_store_visits,
        "avg_online_spend": avg_online_spend,
        "avg_store_spend": avg_store_spend,
        "discount_sensitivity": discount_sensitivity,
        "return_frequency": return_frequency,
        "avg_delivery_days": avg_delivery_days,
        "delivery_fee_sensitivity": delivery_fee_sensitivity,
        "free_return_importance": free_return_importance,
        "product_availability_online": product_availability_online,
        "impulse_buying_score": impulse_buying_score,
        "need_touch_feel_score": need_touch_feel_score,
        "brand_loyalty_score": brand_loyalty_score,
        "environmental_awareness": environmental_awareness,
        "time_pressure_level": time_pressure_level,
        "gender": gender,
        "city_tier": city_tier,
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()
        resultado = response.json()

        if resultado["label"] == "Online":
            st.success("Cliente com perfil de compra Online")
        else:
            st.info("Cliente com perfil de compra em Loja física")

        st.metric("Probabilidade de compra online", f"{resultado['probability_online']:.2%}")
        st.json(resultado)

    except requests.exceptions.ConnectionError:
        st.error("Não foi possível conectar na API. Verifique se o FastAPI está rodando.")
    except requests.exceptions.RequestException as erro:
        st.error(f"Erro ao consultar a API: {erro}")