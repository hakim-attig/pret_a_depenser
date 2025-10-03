import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Dashboard Scoring Crédit",
    page_icon="💳",
    layout="wide"
)

API_URL = "http://localhost:8000"

# Vérifier connexion API
try:
    response = requests.get(f"{API_URL}/status", timeout=2)
    api_status = response.json()
    api_ok = api_status['status'] == 'operational'
except:
    api_ok = False

# En-tête
st.title("💳 Dashboard Scoring Crédit - Prêt à Dépenser")

if not api_ok:
    st.error("⚠️ API non accessible")
    st.stop()

# Informations modèle
try:
    model_info = requests.get(f"{API_URL}/model/info").json()
    st.success(f"✓ Modèle: {model_info['model_type'].upper()} | AUC: {model_info['auc_score']:.4f} | Coût: {model_info['optimal_cost']:,}€ | Seuil: {model_info['optimal_threshold']:.1%}")
except:
    st.warning("Impossible de charger les infos du modèle")

# SIDEBAR - Saisie SK_ID_CURR
st.sidebar.header("🔍 Recherche Client")

selected_client_id = st.sidebar.number_input(
    "SK_ID_CURR",
    min_value=100000,
    max_value=500000,
    value=396899,
    step=1,
    help="Exemples: 396899, 322041, 220127, 251531"
)

st.sidebar.markdown("**Exemples de clients :**")
st.sidebar.markdown("- 396899 (faible risque)")
st.sidebar.markdown("- 322041 (faible risque)")
st.sidebar.markdown("- 345558 (risque élevé)")

# Bouton analyser
if st.sidebar.button("📊 Analyser", type="primary", use_container_width=True):
    with st.spinner("Analyse en cours..."):
        try:
            # Prédiction
            response = requests.post(
                f"{API_URL}/predict",
                json={"client_id": selected_client_id},
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                
                st.header(f"Client SK_ID_CURR: {result['client_id']}")
                
                # Métriques
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    risk = result['risk_score']
                    st.metric("Risque de Défaut", f"{risk:.2%}")
                
                with col2:
                    decision = result['decision']
                    color = "🟢" if decision == "ACCORD" else "🔴"
                    st.metric("Décision", f"{color} {decision}")
                
                with col3:
                    st.metric("Seuil du Modèle", f"{result['threshold']:.1%}")
                
                with col4:
                    real = result['real_label']
                    st.metric("Réalité", real)
                
                if not result['prediction_correct']:
                    st.warning("⚠️ Prédiction incorrecte")
                else:
                    st.success("✓ Prédiction correcte")
                
                # Jauge
                st.subheader("📊 Niveau de Risque")
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=risk * 100,
                    title={'text': "Risque de Défaut (%)"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkred" if risk >= result['threshold'] else "darkgreen"},
                        'steps': [
                            {'range': [0, result['threshold']*100], 'color': "lightgreen"},
                            {'range': [result['threshold']*100, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': result['threshold'] * 100
                        }
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                # SHAP
                st.subheader("🧠 Explication (Top 10 Features)")
                explain_resp = requests.get(f"{API_URL}/explain/{selected_client_id}", timeout=10)
                
                if explain_resp.status_code == 200:
                    explanation = explain_resp.json()
                    shap_df = pd.DataFrame(explanation["top_features"])
                    
                    fig_shap = px.bar(
                        shap_df.sort_values("impact", key=abs),
                        x="impact",
                        y="feature",
                        orientation='h',
                        color="direction",
                        color_discrete_map={"AUGMENTE LE RISQUE": "red", "DIMINUE LE RISQUE": "green"},
                        labels={"impact": "Impact SHAP", "feature": "Variable"},
                        title="Facteurs influençant la décision"
                    )
                    fig_shap.update_layout(height=500, yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig_shap, use_container_width=True)
                    
                    st.info("💡 " + explanation["interpretation"])
                else:
                    st.warning("Explication SHAP non disponible")
                
                # Interprétation
                st.subheader("💡 Interprétation")
                if decision == "ACCORD":
                    st.success(f"""
                    **Crédit Accordé**
                    - Risque: {risk:.2%} < Seuil: {result['threshold']:.1%}
                    - Profil acceptable pour l'octroi du crédit
                    """)
                else:
                    st.error(f"""
                    **Crédit Refusé**
                    - Risque: {risk:.2%} > Seuil: {result['threshold']:.1%}
                    - Profil à risque trop élevé
                    """)
                    
            elif response.status_code == 404:
                st.error(f"Client SK_ID_CURR {selected_client_id} non trouvé dans la base")
            else:
                st.error(f"Erreur API : {response.status_code}")
                
        except Exception as e:
            st.error(f"Erreur : {str(e)}")

st.markdown("---")
st.markdown("**Projet 7 - OpenClassrooms** | Modèle XGBoost Champion")