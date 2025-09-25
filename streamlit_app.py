import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# Configuration avec thème coloré
st.set_page_config(
    page_title="Prêt à Dépenser",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-card {
        background: linear-gradient(45deg, #56ab2f 0%, #a8e6cf 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
    .danger-card {
        background: linear-gradient(45deg, #ff6b6b 0%, #feca57 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

API_URL = "http://127.0.0.1:8000"

# En-tête coloré
st.markdown("""
<div class="main-header">
    <h1>🏦 Prêt à Dépenser - Système d'Évaluation Crédit</h1>
    <p>Plateforme intelligente avec modèle LightGBM (Seuil: 9%)</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("### 🔧 Configuration")

# Test API
try:
    response = requests.get(f"{API_URL}/status", timeout=2)
    api_ok = response.status_code == 200
    if api_ok:
        st.sidebar.markdown("🟢 **API connectée**")
        # Afficher infos modèle
        try:
            model_info = requests.get(f"{API_URL}/model/info").json()
            st.sidebar.write(f"Modèle: {model_info.get('model_type', 'N/A')}")
            st.sidebar.write(f"Seuil: {model_info.get('threshold', 0):.1%}")
        except:
            pass
    else:
        st.sidebar.markdown("🔴 **API erreur**")
except:
    api_ok = False
    st.sidebar.markdown("🔴 **API non accessible**")

# Menu
mode = st.sidebar.radio(
    "### 📋 Mode d'analyse",
    ["🔍 Client Existant", "➕ Nouveau Client"],
    format_func=lambda x: x
)

def create_explanation_charts(client_data, prediction_result):
    """Créer des graphiques d'explication pour le banquier"""
    
    prob = prediction_result["prediction"]["probability"]
    age_years = abs(client_data["DAYS_BIRTH"]) // 365
    ratio_credit = client_data["AMT_CREDIT"] / client_data["AMT_GOODS_PRICE"]
    
    # Graphique en secteurs des facteurs de risque
    factors = {
        "Scores Externes": 30 if (client_data.get("EXT_SOURCE_2", 0.5) + client_data.get("EXT_SOURCE_3", 0.5)) > 1.0 else 15,
        "Ratio de Financement": 25 if ratio_credit > 0.9 else 10,
        "Profil Âge": 20 if age_years < 25 or age_years > 65 else 5,
        "Montant du Crédit": 15 if client_data["AMT_CREDIT"] > 500000 else 8,
        "Historique Paiements": 10 if client_data.get("INST_PAYMENT_PERC_mean", 0.9) < 0.8 else 2
    }
    
    colors = ['#ff6b6b', '#feca57', '#48dbfb', '#ff9ff3', '#54a0ff']
    fig_factors = px.pie(
        values=list(factors.values()),
        names=list(factors.keys()),
        title="🎯 Répartition des Facteurs de Risque",
        color_discrete_sequence=colors
    )
    fig_factors.update_traces(textposition='inside', textinfo='percent+label')
    
    # Graphique comparatif
    comparison_data = pd.DataFrame({
        'Critère': ['Score Externe Moyen', 'Ratio Financement', 'Âge', 'Capacité Remboursement'],
        'Client': [
            (client_data.get("EXT_SOURCE_2", 0.5) + client_data.get("EXT_SOURCE_3", 0.5)) / 2,
            min(ratio_credit, 1.0),
            min(age_years / 70, 1.0),
            min(client_data["AMT_ANNUITY"] / (client_data["AMT_CREDIT"] * 0.05), 1.0)
        ],
        'Seuil Acceptable': [0.6, 0.8, 0.7, 0.8]
    })
    
    fig_comparison = px.bar(
        comparison_data,
        x='Critère',
        y=['Client', 'Seuil Acceptable'],
        title="📊 Comparaison avec les Seuils Standards",
        barmode='group',
        color_discrete_sequence=['#667eea', '#f093fb']
    )
    
    # Timeline simulation
    months = list(range(1, 37))
    monthly_payment = client_data["AMT_ANNUITY"]
    remaining_balance = [client_data["AMT_CREDIT"] - (i * monthly_payment) for i in months]
    risk_evolution = [prob * (1 + 0.1 * np.sin(i/6)) for i in months]
    
    fig_timeline = go.Figure()
    fig_timeline.add_trace(go.Scatter(
        x=months, y=remaining_balance,
        mode='lines+markers',
        name='Solde Restant (€)',
        line=dict(color='#48dbfb', width=3),
        yaxis='y'
    ))
    
    fig_timeline.add_trace(go.Scatter(
        x=months, y=risk_evolution,
        mode='lines+markers',
        name='Évolution du Risque (%)',
        line=dict(color='#ff6b6b', width=3),
        yaxis='y2'
    ))
    
    fig_timeline.update_layout(
        title="📈 Simulation d'Évolution du Crédit sur 3 ans",
        xaxis_title="Mois",
        yaxis=dict(title="Solde Restant (€)", side="left"),
        yaxis2=dict(title="Niveau de Risque", side="right", overlaying="y"),
        hovermode='x unified'
    )
    
    return fig_factors, fig_comparison, fig_timeline

def display_banker_explanation(client_data, result, client_id=None):
    """Affichage détaillé pour explication banquier-client"""
    
    prediction = result["prediction"]
    prob = prediction["probability"]
    decision = prediction["decision"]
    
    # En-tête de résultat avec couleurs
    if "FIABLE" in decision:
        st.markdown(f"""
        <div class="success-card">
            <h2>✅ CRÉDIT ACCORDÉ</h2>
            <p><strong>Risque évalué:</strong> {prob:.1%} (Seuil: 9%)</p>
            {"<p><strong>Client ID:</strong> " + str(client_id) + "</p>" if client_id else ""}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="danger-card">
            <h2>❌ CRÉDIT REFUSÉ</h2>
            <p><strong>Risque évalué:</strong> {prob:.1%} (Seuil dépassé: 9%)</p>
            {"<p><strong>Client ID:</strong> " + str(client_id) + "</p>" if client_id else ""}
        </div>
        """, unsafe_allow_html=True)
    
    # Métriques principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        age_years = abs(client_data["DAYS_BIRTH"]) // 365
        st.metric("👤 Âge", f"{age_years} ans")
    
    with col2:
        ratio = client_data["AMT_CREDIT"] / client_data["AMT_GOODS_PRICE"]
        st.metric("💰 Taux Financement", f"{ratio:.1%}")
    
    with col3:
        monthly_income_est = client_data["AMT_CREDIT"] / 60
        st.metric("📊 Effort Mensuel", f"{client_data['AMT_ANNUITY'] / monthly_income_est:.1%}")
    
    with col4:
        ext_avg = (client_data.get("EXT_SOURCE_1", 0.5) + client_data.get("EXT_SOURCE_2", 0.5) + client_data.get("EXT_SOURCE_3", 0.5)) / 3
        st.metric("⭐ Score Moyen", f"{ext_avg:.2f}/1.0")
    
    # Jauge principale avec seuil 9%
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={'text': "🎯 Niveau de Risque de Défaut", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 9], 'color': '#a8e6cf'},
                {'range': [9, 30], 'color': '#ffd93d'},
                {'range': [30, 50], 'color': '#ff8b94'},
                {'range': [50, 100], 'color': '#ff6b6b'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 9
            }
        }
    ))
    
    fig_gauge.update_layout(height=400, font={'color': "darkblue", 'family': "Arial"})
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Graphiques d'explication
    st.markdown("### 📋 Analyse Détaillée pour Discussion Client")
    
    fig_factors, fig_comparison, fig_timeline = create_explanation_charts(client_data, result)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_factors, use_container_width=True)
    with col2:
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    # Arguments pour le banquier
    st.markdown("### 💬 Arguments pour Discussion avec le Client")
    
    if "FIABLE" in decision:
        st.success(f"""
        **Points Positifs à Mentionner:**
        - Votre profil présente un risque faible de {prob:.1%}
        - Vos scores externes sont satisfaisants
        - Votre capacité de remboursement est adaptée
        - Le ratio de financement reste raisonnable à {client_data["AMT_CREDIT"] / client_data["AMT_GOODS_PRICE"]:.1%}
        
        **Conditions Proposées:**
        - Montant accordé: {client_data["AMT_CREDIT"]:,} €
        - Mensualité: {client_data["AMT_ANNUITY"]:,} €
        - Durée estimée: {client_data["AMT_CREDIT"] // client_data["AMT_ANNUITY"]:.0f} mois
        """)
    else:
        reasons = []
        if prob > 0.09:
            reasons.append("Le score de risque dépasse le seuil de 9%")
        if client_data["AMT_CREDIT"] / client_data["AMT_GOODS_PRICE"] > 0.9:
            reasons.append("Le taux de financement est trop élevé")
        if ext_avg < 0.4:
            reasons.append("Les scores externes sont insuffisants")
        
        st.error(f"""
        **Raisons du Refus:**
        """ + "\n".join([f"- {reason}" for reason in reasons]) + f"""
        
        **Alternatives à Proposer:**
        - Réduire le montant à {int(client_data["AMT_CREDIT"] * 0.8):,} € (risque estimé: {prob * 0.8:.1%})
        - Augmenter l'apport personnel
        - Proposer un co-emprunteur
        - Revoir la demande dans 6 mois
        """)

# Interface principale
if "🔍 Client Existant" in mode:
    st.markdown("## 🔍 Recherche et Analyse Client")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        client_id = st.number_input("🆔 Numéro Client", value=100001, min_value=100000, step=1)
    with col2:
        analyze_btn = st.button("🔍 Analyser le Dossier", type="primary", use_container_width=True)
    
    if analyze_btn:
        if not api_ok:
            st.error("🔌 API non disponible")
        else:
            with st.spinner("🔄 Analyse en cours..."):
                # Simulation données client
                np.random.seed(client_id)
                client_data = {
                    "EXT_SOURCE_2": np.random.uniform(0.2, 0.9),
                    "EXT_SOURCE_3": np.random.uniform(0.2, 0.9),
                    "AMT_GOODS_PRICE": np.random.randint(200000, 800000),
                    "CODE_GENDER": np.random.choice(["M", "F"]),
                    "EXT_SOURCE_1": np.random.uniform(0.2, 0.9),
                    "AMT_ANNUITY": np.random.randint(15000, 50000),
                    "AMT_CREDIT": np.random.randint(250000, 900000),
                    "DAYS_BIRTH": np.random.randint(-25000, -8000),
                    "INST_PAYMENT_PERC_mean": np.random.uniform(0.7, 1.0),
                    "NAME_EDUCATION_TYPE": "Higher education"
                }
                
                try:
                    response = requests.post(f"{API_URL}/predict/demo", json=client_data, timeout=10)
                    if response.status_code == 200:
                        result = response.json()
                        display_banker_explanation(client_data, result, client_id)
                    else:
                        st.error(f"❌ Erreur API: {response.status_code}")
                except Exception as e:
                    st.error(f"⚠️ Erreur: {str(e)}")

else:  # Nouveau Client
    st.markdown("## ➕ Nouvelle Demande de Crédit")
    
    with st.form("nouveau_client_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 👤 Informations Personnelles")
            age = st.slider("Âge", 18, 80, 35)
            gender = st.selectbox("Genre", ["M", "F"])
            education = st.selectbox("Niveau d'études", [
                "Secondary / secondary special",
                "Higher education",
                "Lower secondary"
            ])
            
            st.markdown("#### 💰 Demande de Crédit")
            credit_amount = st.number_input("Montant demandé (€)", value=300000, step=10000)
            goods_price = st.number_input("Prix du bien (€)", value=350000, step=10000)
            annuity = st.number_input("Mensualité souhaitée (€)", value=15000, step=1000)
        
        with col2:
            st.markdown("#### 📊 Évaluations Externes")
            st.info("Scores bureaux de crédit (0=risqué, 1=excellent)")
            ext1 = st.slider("Bureau Crédit 1", 0.0, 1.0, 0.6, step=0.01)
            ext2 = st.slider("Bureau Crédit 2", 0.0, 1.0, 0.6, step=0.01)
            ext3 = st.slider("Bureau Crédit 3", 0.0, 1.0, 0.6, step=0.01)
            
            st.markdown("#### 📈 Historique")
            payment_history = st.slider("Régularité paiements", 0.0, 1.0, 0.9, step=0.05)
            st.caption("1.0 = Toujours ponctuel")
        
        submitted = st.form_submit_button("🎯 Analyser la Demande", type="primary", use_container_width=True)
    
    if submitted:
        if not api_ok:
            st.error("🔌 API non disponible")
        else:
            data = {
                "EXT_SOURCE_2": ext2,
                "EXT_SOURCE_3": ext3,
                "AMT_GOODS_PRICE": goods_price,
                "CODE_GENDER": gender,
                "EXT_SOURCE_1": ext1,
                "AMT_ANNUITY": annuity,
                "AMT_CREDIT": credit_amount,
                "DAYS_BIRTH": -age * 365,
                "INST_PAYMENT_PERC_mean": payment_history,
                "NAME_EDUCATION_TYPE": education
            }
            
            try:
                response = requests.post(f"{API_URL}/predict/demo", json=data, timeout=10)
                if response.status_code == 200:
                    result = response.json()
                    display_banker_explanation(data, result)
                else:
                    st.error(f"❌ Erreur: {response.text}")
            except Exception as e:
                st.error(f"⚠️ Problème: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <strong>🏦 Prêt à Dépenser</strong> - Modèle LightGBM Champion (Seuil: 9%)<br>
    <em>Interface banquier avec explications détaillées</em>
</div>
""", unsafe_allow_html=True)