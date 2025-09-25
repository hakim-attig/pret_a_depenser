# api_demo.py - Version démo simplifiée
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from typing import Optional

# Modèle simplifié basé sur tes TOP features SHAP
class DemoClientData(BaseModel):
    # Top 5 variables les plus importantes selon SHAP
    EXT_SOURCE_2: Optional[float] = Field(0.5, ge=0, le=1, description="Score externe 2 (0-1)")
    EXT_SOURCE_3: Optional[float] = Field(0.5, ge=0, le=1, description="Score externe 3 (0-1)")
    AMT_GOODS_PRICE: float = Field(450000, gt=0, description="Prix du bien en euros")
    CODE_GENDER: str = Field("M", description="Genre: M (Masculin), F (Féminin), XNA (Non spécifié)")
    EXT_SOURCE_1: Optional[float] = Field(0.5, ge=0, le=1, description="Score externe 1 (0-1)")
    
    # Variables financières importantes
    AMT_ANNUITY: float = Field(25000, gt=0, description="Annuité mensuelle")
    AMT_CREDIT: float = Field(500000, gt=0, description="Montant du crédit demandé")
    DAYS_BIRTH: int = Field(-15000, lt=0, description="Age en jours (négatif, ex: -15000 = ~41 ans)")
    
    # Variables comportementales
    INST_PAYMENT_PERC_mean: Optional[float] = Field(1.0, ge=0, le=2, description="% moyen de paiement des échéances")
    NAME_EDUCATION_TYPE: str = Field("Secondary / secondary special", description="Niveau d'éducation")

def predict_demo(client_data: DemoClientData, model, feature_columns, threshold):
    """Fonction de prédiction démo avec variables importantes"""
    
    # Créer DataFrame avec valeurs par défaut
    demo_data = pd.DataFrame(0.0, index=[0], columns=feature_columns)
    
    # Mapping des variables d'entrée
    demo_mapping = {
        'EXT_SOURCE_2': client_data.EXT_SOURCE_2,
        'EXT_SOURCE_3': client_data.EXT_SOURCE_3,
        'AMT_GOODS_PRICE': client_data.AMT_GOODS_PRICE,
        'EXT_SOURCE_1': client_data.EXT_SOURCE_1,
        'AMT_ANNUITY': client_data.AMT_ANNUITY,
        'AMT_CREDIT': client_data.AMT_CREDIT,
        'DAYS_BIRTH': client_data.DAYS_BIRTH,
        'INST_PAYMENT_PERC_mean': client_data.INST_PAYMENT_PERC_mean
    }
    
    # Remplir les features disponibles
    for feature, value in demo_mapping.items():
        if feature in demo_data.columns and value is not None:
            demo_data[feature] = value
    
    # Variables catégorielles
    gender_mapping = {"M": 1, "F": 0, "XNA": 2}
    education_mapping = {
        "Secondary / secondary special": 0,
        "Higher education": 1,
        "Incomplete higher": 2,
        "Lower secondary": 3,
        "Academic degree": 4
    }
    
    if 'CODE_GENDER' in demo_data.columns:
        demo_data['CODE_GENDER'] = gender_mapping.get(client_data.CODE_GENDER, 1)
    
    if 'NAME_EDUCATION_TYPE' in demo_data.columns:
        demo_data['NAME_EDUCATION_TYPE'] = education_mapping.get(client_data.NAME_EDUCATION_TYPE, 0)
    
    # Prédiction
    proba = model.predict_proba(demo_data)[0][1]
    decision = proba >= threshold
    
    # Calculs additionnels pour l'affichage
    age_years = abs(client_data.DAYS_BIRTH) // 365
    credit_goods_ratio = client_data.AMT_CREDIT / client_data.AMT_GOODS_PRICE
    
    return {
        "method": "demo_simplified",
        "key_features_used": 8,
        "prediction": {
            "probability": round(float(proba), 4),
            "decision": "RISQUE DE DÉFAUT" if decision else "CLIENT FIABLE",
            "confidence": round(max(proba, 1-proba), 3),
            "risk_level": "ÉLEVÉ" if proba > 0.7 else "MOYEN" if proba > 0.3 else "FAIBLE"
        },
        "client_analysis": {
            "age_estimé": f"{age_years} ans",
            "ratio_financement": f"{credit_goods_ratio:.1%}",
            "profil_paiement": "Excellent" if client_data.INST_PAYMENT_PERC_mean >= 0.95 else "Bon" if client_data.INST_PAYMENT_PERC_mean >= 0.8 else "À surveiller",
            "scores_externes": f"Source 1: {client_data.EXT_SOURCE_1 or 'N/A'}, Source 2: {client_data.EXT_SOURCE_2 or 'N/A'}, Source 3: {client_data.EXT_SOURCE_3 or 'N/A'}"
        }
    }