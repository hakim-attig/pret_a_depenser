# api_complete.py - Version production complète
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

# Modèle pour recevoir TOUTES les features (simplifié pour demo)
class CompleteClientData(BaseModel):
    # Variables principales (obligatoires)
    AMT_INCOME_TOTAL: float = Field(..., description="Revenus totaux")
    AMT_CREDIT: float = Field(..., description="Montant du crédit")
    AMT_ANNUITY: float = Field(..., description="Annuité")
    AMT_GOODS_PRICE: float = Field(..., description="Prix du bien")
    DAYS_BIRTH: int = Field(..., description="Age en jours (négatif)")
    
    # Variables importantes (optionnelles avec valeurs par défaut)
    EXT_SOURCE_1: Optional[float] = Field(None, description="Score externe 1")
    EXT_SOURCE_2: Optional[float] = Field(None, description="Score externe 2") 
    EXT_SOURCE_3: Optional[float] = Field(None, description="Score externe 3")
    CODE_GENDER: str = Field("M", description="Genre (M/F/XNA)")
    NAME_EDUCATION_TYPE: str = Field("Secondary / secondary special", description="Education")
    
    # Autres variables avec valeurs par défaut
    additional_features: Optional[Dict[str, float]] = Field(default_factory=dict, description="Autres features (optionnel)")

def predict_complete(client_data: CompleteClientData, model, feature_columns, threshold):
    """Fonction de prédiction complète"""
    
    # Créer DataFrame avec toutes les features
    full_data = pd.DataFrame(0.0, index=[0], columns=feature_columns)
    
    # Remplir les variables fournies
    for field_name, value in client_data.dict().items():
        if field_name in full_data.columns and value is not None:
            full_data[field_name] = value
    
    # Ajouter les features additionnelles si fournies
    if client_data.additional_features:
        for feature, value in client_data.additional_features.items():
            if feature in full_data.columns:
                full_data[feature] = value
    
    # Prédiction
    proba = model.predict_proba(full_data)[0][1]
    decision = proba >= threshold
    
    return {
        "method": "complete_features",
        "features_used": len([col for col in full_data.columns if full_data[col].iloc[0] != 0]),
        "total_features": len(feature_columns),
        "probability": round(float(proba), 4),
        "decision": "DÉFAUT" if decision else "BON CLIENT",
        "confidence": round(max(proba, 1-proba), 3)
    }