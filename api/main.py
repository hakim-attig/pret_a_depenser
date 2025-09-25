# main.py - API Scoring Crédit avec LightGBM Champion
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from typing import Optional

app = FastAPI(
    title="API Scoring Crédit - LightGBM Champion",
    description="API avec modèle LightGBM optimisé (seuil: 9%)",
    version="2.1.0"
)

# Charger le modèle au démarrage
model_dir = "../models"

try:
    model = joblib.load(f"{model_dir}/xgboost_final_model.pkl")
    # CORRECTION: Utiliser directement le bon seuil LightGBM
    threshold = 0.090  # Seuil champion LightGBM
    feature_columns = joblib.load(f"{model_dir}/feature_columns.pkl")
    print(f"Modèle LightGBM chargé: {len(feature_columns)} features, seuil: {threshold:.3f}")
    model_loaded = True
except Exception as e:
    print(f"Erreur chargement: {e}")
    model = threshold = feature_columns = None
    model_loaded = False

# Modèle démo (variables importantes)
class DemoClientData(BaseModel):
    EXT_SOURCE_2: Optional[float] = Field(0.5, ge=0, le=1, description="Score externe 2")
    EXT_SOURCE_3: Optional[float] = Field(0.5, ge=0, le=1, description="Score externe 3")
    AMT_GOODS_PRICE: float = Field(450000, gt=0, description="Prix du bien")
    CODE_GENDER: str = Field("M", description="Genre (M/F/XNA)")
    EXT_SOURCE_1: Optional[float] = Field(0.5, ge=0, le=1, description="Score externe 1")
    AMT_ANNUITY: float = Field(25000, gt=0, description="Annuité")
    AMT_CREDIT: float = Field(500000, gt=0, description="Montant crédit")
    DAYS_BIRTH: int = Field(-15000, lt=0, description="Age en jours négatifs")
    INST_PAYMENT_PERC_mean: Optional[float] = Field(1.0, ge=0, description="% paiement moyen")
    NAME_EDUCATION_TYPE: str = Field("Secondary / secondary special", description="Education")

# Modèle complet (toutes variables)
class CompleteClientData(BaseModel):
    AMT_INCOME_TOTAL: float = Field(..., description="Revenus totaux")
    AMT_CREDIT: float = Field(..., description="Montant crédit")
    AMT_ANNUITY: float = Field(..., description="Annuité")
    AMT_GOODS_PRICE: float = Field(..., description="Prix du bien")
    DAYS_BIRTH: int = Field(..., description="Age en jours négatifs")
    EXT_SOURCE_1: Optional[float] = Field(None, description="Score externe 1")
    EXT_SOURCE_2: Optional[float] = Field(None, description="Score externe 2") 
    EXT_SOURCE_3: Optional[float] = Field(None, description="Score externe 3")
    CODE_GENDER: str = Field("M", description="Genre")
    NAME_EDUCATION_TYPE: str = Field("Secondary / secondary special", description="Education")

def predict_demo_logic(client_data: DemoClientData):
    """Logique de prédiction pour le mode démo"""
    demo_data = pd.DataFrame(0.0, index=[0], columns=feature_columns)
    
    # Mapper les variables importantes
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
    
    for feature, value in demo_mapping.items():
        if feature in demo_data.columns and value is not None:
            demo_data[feature] = value
    
    # Variables catégorielles
    gender_map = {"M": 1, "F": 0, "XNA": 2}
    education_map = {
        "Secondary / secondary special": 0,
        "Higher education": 1,
        "Incomplete higher": 2,
        "Lower secondary": 3,
        "Academic degree": 4
    }
    
    if 'CODE_GENDER' in demo_data.columns:
        demo_data['CODE_GENDER'] = gender_map.get(client_data.CODE_GENDER, 1)
    
    if 'NAME_EDUCATION_TYPE' in demo_data.columns:
        demo_data['NAME_EDUCATION_TYPE'] = education_map.get(client_data.NAME_EDUCATION_TYPE, 0)
    
    # Prédiction avec gestion d'erreur
    try:
        proba = model.predict_proba(demo_data)[0][1]
        # Conversion sécurisée en float standard
        if hasattr(proba, 'item'):
            proba = float(proba.item())
        else:
            proba = float(proba)
        
        decision = proba >= threshold
        age_years = abs(client_data.DAYS_BIRTH) // 365
        
        return {
            "method": "demo_mode",
            "prediction": {
                "probability": round(proba, 4),
                "decision": "RISQUE DE DÉFAUT" if decision else "CLIENT FIABLE",
                "confidence": round(max(proba, 1-proba), 3)
            },
            "client_analysis": {
                "age_estimé": f"{age_years} ans",
                "ratio_financement": f"{client_data.AMT_CREDIT / client_data.AMT_GOODS_PRICE:.1%}"
            }
        }
    except Exception as e:
        raise Exception(f"Erreur lors de la prédiction: {str(e)}")

def predict_complete_logic(client_data: CompleteClientData):
    """Logique de prédiction pour le mode complet"""
    full_data = pd.DataFrame(0.0, index=[0], columns=feature_columns)
    
    # Mapper toutes les variables fournies
    for field_name, value in client_data.dict().items():
        if field_name in full_data.columns and value is not None:
            full_data[field_name] = value
    
    # Prédiction avec gestion d'erreur
    try:
        proba = model.predict_proba(full_data)[0][1]
        if hasattr(proba, 'item'):
            proba = float(proba.item())
        else:
            proba = float(proba)
            
        decision = proba >= threshold
        
        return {
            "method": "complete_mode",
            "features_used": len([col for col in full_data.columns if full_data[col].iloc[0] != 0]),
            "probability": round(proba, 4),
            "decision": "DÉFAUT" if decision else "BON CLIENT",
            "confidence": round(max(proba, 1-proba), 3)
        }
    except Exception as e:
        raise Exception(f"Erreur lors de la prédiction: {str(e)}")

@app.get("/")
def root():
    return {
        "API": "Scoring Crédit - Prêt à Dépenser",
        "champion": "LightGBM",
        "threshold": f"{threshold:.1%}" if threshold else "Non chargé",
        "status": "En ligne" if model_loaded else "Modèle non chargé",
        "modes": {
            "demo": "POST /predict/demo - Variables importantes",
            "production": "POST /predict/complete - Toutes variables"
        }
    }

@app.get("/status")
def health_check():
    return {
        "api_status": "online",
        "model_loaded": model_loaded,
        "features_count": len(feature_columns) if feature_columns else 0,
        "model_type": "LightGBM",
        "threshold": float(threshold) if threshold else None
    }

@app.post("/predict/demo")
def predict_demo_endpoint(client: DemoClientData):
    if not model_loaded:
        raise HTTPException(status_code=500, detail="Modèle non disponible")
    
    try:
        result = predict_demo_logic(client)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur prédiction: {str(e)}")

@app.post("/predict/complete")
def predict_complete_endpoint(client: CompleteClientData):
    if not model_loaded:
        raise HTTPException(status_code=500, detail="Modèle non disponible")
    
    try:
        result = predict_complete_logic(client)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erreur prédiction: {str(e)}")

@app.get("/model/info")
def model_info():
    return {
        "model_type": "LightGBM Champion",
        "features_count": len(feature_columns) if feature_columns else 0,
        "threshold": float(threshold) if threshold else None,
        "optimal_cost": 30666,
        "auc_score": 0.7800
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)