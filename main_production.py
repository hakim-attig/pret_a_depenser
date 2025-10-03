from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import Optional

app = FastAPI(
    title="API Scoring Crédit - Production",
    description="API avec champion XGBoost (coût: 30,523€)",
    version="1.0"
)

# Chargement des modèles
MODEL_DIR = "../models"

try:
    model = joblib.load(f"{MODEL_DIR}/champion_model.pkl")
    threshold = joblib.load(f"{MODEL_DIR}/champion_threshold.pkl")
    feature_columns = joblib.load(f"{MODEL_DIR}/feature_columns.pkl")
    metadata = joblib.load(f"{MODEL_DIR}/model_metadata.pkl")
    imputer = joblib.load(f"{MODEL_DIR}/imputer.pkl")
    
    print(f"✓ Modèle chargé: {metadata['model_type']}")
    print(f"✓ Seuil: {threshold:.3f}")
    print(f"✓ Features: {len(feature_columns)}")
    model_loaded = True
except Exception as e:
    print(f"Erreur: {e}")
    model_loaded = False

# Charger les clients démo
try:
    demo_clients = pd.read_csv("../all_clients_validation.csv")
    print(f"✓ {len(demo_clients)} clients démo chargés")
except:
    demo_clients = None

class PredictionRequest(BaseModel):
    client_id: int

@app.get("/")
def root():
    return {
        "api": "Scoring Crédit Production",
        "model": metadata['model_type'] if model_loaded else "Non chargé",
        "threshold": float(threshold) if model_loaded else None,
        "auc": float(metadata['auc_score']) if model_loaded else None,
        "cost": float(metadata['optimal_cost']) if model_loaded else None,
        "status": "OK" if model_loaded else "ERROR"
    }

@app.get("/status")
def health_check():
    return {
        "status": "operational" if model_loaded else "error",
        "model_loaded": model_loaded,
        "demo_clients_available": demo_clients is not None
    }

@app.get("/clients")
def list_clients(limit: int = 100):
    if demo_clients is None:
        raise HTTPException(status_code=500, detail="Clients non disponibles")
    
    clients_list = []
    for idx, row in demo_clients.head(limit).iterrows():
        clients_list.append({
            "client_id": int(row['SK_ID_CURR']),
            "risk_score": float(row['RISK_SCORE']),
            "decision": row['DECISION'],
            "real_target": int(row['REAL_TARGET'])
        })
    
    return {
        "total_clients": len(demo_clients),
        "clients_returned": len(clients_list),
        "clients": clients_list
    }

@app.post("/predict")
def predict_client(request: PredictionRequest):
    """Prédiction pour un client du dataset démo"""
    if not model_loaded:
        raise HTTPException(status_code=500, detail="Modèle non chargé")
    
    if demo_clients is None:
        raise HTTPException(status_code=500, detail="Clients démo non disponibles")
    
    # Vérifier que le client existe
    if request.client_id not in demo_clients.index:
        raise HTTPException(status_code=404, detail=f"Client {request.client_id} non trouvé")
    
    # Récupérer le client
    client_data = demo_clients.loc[request.client_id]
    
    # Préparer les features
    features_to_drop = ['RISK_SCORE', 'DECISION', 'REAL_TARGET']
    X = client_data.drop(features_to_drop)
    X = X[feature_columns]
    
    # Prédiction
    try:
        proba = model.predict_proba(X.values.reshape(1, -1))[0, 1]
        decision = "REFUS" if proba >= threshold else "ACCORD"
        
        # CORRECTION: Conversion explicite en types Python standard
        real_target = int(client_data['REAL_TARGET'])
        is_correct = bool((decision == "ACCORD" and real_target == 0) or 
                         (decision == "REFUS" and real_target == 1))
        
        return {
            "client_id": int(request.client_id),
            "risk_score": float(proba),
            "risk_percentage": f"{proba*100:.2f}%",
            "threshold": float(threshold),
            "decision": decision,
            "real_target": real_target,
            "real_label": "Défaut" if real_target == 1 else "Bon",
            "prediction_correct": is_correct
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur prédiction: {str(e)}")

@app.get("/model/info")
def model_info():
    """Informations sur le modèle"""
    if not model_loaded:
        raise HTTPException(status_code=500, detail="Modèle non chargé")
    
    return {
        "model_type": metadata['model_type'],
        "auc_score": float(metadata['auc_score']),
        "optimal_threshold": float(threshold),
        "optimal_cost": float(metadata['optimal_cost']),
        "num_features": len(feature_columns),
        "training_date": metadata['training_date'],
        "confusion_matrix": metadata['confusion_matrix']
    }

@app.get("/explain/{client_id}")
def explain_prediction(client_id: int):
    """Explication SHAP pour un client"""
    if not model_loaded or demo_clients is None:
        raise HTTPException(status_code=500, detail="Service non disponible")
    
    if client_id not in demo_clients.index:
        raise HTTPException(status_code=404, detail="Client non trouvé")
    
    try:
        # Charger explainer SHAP
        import shap
        explainer = joblib.load(f"{MODEL_DIR}/shap_explainer.pkl")
        
        # Préparer données client
        client_data = demo_clients.loc[client_id]
        X = client_data.drop(['RISK_SCORE', 'DECISION', 'REAL_TARGET'])[feature_columns]
        
        # Calculer SHAP values
        shap_values = explainer.shap_values(X.values.reshape(1, -1))
        if isinstance(shap_values, list):
            shap_values = shap_values[1][0]
        else:
            shap_values = shap_values[0]
        
        # Top 10 features
        feature_impact = list(zip(feature_columns, shap_values, X.values))
        feature_impact.sort(key=lambda x: abs(x[1]), reverse=True)
        
        top_features = []
        for feat, impact, value in feature_impact[:10]:
            top_features.append({
                "feature": feat,
                "impact": float(impact),
                "value": float(value),
                "direction": "AUGMENTE" if impact > 0 else "DIMINUE"
            })
        
        return {
            "client_id": int(client_id),
            "top_features": top_features,
            "interpretation": "Impact positif = augmente le risque | Impact négatif = diminue le risque"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur SHAP: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)