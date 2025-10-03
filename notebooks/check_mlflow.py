import mlflow
from mlflow.tracking import MlflowClient
import os

# 📌 Chemin de ton dossier mlruns
mlruns_path = os.path.abspath("./mlruns")

print(f"📂 Dossier courant : {os.getcwd()}")
print(f"🔗 Tracking URI : file:///{mlruns_path.replace('\\', '/')}")

# ✅ Forcer MLflow à utiliser le bon dossier
mlflow.set_tracking_uri(f"file:///{mlruns_path.replace('\\', '/')}")

# ✅ Initialiser le client MLflow
client = MlflowClient()

# ✅ Lister les expériences
print("\n📊 Liste des expériences :")
experiments = client.search_experiments()
if not experiments:
    print("❌ Aucune expérience trouvée")
else:
    for exp in experiments:
        print(f" - ID={exp.experiment_id} | Nom={exp.name} | Lifecycle={exp.lifecycle_stage}")

# ✅ Lister les runs pour chaque expérience
for exp in experiments:
    runs = client.search_runs([exp.experiment_id])
    print(f"\n📌 Runs pour l'expérience '{exp.name}' ({exp.experiment_id}) :")
    if not runs:
        print("   ⚠️ Aucun run")
    else:
        for run in runs:
            print(f"   - Run ID: {run.info.run_id} | Status: {run.info.status} | Start: {run.info.start_time}")
