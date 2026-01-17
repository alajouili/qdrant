import os
from qdrant_client import QdrantClient
from transformers import CLIPProcessor, CLIPModel
import torch

# --- 1. CONNEXION À LA BASE EXISTANTE ---
dossier_script = os.path.dirname(os.path.abspath(__file__))
os.chdir(dossier_script)

print("🚀 Connexion à Qdrant...")
client = QdrantClient(path="./ma_base_qdrant")
COLLECTION_NAME = "mes_photos"

# --- 2. CHARGEMENT IA ---
print("🧠 Chargement du cerveau (CLIP)...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# --- 3. INTERACTION UTILISATEUR ---
print("\n" + "="*40)
texte_recherche = input("✍️  Que cherches-tu dans tes photos ? (ex: un chien, une voiture...) : ")
print("="*40 + "\n")

# --- 4. VECTORISATION DU TEXTE ---
# On transforme ta phrase en mathématiques
inputs = processor(text=[texte_recherche], return_tensors="pt", padding=True)

with torch.no_grad():
    text_features = model.get_text_features(**inputs)
    # Conversion en liste pour Qdrant
    query_vector = text_features.detach().numpy()[0].tolist()

# --- 5. RECHERCHE ---
hits = client.search(
    collection_name=COLLECTION_NAME,
    query_vector=query_vector,
    limit=3
)

print(f"🔎 Résultats pour '{texte_recherche}' :")
found = False
for hit in hits:
    if hit.score > 0.2: # On filtre les résultats trop faibles
        found = True
        print(f"📸 Trouvé : {hit.payload['filename']}  (Score de ressemblance : {hit.score:.3f})")

if not found:
    print("❌ Aucune image correspondante trouvée (Score trop faible).")