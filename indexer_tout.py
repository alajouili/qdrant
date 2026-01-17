import os
import shutil
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# --- CONFIGURATION ---
dossier_script = os.path.dirname(os.path.abspath(__file__))
os.chdir(dossier_script)

# On définit les dossiers proprement
path_db = "./ma_base_qdrant"
dossier_images = "./images"  # <--- C'est ici qu'il va chercher maintenant !
COLLECTION_NAME = "mes_photos"

# --- 1. INITIALISATION ---
print("🚀 Démarrage de l'indexation massive (Dossier 'images')...")

# Vérification que le dossier images existe
if not os.path.exists(dossier_images):
    print(f"❌ ERREUR : Le dossier '{dossier_images}' n'existe pas ! Crée-le et mets tes photos dedans.")
    exit()

if os.path.exists(path_db):
    shutil.rmtree(path_db)

client = QdrantClient(path=path_db)
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=512, distance=Distance.COSINE),
)

print("🧠 Chargement de l'IA...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# --- 2. SCAN DU DOSSIER IMAGES ---
fichiers = os.listdir(dossier_images)
images_a_traiter = [f for f in fichiers if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))]

print(f"\n📂 J'ai trouvé {len(images_a_traiter)} images dans le dossier '{dossier_images}'.")
print("-" * 30)

compteur = 0
for i, nom_image in enumerate(images_a_traiter):
    try:
        chemin_complet = os.path.join(dossier_images, nom_image)
        
        # Analyse de l'image
        image = Image.open(chemin_complet)
        inputs = processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.get_image_features(**inputs)
            vector = outputs.detach().numpy()[0].tolist()

        # Envoi vers Qdrant
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=i+1,
                    vector=vector,
                    payload={"filename": nom_image} # On garde juste le nom
                )
            ]
        )
        print(f"✅ Indexée : {nom_image}")
        compteur += 1
        
    except Exception as e:
        print(f"❌ Échec sur {nom_image} : {e}")

print("-" * 30)
print(f"🎉 Terminé ! Base de données mise à jour avec tes {compteur} photos bien rangées.")