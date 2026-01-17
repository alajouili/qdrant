import streamlit as st
import os
from qdrant_client import QdrantClient
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# --- 1. CONFIGURATION DE LA PAGE (MODE PRO) ---
st.set_page_config(
    page_title="PixFinder AI",
    page_icon="✨",
    layout="wide",  # Utilise tout l'écran
    initial_sidebar_state="expanded"
)

# --- CSS PERSONNALISÉ (POUR LE STYLE) ---
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stTextInput > div > div > input {
        font-size: 20px;
        padding: 15px;
        border-radius: 10px;
    }
    h1 {
        color: #2c3e50;
        text-align: center;
    }
    .result-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CHEMINS ---
dossier_script = os.path.dirname(os.path.abspath(__file__))
path_db = os.path.join(dossier_script, "ma_base_qdrant")
dossier_images = os.path.join(dossier_script, "images")
COLLECTION_NAME = "mes_photos"

# --- 2. CHARGEMENT IA (CACHE) ---
@st.cache_resource
def load_ai():
    return CLIPModel.from_pretrained("openai/clip-vit-base-patch32"), CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

@st.cache_resource
def load_db():
    return QdrantClient(path=path_db)

try:
    with st.spinner('🚀 Initialisation du moteur neuronal...'):
        model, processor = load_ai()
        client = load_db()
except Exception as e:
    st.error(f"Erreur de chargement : {e}")
    st.stop()

# --- 3. BARRE LATÉRALE (SIDEBAR) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103930.png", width=80)
    st.title("⚙️ Réglages")
    st.markdown("---")
    
    # Curseur pour filtrer les résultats
    seuil = st.slider("🎯 Précision minimale", 0.0, 1.0, 0.20, 0.05)
    st.caption("Plus le seuil est haut, plus la recherche est stricte.")
    
    nb_results = st.number_input("Nombre de résultats max", min_value=1, max_value=20, value=8)
    
    st.markdown("---")
    st.info("💡 **Astuce :** Tu peux chercher des concepts abstraits comme 'joie', 'vacances' ou 'tristesse'.")

# --- 4. INTERFACE PRINCIPALE ---
st.title("✨ PixFinder AI")
st.markdown("### *Retrouve tes souvenirs instantanément*")
st.write("") # Espace vide

# Barre de recherche (Gros input)
col_search, _ = st.columns([2, 1]) # Pour centrer un peu
query = st.text_input("🔍 Décris l'image que tu cherches...", placeholder="Ex: Une voiture rouge sur la plage...")

if query:
    # Vectorisation
    inputs = processor(text=[query], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        query_vector = text_features.detach().numpy()[0].tolist()

    # Recherche Qdrant
    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=nb_results
    )

    st.markdown("---")
    st.subheader(f"📸 Résultats pour : *{query}*")
    
    # --- AFFICHAGE EN GRILLE ---
    # On filtre d'abord
    filtered_hits = [hit for hit in hits if hit.score >= seuil]
    
    if not filtered_hits:
        st.warning("⚠️ Aucune image ne correspond assez bien à ta description. Essaie de baisser la précision dans le menu de gauche.")
    else:
        # Création d'une grille de 4 colonnes
        cols = st.columns(4)
        
        for i, hit in enumerate(filtered_hits):
            nom_fichier = hit.payload['filename']
            chemin_image = os.path.join(dossier_images, nom_fichier)
            
            # Fallback si l'image est à la racine
            if not os.path.exists(chemin_image):
                 chemin_image = os.path.join(dossier_script, nom_fichier)

            if os.path.exists(chemin_image):
                col = cols[i % 4] # Distribution dans les 4 colonnes
                with col:
                    # Affichage propre
                    img = Image.open(chemin_image)
                    st.image(img, use_column_width=True)
                    
                    # Barre de progression pour le score
                    score_pct = int(hit.score * 100)
                    if score_pct > 100: score_pct = 100
                    
                    # Couleur de la barre selon le score
                    color_bar = "green" if score_pct > 25 else "orange"
                    
                    st.progress(score_pct / 100, text=f"Confiance : {score_pct}%")
                    st.caption(f"📄 {nom_fichier}")
            else:
                pass # On n'affiche pas les erreurs pour garder l'interface propre