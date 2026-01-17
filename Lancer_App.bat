@echo off
cd /d "%~dp0"
echo 🚀 Lancement de PixFinder AI...
echo Ne ferme pas cette fenetre noire tant que tu utilises le site.
echo.
python -m streamlit run app.py
pause