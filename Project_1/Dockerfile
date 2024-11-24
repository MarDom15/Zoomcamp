# Utiliser Python comme base
FROM python:3.12.4

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers nécessaires
COPY . /app

# Installer les dépendances
RUN pip install -r requirements.txt

# Exposer le port par défaut de Streamlit
EXPOSE 8501

# Lancer l'application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]
