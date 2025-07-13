#!/bin/bash
# Skrypt do deploymentu aplikacji na Google Cloud Run

# Ustaw zmienne
PROJECT_ID="ai-calambury-prod"
SERVICE_NAME="ai-calambury"
REGION="europe-west1"

# Włącz API Cloud Run jeśli nie jest aktywne
gcloud services enable run.googleapis.com --project $PROJECT_ID

# Skopiuj Dockerfile z podfolderu do katalogu głównego
cp docker/Dockerfile Dockerfile
# Buduj obraz z katalogu głównego
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME --project $PROJECT_ID --timeout=30m .

# Deployuj na Cloud Run
gcloud run deploy $SERVICE_NAME \
    --image gcr.io/$PROJECT_ID/$SERVICE_NAME \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated

echo "Deployment zakończony! Sprawdź adres usługi w konsoli Cloud Run."
