
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# define a root `/` endpoint
@app.get("/")
def index():
    return {"ok": True}


# Implement a /predict endpoint
@app.get("/predict")
def predict(title, 
            original_title, 
            release_date, 
            duration_min, 
            description, 
            budget, 
            original_language, 
            status, 
            number_of_awards_won, 
            number_of_nominations, 
            has_collection, 
            all_genres, 
            top_countries, 
            number_of_top_productions, 
            available_in_english):
    
    dico = {'original_title': [original_title],
            'title': [title],
            'release_date': [release_date],
            'duration_min': [float(duration_min)],
            'description': [description], 
            'budget': [float(budget)],
            'original_language' : [original_language],
            'status': [status],
            'number_of_awards_won': [int(number_of_awards_won)],
            'number_of_nominations' : [int(number_of_nominations)],
            'has_collection': [int(has_collection)],
            'all_genres': [all_genres],
            'top_countries': [top_countries], 
            'number_of_top_productions': [float(number_of_top_productions)],
            'available_in_english': [bool(available_in_english)]
            }
 
    X_pred = pd.DataFrame(data = dico)
    model = joblib.load('model.joblib')
    popularity = model.predict(X_pred)
    return {title: 'title',
            'popularity':popularity}
