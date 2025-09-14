from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Dict

{
  "emotion": "happy",
  "weather_condition": "Ensoleillé",
  "temperature": 24,
  "preferred_style": "casual",
  "preferred_preference": "modern"
}
{
  "emotion": "happy",
  "weather_condition": "Ensoleillé",
  "temperature": 24,
  "preferred_style": "casual",
  "preferred_preference": "modern"
}
from settings import STATIC_ROUTE, STATIC_DIR
from recommender import recommend, IMG_DIR

app = FastAPI(title="Outfit Recommender", version="1.0")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount(STATIC_ROUTE, StaticFiles(directory=str(STATIC_DIR)), name="images")

class ContextIn(BaseModel):
    emotion: str
    weather_condition: str
    temperature: float
    preferred_style: str = Field(default="casual")
    preferred_preference: str = Field(default="modern")

class ItemOut(BaseModel):
    item_name: str
    image_url: str

@app.post("/recommend", response_model=Dict[str, ItemOut])
def recommend_endpoint(ctx: ContextIn):
    result = recommend(ctx.model_dump())
    
    out = {}
    for cat, v in result.items():
        
        path = str(v["image_path"])
        # ensure it's under IMG_DIR
        rel = path.split(str(IMG_DIR))[-1].lstrip("/\\")
        out[cat] = {
            "item_name": v["item_name"],
            "image_url": f"{STATIC_ROUTE}/{rel.replace('\\', '/')}"
        }
    return out

@app.get("/")
def root():
    return {"ok": True, "message": "Outfit recommender ready"}
