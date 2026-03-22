from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from scripts.prediction_engine import GrievancePredictionEngine
import uvicorn
import os

app = FastAPI()

# Enable CORS for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize prediction engine
try:
    engine = GrievancePredictionEngine(model_dir='models')
except Exception as e:
    print(f"Error initializing prediction engine: {e}")
    engine = None

class PredictionRequest(BaseModel):
    text: str
    department: str = "Municipal Services"

@app.get("/")
async def root():
    return {"message": "Grievance Classification API", "status": "ready"}

@app.post("/predict")
async def predict_grievance(request: PredictionRequest):
    if engine is None:
        raise HTTPException(status_code=500, detail="Prediction engine not initialized. Ensure models are trained.")
    
    try:
        result = engine.predict(request.text, department=request.department)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
