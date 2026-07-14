from fastapi import FastAPI

app = FastAPI(
    title="HyperVision KYC AI",
    description="AI-Powered Document Segmentation & Preprocessing API",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "HyperVision KYC API is running"}
