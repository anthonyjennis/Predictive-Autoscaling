import hashlib
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    data = b"x"
    for _ in range(20000):
        data = hashlib.sha256(data).digest()
    return {"ok": True}

@app.get("/health")
def health():
    return {"status": "up"}
