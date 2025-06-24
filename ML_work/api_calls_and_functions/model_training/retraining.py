from fastapi import FastAPI, HTTPException, File, Form


app = FastAPI()
@app.post('/retrain')
async def