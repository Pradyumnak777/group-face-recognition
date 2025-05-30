import os
import fastapi
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from g2f import group2head
from io import BytesIO
import io
from PIL import Image
from starlette.requests import Request

# Ensure the 'static' directory exists
static_directory = "static"
if not os.path.exists(static_directory):
    os.makedirs(static_directory)

app = FastAPI()

templates = Jinja2Templates(directory="/face_attendance/api-calls/detection-segmentation-api/templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/process-image")
async def process_grp_image(request: Request, image_file: UploadFile = File(...)):
    try:
        image_data = await image_file.read()
        if len(os.listdir('/face_attendance/api-calls/detection-segmentation-api/static')) != 0:
            for file in os.listdir('/face_attendance/api-calls/detection-segmentation-api/static'):
                os.remove(os.path.join(f'/face_attendance/api-calls/detection-segmentation-api/static',file))
        output_data = group2head(image_data)
        
        image_urls = []
        #clear static directory
        # if os.path.exists('/interns/iittcseitr24_10/face_attendance/group2face/runs/detect'):
        for i, img in enumerate(output_data):
            img_bytes = io.BytesIO()  # Convert each image into byte format
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            prop_img = Image.open(BytesIO(img_bytes.getvalue()))
            image_url = f'/face_attendance/api-calls/detection-segmentation-api/static/processed_image_{i}.jpg'
            prop_img.save(image_url,'JPEG')
            image_urls.append(image_url)
        
        return templates.TemplateResponse("index.html", {"request": request, "image_urls": image_urls})
    
    except HTTPException as http_err:
        raise http_err
    
    except Exception as e:
        return {"error": str(e)}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "image_urls": None})

@app.get("/{path:path}")
async def catch_all(path: str):
    return {"error": f"The path {path} does not exist"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)