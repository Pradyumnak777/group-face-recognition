import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, HTMLResponse
from g2f import group2head
from io import BytesIO
import io
from PIL import Image
from starlette.requests import Request
import base64
from fastapi.responses import JSONResponse

# # Ensure the 'static' directory exists
# static_directory = "static"
# if not os.path.exists(static_directory):
#     os.makedirs(static_directory)

app = FastAPI()

# templates = Jinja2Templates(directory="/interns/iittcseitr24_10/face_attendance/api-calls/templates")

# app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/process-image")
async def process_grp_image(request: Request, image_file: UploadFile = File(...)):
    try:
        image_data = await image_file.read()
        # if len(os.listdir('/interns/iittcseitr24_10/face_attendance/api-calls/static')) != 0:
        #     for file in os.listdir('/interns/iittcseitr24_10/face_attendance/api-calls/static'):
        #         os.remove(os.path.join(f'/interns/iittcseitr24_10/face_attendance/api-calls/static',file))
        output_data = group2head(image_data)
        
        images_byte_stream = []
        #clear static directory
        # if os.path.exists('/interns/iittcseitr24_10/face_attendance/group2face/runs/detect'):
        for img in output_data:
            img_bytes = io.BytesIO()  # Convert each image into byte format
            img.save(img_bytes, format='JPEG')
            img_bytes.seek(0)
            # prop_img = Image.open(BytesIO(img_bytes.getvalue()))
            # image_url = f'/interns/iittcseitr24_10/face_attendance/api-calls/static/processed_image_{i}.jpg'
            # prop_img.save(image_url,'JPEG')
            
            #convert JPEG bytes to encoded format to enable serialization
            encoded_img = base64.b64encode(img_bytes.read()).decode('utf-8')
            images_byte_stream.append(encoded_img)
        
        
        result = JSONResponse(content=images_byte_stream)
        return result
    
        # return templates.TemplateResponse("index.html", {"request": request, "image_urls": image_urls})
    
    except HTTPException as http_err:
        raise http_err
    
    except Exception as e:
        return {"error": str(e)}

# @app.get("/", response_class=HTMLResponse)
# async def read_root(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request, "image_urls": None})

# @app.get("/{path:path}")
# async def catch_all(path: str):
#     return {"error": f"The path {path} does not exist"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
