import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, HTMLResponse
from io import BytesIO
import io
from PIL import Image
from starlette.requests import Request
import base64
from fastapi.responses import JSONResponse

app = FastAPI()

@app.post("/register-face")
async def process_grp_image(
    request: Request,
    image_files: list[UploadFile] = File(...),
    name: str = Form(...)
):
    try:
        image_bytes_list = []
        
        for image_file in image_files:
            image_data = await image_file.read()
            image_bytes_list.append(image_data)
 
        #send this list of img_byte data to another function
        
        
        # result = JSONResponse(content=images_byte_stream)
        # return result
    
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
