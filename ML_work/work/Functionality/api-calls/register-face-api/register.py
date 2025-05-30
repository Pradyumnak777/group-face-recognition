# import sys
# sys.path.insert(1, 'ML_work/our_method/Functionality/api-calls/register-face-api')
# for id,name in enumerate(sys.path):
#     print(f"index:{id}, name:{name}")
    
from g2f import group2head
import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from typing import List
from PIL import Image
import io
# from itertools import islice

# print(dict(islice(sys.modules.items(),4)))

# if 'g2f' in sys.modules:
#     print(f'''g2f.py imported successfully!''')
# else:
#     print("g2f not imported.")

app = FastAPI()
@app.post("/register-student")
async def upload_imgs(files: List[UploadFile] = File(...), 
                      student_name: str = Form(...),
                      group_name: str = Form(...)):
    #now, save this group as a dataset directory(like LFW) after head_segmentation
    os.makedirs(f'demo/{group_name}/{student_name}', exist_ok = True)
    cur_dir = f'demo/{group_name}/{student_name}'
    for num,file in enumerate(files):
        contents = await file.read()
        #now- perform - face detection(1) and head segmentation(2)
        output_data = group2head(contents)
        for i, img in enumerate(output_data):
            img.save(f"{cur_dir}/{student_name}_{num}.jpg")
    
    return {"message": "Student registered"} 
            

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)