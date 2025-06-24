FROM python:3.10-slim

# setting working dir of app
WORKDIR /app

# copying the files needed


RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    python-multipart \
    fastapi \           
    uvicorn \          
    pillow \           
    ultralytics \       
    torch \             
    torchvision \     
    opencv-python \
    git+https://github.com/wiktorlazarski/head-segmentation.git

RUN pip install --no-cache-dir boto3

#here, '/app/' just uses the 'register.py' file name as it is and stores it in the /app/ directory
COPY register.py /app/ 
COPY g2f.py /app/
COPY preprocessed /app/preprocessed


# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "register:app", "--host", "0.0.0.0", "--port", "8000"]