# Face recognition system (Demo version)

This system is intended to be an on the fly attendance system via a mobile applicatioon. It is designed to handle/perform inference on group photos taken in real world/mildly uncontrollable settings.

### Features

- **One-shot Registration** – A single image is captured during registration. This is augmented to create the positive set.
- **Face Embedding + Custom Model** – Embeddings are extracted using a pre-trained FaceNet model(on vggface) and are then refined using a custom neural head trained with triplet loss, during the registration process.
- **Incremental Learning** – The head is retrained with every new registration to adapt to the expanding pool of users.
- **Two-stage Inference** – As the custom model is naturally poor at eliminating **_unregistered faces_** (as it is only trained with registered people), the embedddings from facenet are first used to filter out potential unregistered candidates(via threshlding). Thehn, the face is passed to the custom model- which is better at discriminating between known faces.

### Brief Workflow

1. **Registration**

   - One image of a new user is captured. _Head segmentation_ is then performed on this image to eliminate the residual background
   - The new image is augmented into multiple views.
   - These are passed through FaceNet to generate 512D embeddings.
   - If another registration exists (if not, it will predict using the facenet embeddings), the cutsom model (trained using triplet loss) maps these embeddings to a lower-dimensional space (64D).
   - This head is incrementally updated to support new registrations.

2. **Attendance Detection**
   - A group photo is submitted.
   - Faces are detected and segmented.
   - Each face is embedded via FaceNet and compared (cosine similarity) with registered users.
   - If similarity passes a threshold, the embedding is passed through the trained head for final classification.

### How to run

- Clone this repo and install the required libraries through the .yaml file.
- go to [this drive link](https://drive.google.com/drive/folders/1IhGQD2WDNha04nj15gO_rgo_WKZPP8sC?usp=sharing) and install this folder, containing the pretrained YOLO model, head_segmentation model and facenet. Place this folder under `ML_work\api_calls_and_functions\shared_material`
- **!!IMPORTANT**- This application primarily uses Amazon S3 for storage and retrieval. It will NOT work unless you make a `.env` file specifying: `AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME` under the `ML_work` directory.
- Navigate to `unified_app.py` and run it. Then via the terminal, run the following command: `uvicorn unified_app:app --host 0.0.0.0 --port 8000` (to run on the local host)
- From the client machine, make a POST request to `/register/register-student` to register a student and `/mark/detect` for the inference API. A simple flutter frontend is provided in this repo. You can create an apk, enable a firewall rule to allow private network traffic through port 8000, and test it wirelessly from your android mobile phone. Please note that you would have to change the API's IP in `face_attendance_app\lib\pages\detect_page.dart` and `face_attendance_app\lib\pages\register_page.dart`

#### Running via docker

If you would like a more hasslefree imlpementation and do not want to inspect/customize the code currently, you can clone the docker image from dockerhub to use the APIs: `docker pull pradyumnak/face-recog-api:latest`. Please note that you have to specify the environment variables again.

- Create a `.env` file in the same directory where `docker run` would be executed. Then run `docker run --env-file .env pradyumnak/face-recog-api:latest`
- Alternatively, pass the environment variables directly.

```
docker run \
-e AWS_ACCESS_KEY_ID=your-access-key \
-e AWS_SECRET_ACCESS_KEY=your-secret-key \
-e S3_BUCKET_NAME=you-bucket-name \
pradyumnak/face-recog-api:latest
```

### Notes and Highlights

- YOLO has been used to perform face detection
- [head-segmentation by Wiktor Łazarski](https://github.com/wiktorlazarski/head-segmentation) has been used to segment heads.
- InceptionResnetV1 has been used to generate the 512d embeddings (facenet).
- Custom trained triplet loss based model has been used to generate 64d embeddings (from 512d).

## Aditional Info

- This project is purely an experimental, learning focussed attempt at building a facial recognition system
- Testing has been conducted in limited, controlled environments and **not** on standardized public datasets.
- Optimizations in inference speed and a more intuitive frontened attendance system (groups, subjects, etc.) are planned in the future.
