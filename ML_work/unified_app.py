from fastapi import FastAPI

from api_calls_and_functions.register_face_api.register import app as register_app
from api_calls_and_functions.mark_person_api.test import app as test_app

app = FastAPI()

app.mount("/register", register_app)
app.mount("/mark", test_app)

