# db2.py

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from pymongo import MongoClient
from fastapi import APIRouter, Request
from bson import ObjectId
# MongoDB connection
client = MongoClient("mongodb://localhost:27017")
db = client["hospital_db"]
records_collection = db["records"]

# Define API router
router = APIRouter()

# Define the model for the incoming edited patient data
class PatientRecord(BaseModel):
    _id: str
    name: str
    dob: str
    age: str
    gender: str
    address: str
    phone_number: str
    email_id: str
    admissionheight: str
    admissionweight: str
    blood_pressure: str
    unitvisitnumber: str
    apacheadmissiondx: str
    picture: Optional[str] = None

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from pymongo import MongoClient
from database import patients_collection  # Import patients collection

client = MongoClient("mongodb://localhost:27017")
db = client["hospital_db"]
records_collection = db["records"]

router = APIRouter()



@router.post("/save_to_records")
async def save_to_records(request: Request):
    try:
        data = await request.json()
        print("Received record:", data)  # Debug

        # Attempt to convert _id to ObjectId if it's valid
        record_id = data.get("_id")
        if record_id:
            try:
                record_id = ObjectId(record_id)
            except:
                record_id = record_id  # Keep as-is if not a valid ObjectId

        record_data = {
            "name": data.get("name"),
            "dob": data.get("dob"),
            "age": data.get("age"),
            "gender": data.get("gender"),
            "address": data.get("address"),
            "phone_number": data.get("phone_number"),
            "email_id": data.get("email_id"),
            "admissionheight": data.get("admissionheight"),
            "admissionweight": data.get("admissionweight"),
            "blood_pressure": data.get("blood_pressure"),
            "unitvisitnumber": data.get("unitvisitnumber"),
            "apacheadmissiondx": data.get("apacheadmissiondx"),
            "predicted_visit_type": data.get("predicted_visit_type"),
            "image_base64": data.get("image_base64")
        }

        # Insert or update
        if isinstance(record_id, ObjectId):
            records_collection.replace_one({"_id": record_id}, record_data, upsert=True)
        else:
            records_collection.insert_one(record_data)

        return {"message": "Record saved successfully with prediction and picture."}

    except Exception as e:
        return {"message": f"Error saving record: {str(e)}"}