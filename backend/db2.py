# db2.py

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
from pymongo import MongoClient

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



# New endpoint to save edited patient details into 'records' collection
@router.post("/save_to_records")
async def save_to_records(record: PatientRecord):
    print("Received record:", record.dict())  # Debug print
    result = records_collection.insert_one(record.dict())
    return {"message": "Record saved to 'records' collection", "id": str(result.inserted_id)}
