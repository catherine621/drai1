from pydantic import BaseModel

class Patient(BaseModel):
    name: str
    dob: str
    address: str
    phone_number:str
    email_id:str
    height: str
    weight: str
    blood_pressure: str
    image_base64: str
