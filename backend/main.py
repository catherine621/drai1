from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from bson.objectid import ObjectId
from database import patients_collection
from pydantic import BaseModel
from twilio.rest import Client
import pandas as pd
import base64
import requests
import re
import os
import traceback
from dotenv import load_dotenv
from fastapi import Body
from bson import ObjectId
from datetime import datetime
import face_recognition
import numpy as np
import cv2
from io import BytesIO
from PIL import Image, UnidentifiedImageError
from database import patients_collection
from db2 import router as db2_router  # Import the router from db2
import io
import faiss
from fastapi import APIRouter
import json
from concurrent.futures import ThreadPoolExecutor
import asyncio
<<<<<<< HEAD
from flask import Flask, jsonify 
import h5py
=======
from flask import Flask, jsonify
from ultralytics import YOLO
import logging
>>>>>>> 38f614d (Updated hospital frontend files)

# Load environment variables

load_dotenv()

app = FastAPI()
router = APIRouter()


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(db2_router)


# Twilio setup
account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
twilio_number = os.getenv("TWILIO_PHONE_NUMBER")
twilio_client = Client(account_sid, auth_token)

# Patient Model for prediction
class Patient(BaseModel):
    age: int
    gender: str
    admissionheight: int
    admissionweight: int
    unitvisitnumber: int
    apacheadmissiondx: str

# Clean dataset for few-shot prompting
df = pd.read_csv("EHR_cleaned.csv")

def to_float(val):
    try:
        return float(val)
    except:
        return None

df["admissionheight"] = df["admissionheight"].apply(to_float)
df["admissionweight"] = df["admissionweight"].apply(to_float)
df["unitvisitnumber"] = df["unitvisitnumber"].apply(to_float)

df = df.dropna(subset=["age", "admissionheight", "admissionweight", "unitvisitnumber", "apacheadmissiondx", "gender", "visittype"])

# Few-shot prompt generation
def row_to_prompt(row):
    return f"""Patient:
- Age: {int(row['age'])}
- Gender: {row['gender']}
- Height: {int(row['admissionheight'])} cm
- Weight: {int(row['admissionweight'])} kg
- Unit Visit Number: {int(row['unitvisitnumber'])}
- Diagnosis: {row['apacheadmissiondx']}
Visit Type: {row['visittype']}"""

few_shot_examples = "\n\n".join([row_to_prompt(row) for _, row in df.sample(5, random_state=42).iterrows()])

def build_prompt(data: Patient):
    return f"""You are a medical assistant. Based on the patient's information, predict the hospital visit type (Emergency, Follow-up, or Regular) without any reasoning.

{few_shot_examples}

Patient:
- Age: {data.age}
- Gender: {data.gender}
- Height: {data.admissionheight} cm
- Weight: {data.admissionweight} kg
- Unit Visit Number: {data.unitvisitnumber}
- Diagnosis: {data.apacheadmissiondx}
Visit Type:"""

def call_groq_api(prompt: str, api_key: str):
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "llama3-70b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 50,
        },
    )
    if response.status_code != 200:
        raise Exception(f"Groq API Error: {response.status_code}, {response.text}")
    return response.json()

def parse_response(result_json):
    content = result_json["choices"][0]["message"]["content"].strip()
    match = re.search(r"visit type to be: (\w+)", content, re.IGNORECASE)
    return match.group(1) if match else content

@app.get("/")
def read_root():
    return {"message": "FastAPI server is running!"}

@app.get("/api/patients")
async def get_patients():
    patients = []
    async for patient in patients_collection.find():
        patient["medical_id"] = str(patient["_id"])
        patient.pop("_id", None)
        patients.append(patient)
    return patients






def get_valid_int(doc, field_name):
    value = doc.get(field_name, "")
    if value is None or str(value).strip() == "":
        raise ValueError(f"{field_name} is missing or empty.")
    return int(value)



@app.get("/predict/{patient_id}")
async def predict_visit_type_from_db(patient_id: str):
    try:
        # 1. Fetch patient document
        patient_doc = await patients_collection.find_one({"_id": ObjectId(patient_id)})
        if not patient_doc:
            raise HTTPException(status_code=404, detail="Patient not found")

        # 2. Parse and validate all required fields
        try:
            patient = Patient(
                age=get_valid_int(patient_doc, "age"),
                gender=str(patient_doc.get("gender", "")).strip(),
                admissionheight=get_valid_int(patient_doc, "admissionheight"),
                admissionweight=get_valid_int(patient_doc, "admissionweight"),
                unitvisitnumber=get_valid_int(patient_doc, "unitvisitnumber"),
                apacheadmissiondx=str(patient_doc.get("apacheadmissiondx", "")).strip()
            )
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))

        # 3. Build prompt & call model
        prompt = build_prompt(patient)
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        result = call_groq_api(prompt, GROQ_API_KEY)
        prediction = parse_response(result)

        return {
            "patient_details": patient.dict(),
            "predicted_visit_type": prediction
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
















metadata = []
faiss_index = None

ENCODINGS_PATH = "encodings.h5"
#ENCODINGS_PATH = "encodings.npy"
METADATA_PATH = "metadata.json"

<<<<<<< HEAD
=======
async def build_and_save_index():
    global faiss_index, metadata

    print("🔧 Starting to build FAISS index...")

    metadata.clear()
    encodings = []
    


    patients = await patients_collection.find().to_list(length=100)
    print(f"📦 Found {len(patients)} patients in MongoDB")

    for patient in patients:
        try:
            encoding = patient.get("face_embedding")
            if encoding and isinstance(encoding, list) and len(encoding) == 128:
                encodings.append(np.array(encoding, dtype=np.float32))
                metadata.append({
                    "name": patient.get("name", "Unknown"),
                    "medical_id": str(patient.get("_id"))
                })
                print(f"✅ Valid encoding for {patient.get('name')}")
            else:
                print(f"⚠ Invalid embedding for {patient.get('name')}: {encoding}")
        except Exception as e:
            print(f"❌ Error processing patient: {e}")
            continue

    if encodings:
        encodings_np = np.array(encodings).astype("float32")
        faiss.normalize_L2(encodings_np)

        np.save(ENCODINGS_PATH, encodings_np)
        with open(METADATA_PATH, "w") as f:
            json.dump(metadata, f)

        dim = 128
        faiss_index = faiss.IndexFlatIP(dim)
        faiss_index.add(encodings_np)
        print(f"✅ FAISS index built with {len(encodings)} encodings.")
    else:
        print("❌ No valid face encodings found. metadata.json will be empty.")
@app.on_event("startup")
async def startup_event():
    await load_index_from_disk()
    asyncio.create_task(poll_patient_updates())  # Polls for new patients periodically

>>>>>>> 38f614d (Updated hospital frontend files)
# -------------------------------
# Load FAISS Index
# -------------------------------
async def load_index_from_disk():
    global faiss_index, metadata

    if not os.path.exists(ENCODINGS_PATH) or not os.path.exists(METADATA_PATH):
<<<<<<< HEAD
        raise FileNotFoundError("Encodings or metadata files are missing.")

=======
        print("⚠ Missing index files. Building fresh index...")
        await build_and_save_index()
        return
>>>>>>> 38f614d (Updated hospital frontend files)

    try:
        with h5py.File(ENCODINGS_PATH, "r") as f:
            encodings_np = f["encodings"][:]
        faiss.normalize_L2(encodings_np)  # Normalize for IndexFlatIP
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)

        faiss_index = faiss.IndexFlatIP(128)
        faiss.normalize_L2(encodings_np)
        faiss_index.add(encodings_np)
        print("✅ FAISS index and metadata loaded from disk.")
    except Exception as e:
        print(f"❌ Failed to load index: {e}")
        await build_and_save_index()

async def watch_patient_changes():
    print("👀 Starting MongoDB watch...")
    async with patients_collection.watch(
        [{"$match": {"operationType": "insert"}}],
        full_document="updateLookup"
    ) as stream:
        async for change in stream:
            try:
                new_doc = change.get("fullDocument")
                print(f"📥 New patient inserted: {new_doc.get('name')}")

                encoding = new_doc.get("face_embedding")
                if encoding and isinstance(encoding, list) and len(encoding) == 128:
                    encoding_np = np.array([encoding], dtype=np.float32)
                    faiss.normalize_L2(encoding_np)

                    # Add new encoding to index
                    if faiss_index is None:
                        print("⚠ FAISS index not initialized. Rebuilding index...")
                        await build_and_save_index()
                        return

                    faiss_index.add(encoding_np)

                    # Append metadata and save to file
                    new_meta = {
                        "name": new_doc.get("name", "Unknown"),
                        "medical_id": str(new_doc.get("_id"))
                    }
                    metadata.append(new_meta)

                    # Save updated metadata
                    with open(METADATA_PATH, "w") as f:
                        json.dump(metadata, f)

                    # Append new encoding to encodings.npy
                    existing_encodings = np.load(ENCODINGS_PATH)
                    updated_encodings = np.vstack([existing_encodings, encoding_np])
                    np.save(ENCODINGS_PATH, updated_encodings)

                    print(f"✅ Index and metadata updated with new patient: {new_doc['name']}")
                else:
                    print(f"⚠ Skipped: Invalid face embedding for {new_doc.get('name')}")

            except Exception as e:
                print(f"❌ Error in watch_patient_changes: {e}")
import aiohttp

async def poll_patient_updates(interval_seconds: int = 30):
    global faiss_index, metadata

    seen_ids = set(m["medical_id"] for m in metadata)
    print("🔁 Starting patient polling loop...")

    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:8000/api/patients") as response:
                    if response.status != 200:
                        print(f"⚠ Failed to fetch patients: {response.status}")
                        await asyncio.sleep(interval_seconds)
                        continue

                    new_patients = await response.json()

                    new_encodings = []
                    new_metadata = []

                    for patient in new_patients:
                        medical_id = str(patient.get("medical_id"))
                        if medical_id in seen_ids:
                            continue

                        encoding = patient.get("face_embedding")
                        if encoding and isinstance(encoding, list) and len(encoding) == 128:
                            encoding_np = np.array([encoding], dtype=np.float32)
                            faiss.normalize_L2(encoding_np)

                            if faiss_index is None:
                                print("⚠ FAISS index not initialized. Rebuilding index...")
                                await build_and_save_index()
                                break

                            faiss_index.add(encoding_np)

                            meta = {
                                "name": patient.get("name", "Unknown"),
                                "medical_id": medical_id
                            }

                            metadata.append(meta)
                            new_encodings.append(encoding_np)
                            new_metadata.append(meta)
                            seen_ids.add(medical_id)

                            print(f"✅ Added new patient to index: {patient.get('name')}")

                    # Save updated metadata and encodings if new were added
                    if new_encodings:
                        with open(METADATA_PATH, "w") as f:
                            json.dump(metadata, f)

                        existing_encodings = np.load(ENCODINGS_PATH)
                        updated_encodings = np.vstack([existing_encodings] + new_encodings)
                        np.save(ENCODINGS_PATH, updated_encodings)

        except Exception as e:
            print(f"❌ Polling error: {e}")

        await asyncio.sleep(interval_seconds)

# -------------------------------
# Match Face Endpoint
# -------------------------------

<<<<<<< HEAD
index_lock=asyncio.Lock()
=======

try:
    model = YOLO("yolov8n-face-lindevs.pt")  # Ensure this model file exists
    logging.info("✅ YOLO model loaded")
except Exception as e:
    logging.error(f"❌ Failed to load YOLO model: {e}")
    model = None

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

>>>>>>> 38f614d (Updated hospital frontend files)
@app.post("/match_face/")
async def match_face(file: UploadFile = File(...)):
    global faiss_index, metadata

    try:
        if model is None:
            raise HTTPException(status_code=500, detail="YOLO model not loaded")

        image_bytes = await file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        results = model(image_bgr)[0]
        if len(results.boxes) == 0:
            raise HTTPException(status_code=400, detail="No face detected in the image.")

        # Use the first detected face
        x1, y1, x2, y2 = map(int, results.boxes[0].xyxy[0])
        cropped_face = image_bgr[y1:y2, x1:x2]

        # Convert to RGB for face_recognition
        cropped_rgb = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(cropped_rgb)

        if not encodings:
            raise HTTPException(status_code=400, detail="Failed to encode the detected face.")

        uploaded_encoding = np.array(encodings[0], dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(uploaded_encoding)
        
        async with index_lock:
            if faiss_index is None or len(metadata) == 0 or faiss_index.ntotal == 0:
               await load_index_from_disk()

        print("✅ Performing FAISS search...")
        D, I = faiss_index.search(uploaded_encoding, k=1)

        similarity = D[0][0]
        index = I[0][0]

        print(f"🔎 Similarity: {similarity}")
        print(f"🔎 Index: {index}")

        if index < len(metadata):
            print(f"🔎 Closest Match: {metadata[index]}")

        if similarity > 0.9:  # Adjust threshold as needed
            match = metadata[index]
            return {
                "status": "matched",
                "name": match["name"]
            }

        return {"status": "not_found"}

    except HTTPException as http_exc:
        raise http_exc

    except Exception as e:
        logging.error(f"Error in match_face: {e}")
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": f"Exception occurred: {str(e)}"
        })
executor = ThreadPoolExecutor()



async def append_encoding_to_index(new_encoding, new_metadata):
    global faiss_index, metadata

    new_encoding = np.array(new_encoding, dtype=np.float32).reshape(1, 128)
    faiss.normalize_L2(new_encoding)
    
    # Append to HDF5 encodings file
    if not os.path.exists(ENCODINGS_PATH):
        with h5py.File(ENCODINGS_PATH, "w") as f:
            f.create_dataset(
                "encodings",
                data=new_encoding,
                maxshape=(None, 128),
                chunks=True,
                dtype='float32'
            )
    else:
        with h5py.File(ENCODINGS_PATH, "a") as f:
            ds = f["encodings"]
            current_len = ds.shape[0]
            ds.resize((current_len + 1, 128))
            ds[current_len] = new_encoding

    # Update metadata JSON
    if not os.path.exists(METADATA_PATH):
        metadata = []
    else:
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)
    metadata.append(new_metadata)

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f)

    # Update in-memory FAISS index
    if faiss_index is None or faiss_index.ntotal == 0:
        with h5py.File(ENCODINGS_PATH, "r") as f:
            all_encodings = f["encodings"][:]
        faiss_index = faiss.IndexFlatIP(128)
        faiss_index.add(all_encodings)
    else:
        faiss_index.add(new_encoding)

        


####Registering Patients
@app.post("/api/patients")
async def register_patient(
    name: str = Form(...),
    dob: str = Form(...),
    address: str = Form(...),
    phone_number: str = Form(...),
    email_id: str = Form(...),
    admissionheight: str = Form(...),
    admissionweight: str = Form(...),
    blood_pressure: str = Form(...),
    age: str = Form(...),
    gender: str = Form(...),
    unitvisitnumber: str = Form(...),
    apacheadmissiondx: str = Form(...),
    picture: UploadFile = File(...)
):
    try:
        # Read image and convert to base64
        image_bytes = await picture.read()
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')

        # Load image using PIL and convert to numpy array
        image = face_recognition.load_image_file(io.BytesIO(image_bytes))

        # Detect face and extract embeddings
        face_encodings = face_recognition.face_encodings(image)

        if not face_encodings:
            return JSONResponse(status_code=400, content={"error": "No face found in the image."})

        face_embedding = face_encodings[0].tolist()  # Convert NumPy array to list for MongoDB

        # Format phone number
        if not phone_number.startswith("+"):
            phone_number = "+91" + phone_number.lstrip("0")

        # Create the patient document
        patient_doc = {
            "name": name,
            "dob": dob,
            "address": address,
            "phone_number": phone_number,
            "email_id": email_id,
            "admissionheight": admissionheight,
            "admissionweight": admissionweight,
            "blood_pressure": blood_pressure,
            "age": age,
            "gender": gender,
            "unitvisitnumber": unitvisitnumber,
            "apacheadmissiondx": apacheadmissiondx,
            "image_base64": image_base64,
            "face_embedding": face_embedding
        }

        # Insert into MongoDB
        insert_result = await patients_collection.insert_one(patient_doc)
        
        #only append if encodoing exists 
        if face_embedding:
            await append_encoding_to_index(
                face_embedding,
                {"name":name}
            )
            
        # Send SMS using Twilio
        message = twilio_client.messages.create(
            body=f"Hi {name}, you are successfully registered at the hospital.",
            from_=twilio_number,
            to=phone_number
        )

        return {
            "message": "Patient registered successfully",
            "sms_sid": message.sid,
            "patient_id": str(insert_result.inserted_id)
        }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})




executor = ThreadPoolExecutor()



def blocking_get_patient(object_id):
    patient = patients_collection.find_one({"_id": object_id})
    print("🔍 blocking_get_patient result:", patient)
    return patient

# ✅ This wraps the blocking function in a thread
async def get_patient_from_db(object_id):
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(executor, blocking_get_patient, object_id)
    print("✅ get_patient_from_db result:", result)
    return result


@app.get("/get_patient/{medical_id}")
async def get_patient(medical_id: str):
    from bson import ObjectId

    try:
        patient = await patients_collection.find_one({"_id": ObjectId(medical_id)})
    except Exception as e:
        print(f"⚠ Error converting to ObjectId: {e}")
        raise HTTPException(status_code=400, detail="Invalid medical ID format")

    if not patient:
        print("❌ Patient not found in DB")
        raise HTTPException(status_code=404, detail="Patient not found")

    patient["_id"] = str(patient["_id"])
    return patient














@app.post("/predict_and_notify/{patient_id}")
async def predict_and_notify(patient_id: str, reason_for_visit: str = Body(...)):
    try:
        # Convert and validate ObjectId
        try:
            patient_obj_id = ObjectId(patient_id)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid patient ID format")

        # Fetch patient document
        patient_doc = await patients_collection.find_one({"_id": patient_obj_id})
        if not patient_doc:
            raise HTTPException(status_code=404, detail="Patient not found")

        # Parse and validate fields
        try:
            patient = Patient(
                age=get_valid_int(patient_doc, "age"),
                gender=str(patient_doc.get("gender", "")).strip(),
                admissionheight=get_valid_int(patient_doc, "admissionheight"),
                admissionweight=get_valid_int(patient_doc, "admissionweight"),
                unitvisitnumber=get_valid_int(patient_doc, "unitvisitnumber"),
                apacheadmissiondx=str(patient_doc.get("apacheadmissiondx", "")).strip()
            )
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))

        # Build enhanced prompt
        prompt = build_prompt(patient) + f"\nAdditional Visit Reason: {reason_for_visit}\nVisit Type:"

        # Call GROQ API
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        result = call_groq_api(prompt, GROQ_API_KEY)
        prediction_raw = parse_response(result)
        prediction = prediction_raw.strip().capitalize()

        print(f"Prediction raw: '{prediction_raw}' → normalized: '{prediction}'")

        # Department contact mapping
        department_mapping = {
            "Emergency": "+919952560548",
            "Follow-up": "+919952560549",
            "Regular": "+919952560550"
        }

        contact_number = department_mapping.get(prediction)
        if not contact_number:
            raise HTTPException(status_code=400, detail=f"No contact number mapped for predicted type '{prediction}'")

        # Send SMS to department
        dept_message = twilio_client.messages.create(
            body=f"A new {prediction} patient has been registered. Patient: {patient_doc['name']}, Reason: {reason_for_visit}",
            from_=twilio_number,
            to=contact_number
        )

        return {
            "message": "Prediction made and department notified successfully",
            "predicted_visit_type": prediction,
            "department_contacted": contact_number,
            "department_sms_sid": dept_message.sid
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction and notification failed: {str(e)}")

@app.get("/check_notification/{patient_id}")
async def check_notification(patient_id: str):
    try:
        print(f"Received patient ID: {patient_id}")
        try:
            patient_obj_id = ObjectId(patient_id)
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid patient ID format")

        patient_doc = await patients_collection.find_one({"_id": patient_obj_id})
        if not patient_doc:
            raise HTTPException(status_code=404, detail="Patient not found")

        # Extract notification fields
        prediction = patient_doc.get("last_prediction")
        sms_sid = patient_doc.get("last_sms_sid")
        reason = patient_doc.get("last_reason", "Not provided")
        notified_at = patient_doc.get("notified_at")

        if not prediction or not sms_sid:
            return {
                "notified": False,
                "message": "No notification has been sent yet for this patient."
            }

        return {
            "notified": True,
            "predicted_visit_type": prediction,
            "reason_for_visit": reason,
            "twilio_message_sid": sms_sid,
            "notified_at": notified_at,
            "emergency_notified": prediction.lower() == "emergency"
        }

    except Exception as e:
        print("Check notification error:", str(e))
        raise HTTPException(status_code=500, detail="Failed to check notification status")