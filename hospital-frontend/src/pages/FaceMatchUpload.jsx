import React, { useRef, useState, useEffect } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";

const FaceMatchUpload = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    startCamera();
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      console.error("Camera error:", err);
      setError("Failed to access camera.");
    }
  };

  const captureAndSubmit = () => {
    setError(null);
    const context = canvasRef.current.getContext("2d");
    context.drawImage(videoRef.current, 0, 0, 300, 200);

    canvasRef.current.toBlob(async (blob) => {
      if (!blob) {
        setError("Failed to capture image.");
        return;
      }

      const imageUrl = URL.createObjectURL(blob);
      setImagePreview(imageUrl);

      const formData = new FormData();
      formData.append("file", blob, "capture.jpg");

      await handleFaceMatch(formData);
    }, "image/jpeg");
  };

const handleFaceMatch = async (formData) => {
  try {
    const response = await axios.post("http://localhost:8000/match_face/", formData);
    const data = response.data;

    console.log("Backend response:", data); // ✅ Debug log

    if (data && data.status === "matched" && data.medical_id) {
      navigate(`/patient-details/${data.medical_id}`, {
        state: { patientData: data },
      });
    } else {
      navigate("/patient-details", {
        state: {
          data: {
            match: false,
            message: "No matching patient found.",
          },
        },
      });
    }
  } catch (err) {
    console.error("Face match error:", err);
    navigate("/patient-details", {
      state: {
        data: {
          match: false,
          message: err.response?.data?.detail || "Error during face match.",
        },
      },
    });
  }
};


  return (
    <div style={{ fontFamily: "Arial", padding: "20px" }}>
      <h2>🎥 Face Match Upload</h2>
      <video ref={videoRef} width="300" height="200" autoPlay />
      <br />
      <button onClick={captureAndSubmit}>Capture & Submit</button>

      <canvas ref={canvasRef} width="300" height="200" style={{ display: "none" }} />

      {imagePreview && (
        <div>
          <h3>🖼️ Captured Image:</h3>
          <img src={imagePreview} alt="Captured" width="300" />
        </div>
      )}

      {error && (
        <div style={{ marginTop: "20px", color: "red" }}>
          <h3>❌ Error:</h3>
          <p>{error}</p>
        </div>
      )}
    </div>
  );
};

export default FaceMatchUpload;
