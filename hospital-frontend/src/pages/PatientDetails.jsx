import React, { useState, useEffect } from "react";
import { useParams } from "react-router-dom";
import "../css/PatientDetails.css";

const PatientDetails = () => {
  const { id } = useParams();
  const [formData, setFormData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchPatient = async () => {
      try {
        const res = await fetch(`http://localhost:8000/get_patient/${id}`);
        const data = await res.json();

        if (res.ok) {
          setFormData(data);
        } else {
          alert(data.message || "Failed to load patient.");
        }
      } catch (err) {
        console.error("Error fetching patient:", err);
        alert("Server error while fetching patient.");
      } finally {
        setLoading(false);
      }
    };

    if (id) {
      fetchPatient();
    } else {
      alert("Invalid patient ID");
      setLoading(false);
    }
  }, [id]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

const handleSubmit = async (e) => {
  e.preventDefault();
  try {
    // Step 1: Save initial patient details
    const res = await fetch("http://localhost:8000/save_to_records", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(formData),
    });
    const result = await res.json();

    if (!res.ok) {
      throw new Error(result.message || "Failed to save changes.");
    }

    alert(result.message || "Saved successfully!");

    // Step 2: Call prediction API
    const predictRes = await fetch(`http://localhost:8000/predict/${formData._id}`);
    const predictData = await predictRes.json();

    if (!predictRes.ok) {
      alert(predictData.detail || "Prediction failed.");
      return;
    }

    alert(`✅ Predicted Visit Type: ${predictData.predicted_visit_type}`);

    // Step 3: Add prediction result to formData
    const updatedFormData = {
      ...formData,
      predicted_visit_type: predictData.predicted_visit_type,
    };

    // Step 4: Save the updated data (with prediction) again to the DB
    const savePredicted = await fetch("http://localhost:8000/save_to_records", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(updatedFormData),
    });

    const finalSaveResult = await savePredicted.json();

    if (!savePredicted.ok) {
      alert(finalSaveResult.message || "Failed to save prediction.");
    } else {
      alert("📁 Final patient data with prediction saved to records database.");
    }

  } catch (err) {
    console.error("Error:", err);
    alert("An error occurred during saving or prediction.");
  }
};


  if (loading) return <p>Loading patient data...</p>;
  if (!formData) return <p>No patient data found.</p>;

  return (
    <div className="patient-container">
      <h2>✏️ Edit Patient Details</h2>

      <form onSubmit={handleSubmit} className="patient-form">
        {Object.entries(formData).map(([key, value]) =>
          key !== "picture" ? (
            <div className="form-group" key={key}>
              <label>{key === "_id" ? "Patient ID" : key.replace(/_/g, " ")}:</label>
              <input
                type="text"
                name={key}
                value={value}
                onChange={handleChange}
                readOnly={key === "_id"}
              />
            </div>
          ) : null
        )}

       {formData.image_base64 && (
  <div className="picture-preview">
    <p>
      <strong>Picture Preview:</strong>
    </p>
    <img
      src={`data:image/jpeg;base64,${formData.image_base64}`}
      alt="Patient"
    />
  </div>
)}


        <button type="submit" className="submit-btn">
          💾 Save Changes
        </button>
      </form>
    </div>
  );
};

export default PatientDetails;
