import React, { useState, useRef } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';
import '../css/newpatientform.css';

const UserForm = () => {
  const webcamRef = useRef(null);

  const [formData, setFormData] = useState({
  name: '',
  dob: '',
  address: '',
  phone_number: '',
  email_id: '',
  admissionheight: '',
  admissionweight: '',
  blood_pressure: '',
  picture: null,
  age: '',
  gender: '',
  unitvisitnumber: '',
  apacheadmissiondx: ''
});


  const [imagePreview, setImagePreview] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

 const captureImage = () => {
  const imageSrc = webcamRef.current.getScreenshot();
  setImagePreview(imageSrc); // Show preview
  fetch(imageSrc)
    .then(res => res.blob())
    .then(blob => {
      const file = new File([blob], 'webcam.jpg', { type: 'image/jpeg' });
      setFormData(prev => ({ ...prev, picture: file }));
    });
};

    const handleSubmit = async (e) => {
    e.preventDefault();
    const data = new FormData();

    for (const key in formData) {
      data.append(key, formData[key]);
    }

    try {
      await axios.post('http://localhost:8000/api/patients', data, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      alert('Patient registered successfully');
    } catch (error) {
      console.error('Registration failed:', error.response?.data || error.message);
      alert('Error registering patient');
    }
  };

  return (
    <div className="form-container">
      <h2>New Patient Registration</h2>
      <form onSubmit={handleSubmit} className="form-card">
        <div className="form-group">
          <label>Name</label>
          <input type="text" name="name" onChange={handleChange} required />
        </div>

        <div className="form-group">
          <label>Date of Birth</label>
          <input type="date" name="dob" onChange={handleChange} required />
        </div>

        <div className="form-group">
          <label>Address</label>
          <input type="text" name="address" onChange={handleChange} required />
        </div>

        <div className="form-group">
          <label>Phone Number</label>
          <input type="text" name="phone_number" onChange={handleChange} required />
        </div>

        <div className="form-group">
          <label>Email Id</label>
          <input type="text" name="email_id" onChange={handleChange} required />
        </div>

        <div className="form-group">
          <label>Height (in cm)</label>
          <input type="text" name="admissionheight" onChange={handleChange} required />
        </div>
        
        <div className="form-group">
          <label>Weight (in kg)</label>
          <input type="text" name="admissionweight" onChange={handleChange} required />
        </div>
        
        <div className="form-group">
          <label>Blood Pressure (in mmHg)</label>
          <input type="text" name="blood_pressure" onChange={handleChange} required />
        </div>

       <div className="form-group">
  <label>Age</label>
  <input type="text" name="age" onChange={handleChange} required />
</div>

<div className="form-group">
  <label>Gender</label>
  <select name="gender" onChange={handleChange} required>
    <option value="">Select</option>
    <option value="Male">Male</option>
    <option value="Female">Female</option>
    <option value="Other">Other</option>
  </select>
</div>

<div className="form-group">
  <label>Unit Visit Number</label>
  <input type="text" name="unitvisitnumber" onChange={handleChange} required />
</div>

<div className="form-group">
  <label>Diagnosis</label>
  <input type="text" name="apacheadmissiondx" onChange={handleChange} required />
</div>


        <div className="form-group">
          <label>Capture Photo</label>
          <Webcam ref={webcamRef} screenshotFormat="image/jpeg"/>
          <button type="button" className="capture-btn" onClick={captureImage}>Capture</button>

          {imagePreview && (
            <div className="image-preview">
              <p>Captured Image Preview:</p>
              <img src={imagePreview} alt="Preview" width="200" />
        </div>
  )}
        </div>

        <button type="submit" className="submit-btn">Register</button>
      </form>
    </div>
  );
};

export default UserForm;
