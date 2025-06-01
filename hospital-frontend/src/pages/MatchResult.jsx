import React from "react";
import { useNavigate } from "react-router-dom";

const MatchResult = ({ match }) => {
  const navigate = useNavigate();

  const handleViewDetails = () => {
    if (match && match._id) {
      navigate(`/patient/${match._id}`);
    } else {
      alert("No valid patient ID");
    }
  };

  return (
    <div>
      <h2>🎯 Match Found</h2>
      <button onClick={handleViewDetails}>View Patient Details</button>
    </div>
  );
};

export default MatchResult;
