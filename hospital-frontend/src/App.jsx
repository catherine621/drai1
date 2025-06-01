import './App.css';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/layout';
import UserForm from './pages/newpatientform';
import FaceMatchUpload from './pages/FaceMatchUpload';
import PatientDetails from './pages/PatientDetails';

function App() {
  return (
    <Router>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<UserForm />} />
          <Route path="/match-face" element={<FaceMatchUpload />} />
          <Route path="/get_patient/:id" element={<PatientDetails />} />
          <Route path="/patient-details/:id" element={<PatientDetails />} />

          {/* Optional route without ID */}
          <Route path="/patient-details" element={<PatientDetails />} />
        </Route>
      </Routes>
    </Router>
  );
}

export default App;
