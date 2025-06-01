import React from 'react';
import { Link } from 'react-router-dom';
import '../css/navbar.css';

const Navbar = () => {
  return (
    <header className="navbar">
      <div className="navbar-left">
        <img src="/logo.png" alt="Mediox Logo" className="logo" />
        <span className="brand-name">Mediox</span>
      </div>

      <nav className="navbar-links">
        <Link to="/">Home</Link>
        <Link to="/about">About Us</Link>
        <Link to="/services">Services</Link>
        <Link to="/pages">Pages</Link>
        <Link to="/news">News</Link>
        <Link to="/contact">Contact</Link>
        <Link to="/match-face">Face Match</Link> {/* ✅ Added link */}
      </nav>

      <div className="navbar-actions">
        <div className="call-box">
          <span className="phone-icon">📞</span>
          <div className="call-info">
            <small>Call Emergency</small>
            <strong>+88 0123 654 99</strong>
          </div>
        </div>
        <Link to="/appointment" className="appointment-btn">Make An Appointment →</Link>
      </div>
    </header>
  );
};

export default Navbar;
