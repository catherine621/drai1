import React from 'react';
import Navbar from './navbar';
import { Outlet } from 'react-router-dom';

const Layout = () => {
  return (
    <>
      <Navbar />
      <main style={{ padding: '0px' }}>
        <Outlet />
      </main>
    </>
  );
};

export default Layout;
