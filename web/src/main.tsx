import React from 'react';
import ReactDOM from 'react-dom/client';
import './style.css';
import App from './App';
// Design layers load last so their scoped overrides win ties with component CSS.
import './designs/registry';
ReactDOM.createRoot(document.getElementById('root')!).render(<React.StrictMode><App/></React.StrictMode>);
