import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import './styles/globals.css';

const saved = localStorage.getItem('theme'); 
const prefersDark = window.matchMedia?.('(prefers-color-scheme: dark)')?.matches;
const shouldDark = saved === 'dark' || (!saved && prefersDark);
document.documentElement.classList.toggle('dark', shouldDark);

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)