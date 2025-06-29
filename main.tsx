import React from 'react';
import ReactDOM from 'react-dom/client';
import WebGPUNeuralAutomata from './src/client/app';

ReactDOM
  .createRoot(document.getElementById('root')!)
  .render(<React.StrictMode><WebGPUNeuralAutomata /></React.StrictMode>);