# MindRAG Frontend

This is the frontend for the MindRAG application, a Retrieval-Augmented Generation system.

## Quick Start for Windows

1. Make sure Node.js is properly installed (v18+ recommended)
2. Double-click on `start-app.bat` to start the development server
3. Open your browser to http://localhost:3000

## Features

- Document upload and processing
- Question answering based on uploaded documents
- Dark/light mode support
- Modern, responsive UI with Chakra UI

## Technology Stack

- React 18
- Create React App
- Chakra UI for components
- Axios for API requests

## Backend Connection

The frontend is configured to connect to the backend at http://localhost:8000 through a proxy. All API requests will be automatically routed to the backend.

## Troubleshooting

If you encounter any issues:

1. Make sure you're using a compatible Node.js version (v18+)
2. Try reinstalling the dependencies:
   ```
   npm install
   ```
3. If you need to create a fresh installation:
   ```
   npm install --legacy-peer-deps
   ```

## Development Commands

- Start development server:
  ```
  npm start
  ```
  or
  ```
  npm run dev
  ```
- Build for production:
  ```
  npm run build
  ```
- Run tests:
  ```
  npm test
  ```