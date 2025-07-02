# Property Labeling Bot

A full-stack application that uses Google's Gemini Vision Language Model (VLM) to automatically label property images. The application consists of a React frontend for image upload and display, and a FastAPI backend for processing images through the Gemini API.

## Features

- **Image Upload**: Upload individual images or ZIP files containing multiple images
- **AI-Powered Labeling**: Uses Gemini VLM to automatically generate descriptive labels for property images
- **Grid Preview**: View labeled images in a responsive grid layout
- **Real-time Processing**: Process images and get results instantly
- **Modern UI**: Clean, responsive React interface

## Tech Stack

### Backend
- **FastAPI**: Modern Python web framework
- **Python 3.9+**: Core programming language
- **Uvicorn**: ASGI server for running FastAPI

### Frontend
- **React**: JavaScript library for building user interfaces
- **CSS3**: Styling and responsive design
- **HTML5**: Markup structure

## Project Structure

```
property-labeling-bot/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   ├── imagenet_classes.txt # Image classification classes
│   └── .env                 # Environment variables (not in repo)
├── frontend/
│   ├── src/
│   │   ├── App.js           # Main React component
│   │   ├── App.css          # Main styles
│   │   └── index.js         # React entry point
│   ├── public/
│   │   └── index.html       # HTML template
│   ├── package.json         # Node.js dependencies
│   └── README.md            # Frontend documentation
└── README.md                # This file
```

## Setup Instructions

### Prerequisites
- Python 3.9 or higher
- Node.js 14 or higher
- npm or yarn


### Backend Setup

1. **Navigate to backend directory**:
   ```bash
   cd property-labeling-bot/backend
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Start the backend server**:
   ```bash
   uvicorn main:app --reload --port 8000
   ```

### Frontend Setup

1. **Navigate to frontend directory**:
   ```bash
   cd property-labeling-bot/frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start the development server**:
   ```bash
   npm start
   ```

The application will be available at:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## Usage

1. **Upload Images**: Use the file upload button to select individual images or ZIP files
2. **Process Images**: The backend will send images to Gemini API for analysis
3. **View Results**: Labeled images will be displayed in a grid layout with their descriptions
4. **Download Results**: Option to download processed images as a ZIP file

## API Endpoints

- `POST /upload/`: Upload and process individual images
- `POST /upload-zip/`: Upload and process ZIP files
- `GET /health`: Health check endpoint


## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions, please open an issue on the GitHub repository. 
