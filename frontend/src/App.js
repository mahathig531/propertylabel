import React, { useState } from 'react';
import './App.css';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [selectedZipFile, setSelectedZipFile] = useState(null);
  const [processedImages, setProcessedImages] = useState([]);
  const [objectDetectionResult, setObjectDetectionResult] = useState(null);
  const [highlightedClass, setHighlightedClass] = useState(null);
  const [highlightedImage, setHighlightedImage] = useState(null);
  const [originalImage, setOriginalImage] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [processingMode, setProcessingMode] = useState('room'); // 'room', 'objects', 'complete'

  const handleSingleFileChange = (e) => {
    setSelectedFile(e.target.files[0]);
    setError('');
    setHighlightedClass(null);
    setHighlightedImage(null);
    setOriginalImage(null);
  };

  const handleZipFileChange = (e) => {
    setSelectedZipFile(e.target.files[0]);
    setError('');
  };

  const processSingleImage = async () => {
    if (!selectedFile) {
      setError('Please select an image file.');
      return;
    }

    setLoading(true);
    setError('');
    setProcessedImages([]);
    setObjectDetectionResult(null);
    setHighlightedClass(null);
    setHighlightedImage(null);
    setOriginalImage(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      let endpoint = '/upload/';
      if (processingMode === 'objects') {
        endpoint = '/detect-objects/';
      } else if (processingMode === 'complete') {
        endpoint = '/process-complete/';
      }

      const response = await fetch(`http://localhost:8000${endpoint}`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to process image');
      }

      const data = await response.json();
      
      if (processingMode === 'objects') {
        setObjectDetectionResult(data);
        setOriginalImage(selectedFile ? URL.createObjectURL(selectedFile) : null);
      } else {
        // Add filename to the result for display
        const resultWithFilename = {
          ...data,
          filename: selectedFile.name
        };
        setProcessedImages([resultWithFilename]);
      }
    } catch (err) {
      setError(err.message || 'Failed to process image');
    }
    setLoading(false);
  };

  const processZipFile = async () => {
    if (!selectedZipFile) {
      setError('Please select a ZIP file.');
      return;
    }

    setLoading(true);
    setError('');
    setProcessedImages([]);
    setObjectDetectionResult(null);
    setHighlightedClass(null);
    setHighlightedImage(null);
    setOriginalImage(null);

    const formData = new FormData();
    formData.append('file', selectedZipFile);

    try {
      const response = await fetch('http://localhost:8000/upload-zip/', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to process ZIP file');
      }

      const data = await response.json();
      // Fix: Use data.results instead of data.images
      setProcessedImages(data.results || []);
    } catch (err) {
      setError(err.message || 'Failed to process ZIP file');
    }
    setLoading(false);
  };

  // Helper to convert base64 image to File for highlight endpoint
  const base64ToFile = (base64, filename = 'image.jpg') => {
    const byteString = atob(base64);
    const ab = new ArrayBuffer(byteString.length);
    const ia = new Uint8Array(ab);
    for (let i = 0; i < byteString.length; i++) {
      ia[i] = byteString.charCodeAt(i);
    }
    return new File([ab], filename, { type: 'image/jpeg' });
  };

  // Handle click on detected object
  const handleObjectClick = async (objClass) => {
    if (!objectDetectionResult) return;
    // Toggle highlight
    if (highlightedClass === objClass) {
      setHighlightedClass(null);
      setHighlightedImage(null);
      return;
    }
    setHighlightedClass(objClass);
    setHighlightedImage(null);
    // Send image and class to backend
    const file = base64ToFile(objectDetectionResult.image);
    const formData = new FormData();
    formData.append('file', file);
    formData.append('class_name', objClass);
    try {
      const highlightResponse = await fetch('http://localhost:8000/highlight-object/', {
        method: 'POST',
        body: formData,
      });
      if (!highlightResponse.ok) throw new Error('Failed to highlight object');
      const data = await highlightResponse.json();
      setHighlightedImage(`data:image/jpeg;base64,${data.image}`);
    } catch (err) {
      setError('Failed to highlight object');
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>🏠 Property Labeling Bot</h1>
      </header>

      <div className="main-container">
        <div className="upload-section">
          <div className="mode-selector simple-modes">
            <button 
              className={processingMode === 'room' ? 'active' : ''}
              onClick={() => setProcessingMode('room')}
            >
              🏷️ Label Room
            </button>
            <button 
              className={processingMode === 'objects' ? 'active' : ''}
              onClick={() => setProcessingMode('objects')}
            >
              🎯 Detect Objects
            </button>
            <button 
              className={processingMode === 'complete' ? 'active' : ''}
              onClick={() => setProcessingMode('complete')}
            >
              🔄 Complete Processing
            </button>
          </div>

          <div className="upload-options">
            <div className="single-upload">
              <h3>Single Image Upload</h3>
              <input 
                type="file" 
                accept=".jpg,.jpeg,.png" 
                onChange={handleSingleFileChange}
                className="file-input"
              />
              <button 
                onClick={processSingleImage} 
                disabled={loading || !selectedFile}
                className="process-btn"
              >
                {loading ? '🔄 Processing...' : '🚀 Process Image'}
              </button>
            </div>

            <div className="zip-upload">
              <h3>Batch Upload (ZIP)</h3>
              <input 
                type="file" 
                accept=".zip" 
                onChange={handleZipFileChange}
                className="file-input"
              />
              <button 
                onClick={processZipFile} 
                disabled={loading || !selectedZipFile}
                className="process-btn"
              >
                {loading ? '🔄 Processing...' : '📦 Process ZIP'}
              </button>
            </div>
          </div>
        </div>

        {error && (
          <div className="error-message">
            ❌ {error}
          </div>
        )}

        <div className="results-section">
          {objectDetectionResult && (
            <div className="object-detection-results">
              <h2>Object Detection Result</h2>
              <div className="detection-container">
                <div className="detected-image">
                  <img 
                    src={highlightedImage ? highlightedImage : originalImage} 
                    alt="Object detection result" 
                  />
                </div>
                <div className="detection-info">
                  <h3>Detected Objects ({objectDetectionResult.detections?.count || 0})</h3>
                  {objectDetectionResult.detections?.objects?.length > 0 ? (
                    <div className="objects-list">
                      {objectDetectionResult.detections.objects.map((obj, index) => (
                        <div 
                          key={index} 
                          className={`object-item${highlightedClass === obj.class ? ' highlighted' : ''}`}
                          onClick={() => handleObjectClick(obj.class)}
                          style={{ cursor: 'pointer' }}
                        >
                          <span className="object-name">🎯 {obj.class}</span>
                          <span className="object-confidence">{obj.confidence}%</span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p>No objects detected</p>
                  )}
                </div>
              </div>
            </div>
          )}

          {processedImages.length > 0 && (
            <div className="room-classification-results">
              <h2>Results</h2>
              <div className="image-grid">
                {processedImages.map((img, index) => (
                  <div 
                    key={img.filename || index} 
                    className="image-card"
                  >
                    <img 
                      src={`data:image/jpeg;base64,${img.image}`} 
                      alt={img.room_info?.room_type || 'Processed image'} 
                    />
                    <div className="image-info">
                      <div className="filename">{img.filename || `Image ${index + 1}`}</div>
                      {img.room_info && (
                        <div className="room-label">
                          🏠 {img.room_info.room_type || 'Unknown'}
                        </div>
                      )}
                      {img.room_info && (
                        <div className="confidence">
                          Confidence: {img.room_info.confidence?.toFixed(1)}%
                        </div>
                      )}
                      {img.detections && (
                        <div className="objects-list">
                          <strong>Objects Detected:</strong>
                          {img.detections.objects.map((obj, i) => (
                            <div key={i} className="object-item">
                              <span className="object-name">🎯 {obj.class}</span>
                              <span className="object-confidence">{obj.confidence}%</span>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
 