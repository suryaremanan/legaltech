import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [files, setFiles] = useState([]);
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileContent, setFileContent] = useState(null);
  const [chatMessage, setChatMessage] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // Scan folders on component mount
    scanFolders();
  }, []);

  const scanFolders = async () => {
    try {
      const response = await axios.get('http://localhost:8000/scan-folders');
      setFiles(response.data.files);
    } catch (error) {
      console.error('Error scanning folders:', error);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    setSelectedFile(file);
    setFileContent(null);
    setLoading(true);

    try {
      const formData = new FormData();
      formData.append('file', file);
      
      const response = await axios.post('http://localhost:8000/process-pdf', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      if (response.data.status === 'error') {
        alert(response.data.message);
        return;
      }
      
      setFileContent(response.data);
    } catch (error) {
      console.error('Error uploading file:', error);
      alert(error.response?.data?.detail || error.response?.data?.message || 'Error uploading file');
    } finally {
      setLoading(false);
    }
  };

  const handleFileSelect = async (file) => {
    setSelectedFile(file);
    setFileContent(null);
    setLoading(true);

    try {
      const formData = new FormData();
      formData.append('file_path', file.path);
      
      const response = await axios.post('http://localhost:8000/process-pdf', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      if (response.data.status === 'error') {
        alert(response.data.message);
        return;
      }
      
      setFileContent(response.data);
    } catch (error) {
      console.error('Error processing file:', error);
      alert(error.response?.data?.detail || error.response?.data?.message || 'Error processing file');
    } finally {
      setLoading(false);
    }
  };

  const sendMessage = async () => {
    if (!chatMessage.trim()) return;

    const newMessage = { type: 'user', content: chatMessage };
    setChatHistory([...chatHistory, newMessage]);
    
    try {
      const formData = new FormData();
      formData.append('message', chatMessage);
      formData.append('context', fileContent?.text || '');
      
      const response = await axios.post('http://localhost:8000/chat', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      setChatHistory(prev => [...prev, {
        type: 'bot',
        content: response.data.response
      }]);
    } catch (error) {
      console.error('Error sending message:', error);
      alert(error.response?.data?.detail || 'Error sending message');
    }
    
    setChatMessage('');
  };

  return (
    <div className="App">
      <div className="container">
        <div className="file-explorer">
          <h2>Legal Documents</h2>
          <input
            type="file"
            accept=".pdf"
            onChange={handleFileUpload}
            style={{ marginBottom: '20px' }}
          />
          <div className="file-list">
            {files.map((file, index) => (
              <div 
                key={index} 
                className={`file-item ${selectedFile?.path === file.path ? 'selected' : ''}`}
                onClick={() => handleFileSelect(file)}
              >
                <span>{file.filename}</span>
                <small>{file.folder_path || 'root'}</small>
              </div>
            ))}
          </div>
        </div>

        <div className="content-viewer">
          {loading ? (
            <div className="loading">Processing...</div>
          ) : fileContent ? (
            <>
              <h3>Document Summary</h3>
              <div className="summary">{fileContent.summary}</div>
              <h3>Preview</h3>
              <div className="preview">{fileContent.text}</div>
            </>
          ) : (
            <div className="no-selection">Select a file to view</div>
          )}
        </div>

        <div className="chat-section">
          <div className="chat-history">
            {chatHistory.map((msg, index) => (
              <div key={index} className={`chat-message ${msg.type}`}>
                {msg.content}
              </div>
            ))}
          </div>
          <div className="chat-input">
            <input
              type="text"
              value={chatMessage}
              onChange={(e) => setChatMessage(e.target.value)}
              placeholder="Ask a question about the document..."
              onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
            />
            <button onClick={sendMessage}>Send</button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App; 