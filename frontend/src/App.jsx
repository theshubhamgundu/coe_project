import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Plus, 
  Search, 
  Asterisk, 
  Zap, 
  MoreHorizontal, 
  Layout, 
  Cpu, 
  Activity, 
  ShieldCheck, 
  Clock, 
  Droplets, 
  Lightbulb, 
  Volume2, 
  ParkingCircle, 
  Construction, 
  Trash2, 
  Hammer, 
  TrafficCone,
  AlertCircle,
  Paperclip,
  Wand2,
  TrendingDown,
  TrendingUp,
  Fingerprint,
  FileSearch,
  CheckCircle2,
  XCircle,
  Microscope,
  Box,
  Layers,
  ArrowRight,
  MoveUpRight,
  Target,
  BarChart3,
  Dna,
  Binary
} from 'lucide-react';
import './index.css';

function App() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [text]);

  const handlePredict = async (e) => {
    if (e) e.preventDefault();
    if (!text.trim() || loading) return;

    setLoading(true);
    setError(null);
    try {
      let api_url = import.meta.env.VITE_API_URL || "http://localhost:8000";
      // Ensure no double slashes cause a 404 error
      api_url = api_url.replace(/\/$/, "");
      const response = await axios.post(`${api_url}/predict`, {
        text: text,
      });
      setPrediction(response.data);
    } catch (err) {
      console.error(err);
      if (err.message === "Network Error") {
        setError("Inference Engine is waking up (Render Free Tier). Please wait 30 seconds and try again.");
      } else {
        setError(err.response?.data?.detail || "System Link Failure. Internal neural core unreachable.");
      }
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handlePredict();
    }
  };

  return (
    <div className="container">
      <div className="bg-overlay"></div>

      <header style={{ textAlign: 'center' }}>
        <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 1 }}>
          <div style={{ display: 'inline-flex', alignItems: 'center', gap: '8px', border: '1px solid rgba(255, 69, 58, 0.4)', padding: '5px 12px', borderRadius: '100px', fontSize: '0.6rem', letterSpacing: '2px', fontWeight: 800, color: '#ff453a', marginBottom: '2rem' }}>
             <Binary size={12} /> SYSTEM ACTIVE
          </div>
          <h1>Grievance Lab</h1>
          <p className="subtitle">Public Grievance Neural Classification & Priority Logic</p>
        </motion.div>
      </header>

      <motion.div 
         initial={{ opacity: 0 }}
         animate={{ opacity: 1 }}
         transition={{ duration: 1, delay: 0.3 }}
         className="prompt-area"
      >
        <textarea
          ref={textareaRef}
          rows={1}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Input grievance data verbatim..."
        />
        <button 
           className="btn-icon" 
           onClick={handlePredict} 
           disabled={loading || !text.trim()}
        >
          {loading ? <div className="loader" /> : <MoveUpRight size={22} />}
        </button>
      </motion.div>

      <AnimatePresence>
        {error && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} style={{ color: '#ff3b30', fontSize: '0.8rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '1px' }}>
            <AlertCircle size={14} style={{ verticalAlign: 'middle', marginRight: '8px' }} />
            {error}
          </motion.div>
        )}

        {prediction && (
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="response-view"
          >
            <div className="result-row">
              <span className="label">Category Prediction</span>
              <span className="value" style={{ letterSpacing: '0px' }}>{prediction.predicted_category}</span>
            </div>

            <div className="result-row">
              <span className="label">Urgency Score</span>
              <span className={`value priority-${prediction.priority_level}`}>
                {prediction.priority_level}
              </span>
            </div>

            <div className="result-row">
              <span className="label">Confidence Vector</span>
              <span className="value">{(prediction.category_confidence * 100).toFixed(2)}%</span>
            </div>

            <div className="result-row" style={{ marginTop: '3rem', border: 'none', background: 'rgba(255,255,255,0.02)', padding: '2rem', borderRadius: '12px' }}>
              <p style={{ color: 'rgba(255,255,255,0.3)', fontSize: '0.75rem', lineHeight: 1.8, fontWeight: 300 }}>
                 Neural audit for payload [G-9283] complete. Prediction verified against text statistics (Pol: {prediction.sentiment_score.toFixed(3)}). Priority reliability factor: {(prediction.priority_confidence * 100).toFixed(1)}%.
              </p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default App;
