import React, { useEffect, useState } from 'react';
import { Cpu, Camera, History, FileText, Activity } from 'lucide-react';
import ocrApi from '../services/api';

interface HeaderProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
}

export const Header: React.FC<HeaderProps> = ({ activeTab, setActiveTab }) => {
  const [apiOnline, setApiOnline] = useState<boolean>(true);
  const [device, setDevice] = useState<string>('auto');

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res = await ocrApi.getHealth();
        setApiOnline(res.status === 'ok');
        if (res.device) setDevice(res.device.toUpperCase());
      } catch {
        setApiOnline(false);
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="header">
      <div className="logo-group">
        <div className="logo-icon">✨</div>
        <div>
          <h1 className="logo-title">Maharaja OCR Studio</h1>
          <p className="logo-subtitle">AI Handwritten Character Recognition</p>
        </div>
      </div>

      <nav className="nav-tabs">
        <button
          className={`nav-tab ${activeTab === 'workspace' ? 'active' : ''}`}
          onClick={() => setActiveTab('workspace')}
        >
          <FileText size={16} /> Workspace
        </button>
        <button
          className={`nav-tab ${activeTab === 'webcam' ? 'active' : ''}`}
          onClick={() => setActiveTab('webcam')}
        >
          <Camera size={16} /> Webcam Live
        </button>
        <button
          className={`nav-tab ${activeTab === 'history' ? 'active' : ''}`}
          onClick={() => setActiveTab('history')}
        >
          <History size={16} /> Analytics
        </button>
        <button
          className={`nav-tab ${activeTab === 'models' ? 'active' : ''}`}
          onClick={() => setActiveTab('models')}
        >
          <Cpu size={16} /> Neural Architectures
        </button>
      </nav>

      <div className="status-badge" style={{ borderColor: apiOnline ? 'rgba(16, 185, 129, 0.3)' : 'rgba(244, 63, 94, 0.3)', color: apiOnline ? '#10b981' : '#f43f5e', background: apiOnline ? 'rgba(16, 185, 129, 0.1)' : 'rgba(244, 63, 94, 0.1)' }}>
        <div className="status-dot" style={{ backgroundColor: apiOnline ? '#10b981' : '#f43f5e', boxShadow: `0 0 10px ${apiOnline ? '#10b981' : '#f43f5e'}` }} />
        <Activity size={15} />
        <span>{apiOnline ? `Backend Live (${device})` : 'Offline / Reconnecting'}</span>
      </div>
    </header>
  );
};
