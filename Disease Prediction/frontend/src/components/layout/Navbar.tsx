import { Menu, Bell, User as UserIcon } from 'lucide-react';
import { useLocation, Link } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import { useState, useRef, useEffect } from 'react';

export function Navbar({ onMenuClick }: { onMenuClick: () => void }) {
  const location = useLocation();
  const { user, logout } = useAuth();
  const [showDropdown, setShowDropdown] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const getPageTitle = () => {
    const path = location.pathname;
    if (path === '/dashboard') return 'Dashboard';
    if (path.startsWith('/predict/heart')) return 'Heart Disease Prediction';
    if (path.startsWith('/predict/diabetes')) return 'Diabetes Prediction';
    if (path.startsWith('/predict/breast-cancer')) return 'Breast Cancer Prediction';
    if (path.startsWith('/predict')) return 'Predict';
    if (path === '/history') return 'Prediction History';
    if (path === '/models') return 'Model Performance';
    if (path === '/profile') return 'Profile';
    if (path === '/about') return 'About';
    if (path === '/admin') return 'Admin Panel';
    return 'MedPredict AI';
  };

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setShowDropdown(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <header className="h-16 bg-white border-b border-slate-200 flex items-center justify-between px-4 sm:px-6 z-10">
      <div className="flex items-center flex-1">
        <button onClick={onMenuClick} className="lg:hidden p-2 -ml-2 mr-2 text-slate-500 hover:text-slate-700 rounded-md hover:bg-slate-100">
          <Menu className="h-6 w-6" />
        </button>
        <h1 className="text-lg font-semibold text-slate-900 truncate">{getPageTitle()}</h1>
      </div>

      <div className="flex items-center space-x-4">
        <button className="p-2 text-slate-400 hover:text-slate-500 relative">
          <Bell className="h-5 w-5" />
          <span className="absolute top-1.5 right-1.5 h-2 w-2 rounded-full bg-red-500"></span>
        </button>

        <div className="relative" ref={dropdownRef}>
          <button 
            onClick={() => setShowDropdown(!showDropdown)}
            className="flex items-center space-x-2 p-1.5 rounded-full hover:bg-slate-100"
          >
            <div className="h-8 w-8 rounded-full bg-blue-100 flex items-center justify-center text-blue-600">
              <UserIcon className="h-5 w-5" />
            </div>
          </button>

          {showDropdown && (
            <div className="absolute right-0 mt-2 w-48 bg-white rounded-md shadow-lg py-1 border border-slate-200">
              <div className="px-4 py-2 border-b border-slate-100">
                <p className="text-sm font-medium text-slate-900 truncate">{user?.name}</p>
                <p className="text-xs text-slate-500 truncate">{user?.email}</p>
              </div>
              <Link to="/profile" className="block px-4 py-2 text-sm text-slate-700 hover:bg-slate-100" onClick={() => setShowDropdown(false)}>
                Profile
              </Link>
              <button 
                onClick={() => { setShowDropdown(false); logout(); }} 
                className="block w-full text-left px-4 py-2 text-sm text-red-600 hover:bg-slate-100"
              >
                Logout
              </button>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
