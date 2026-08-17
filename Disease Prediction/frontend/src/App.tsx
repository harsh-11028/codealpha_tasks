import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import { Layout } from './components/layout/Layout';
import { ProtectedRoute } from './components/ProtectedRoute';

import Login from './pages/Login';
import Register from './pages/Register';
import Dashboard from './pages/Dashboard';
import PredictSelect from './pages/predict/PredictSelect';
import HeartPrediction from './pages/predict/HeartPrediction';
import DiabetesPrediction from './pages/predict/DiabetesPrediction';
import BreastCancerPrediction from './pages/predict/BreastCancerPrediction';
import History from './pages/History';
import Models from './pages/Models';
import Profile from './pages/Profile';
import About from './pages/About';
import Admin from './pages/Admin';

export default function App() {
  return (
    <AuthProvider>
      <Router>
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />

          <Route element={<ProtectedRoute />}>
            <Route element={<Layout />}>
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/predict" element={<PredictSelect />} />
              <Route path="/predict/heart" element={<HeartPrediction />} />
              <Route path="/predict/diabetes" element={<DiabetesPrediction />} />
              <Route path="/predict/breast-cancer" element={<BreastCancerPrediction />} />
              <Route path="/history" element={<History />} />
              <Route path="/models" element={<Models />} />
              <Route path="/profile" element={<Profile />} />
              <Route path="/about" element={<About />} />
              
              {/* Admin Routes */}
              <Route element={<ProtectedRoute requireAdmin />}>
                <Route path="/admin" element={<Admin />} />
              </Route>
            </Route>
          </Route>
        </Routes>
      </Router>
    </AuthProvider>
  );
}
