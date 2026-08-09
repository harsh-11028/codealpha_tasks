// src/App.tsx — Root application with BrowserRouter, route protection, and global ErrorBoundary.
//
// ROUTING NOTES:
//   • BrowserRouter uses the History API — Vite is set to `appType: 'spa'` which
//     serves index.html for every URL, so direct URL access and refresh work.
//   • Every route is wrapped in its own ErrorBoundary so one crashing page never
//     takes down the whole application.
//   • A wildcard `path="*"` route shows the 404 page for unknown URLs instead
//     of silently redirecting to home.

import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { Navbar }        from './components/Navbar';
import { ErrorBoundary } from './components/ErrorBoundary';
import { Home }          from './pages/Home';
import { Dashboard }     from './pages/Dashboard';
import { Predict }       from './pages/Predict';
import { History }       from './pages/History';
import { Analytics }     from './pages/Analytics';
import { About }         from './pages/About';
import { NotFound }      from './pages/NotFound';
import './index.css';

function App() {
  return (
    <Router>
      {/* Global boundary — catches errors outside any page boundary */}
      <ErrorBoundary>
        <div className="dark min-h-screen bg-background text-foreground">
          <Navbar />
          {/* Per-route boundaries — a crashed page shows only that section's error */}
          <Routes>
            <Route path="/"          element={<ErrorBoundary><Home      /></ErrorBoundary>} />
            <Route path="/dashboard" element={<ErrorBoundary><Dashboard /></ErrorBoundary>} />
            <Route path="/predict"   element={<ErrorBoundary><Predict   /></ErrorBoundary>} />
            <Route path="/history"   element={<ErrorBoundary><History   /></ErrorBoundary>} />
            <Route path="/analytics" element={<ErrorBoundary><Analytics /></ErrorBoundary>} />
            <Route path="/about"     element={<ErrorBoundary><About     /></ErrorBoundary>} />
            {/* Catch-all: shows 404 instead of silently going to home */}
            <Route path="*"          element={<ErrorBoundary><NotFound  /></ErrorBoundary>} />
          </Routes>
        </div>
      </ErrorBoundary>
    </Router>
  );
}

export default App;
