// src/pages/NotFound.tsx — 404 page for unknown routes
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';

export const NotFound = () => (
  <div className="min-h-[80vh] flex items-center justify-center px-4">
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-panel rounded-2xl p-10 text-center max-w-md"
    >
      <p className="text-7xl font-outfit font-bold text-white/10 mb-4">404</p>
      <h1 className="text-2xl font-outfit font-bold mb-2">Page Not Found</h1>
      <p className="text-muted-foreground text-sm mb-6">
        The page you're looking for doesn't exist.
      </p>
      <Link
        to="/"
        className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 transition-colors"
      >
        ← Back to Home
      </Link>
    </motion.div>
  </div>
);
