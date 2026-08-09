// src/components/Navbar.tsx — Sticky glassmorphism navigation bar

import { NavLink } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  Home,
  LayoutDashboard,
  Mic,
  History,
  BarChart2,
  Info,
} from 'lucide-react';

const links = [
  { to: '/',          label: 'Home',      Icon: Home },
  { to: '/dashboard', label: 'Dashboard', Icon: LayoutDashboard },
  { to: '/predict',   label: 'Predict',   Icon: Mic },
  { to: '/history',   label: 'History',   Icon: History },
  { to: '/analytics', label: 'Analytics', Icon: BarChart2 },
  { to: '/about',     label: 'About',     Icon: Info },
];

export const Navbar = () => (
  <motion.nav
    initial={{ y: -60, opacity: 0 }}
    animate={{ y: 0, opacity: 1 }}
    transition={{ duration: 0.4, ease: 'easeOut' }}
    className="sticky top-0 z-50 w-full border-b border-white/5 bg-black/60 backdrop-blur-xl"
  >
    <div className="container mx-auto px-4 flex items-center justify-between h-16">
      {/* Logo */}
      <NavLink to="/" className="flex items-center space-x-2">
        <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-sm">
          🎤
        </div>
        <span className="font-outfit font-bold text-white text-lg hidden sm:block">
          SER<span className="text-blue-400">ai</span>
        </span>
      </NavLink>

      {/* Links */}
      <div className="flex items-center space-x-1">
        {links.map(({ to, label, Icon }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-all duration-200 ${
                isActive
                  ? 'bg-white/10 text-white'
                  : 'text-white/50 hover:text-white hover:bg-white/5'
              }`
            }
          >
            <Icon className="w-4 h-4" />
            <span className="hidden md:inline">{label}</span>
          </NavLink>
        ))}
      </div>
    </div>
  </motion.nav>
);
