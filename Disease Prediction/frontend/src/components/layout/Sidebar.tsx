import { useLocation, Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import { BarChart2, Activity, Heart, Droplets, Zap, Clock, Cpu, Info, LogOut, Settings } from 'lucide-react';
import { cn } from '../../utils/cn';

export function Sidebar({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  const { user, logout } = useAuth();
  const location = useLocation();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const navItems = [
    { name: 'Dashboard', path: '/dashboard', icon: BarChart2 },
    { name: 'Predict', path: '/predict', icon: Activity },
    { name: 'Heart Disease', path: '/predict/heart', icon: Heart, indent: true },
    { name: 'Diabetes', path: '/predict/diabetes', icon: Droplets, indent: true },
    { name: 'Breast Cancer', path: '/predict/breast-cancer', icon: Zap, indent: true },
    { name: 'History', path: '/history', icon: Clock },
    { name: 'Models', path: '/models', icon: Cpu },
    { name: 'About', path: '/about', icon: Info },
  ];

  return (
    <>
      {isOpen && <div className="fixed inset-0 z-40 bg-slate-900/50 lg:hidden" onClick={onClose} />}
      <div className={cn(
        "fixed inset-y-0 left-0 z-50 w-64 bg-white border-r border-slate-200 transform transition-transform duration-200 ease-in-out flex flex-col lg:translate-x-0 lg:static lg:h-screen",
        isOpen ? "translate-x-0" : "-translate-x-full"
      )}>
        <div className="h-16 flex items-center px-6 border-b border-slate-200">
          <Heart className="h-6 w-6 text-blue-600 mr-2" />
          <span className="text-xl font-bold text-slate-900">MedPredict AI</span>
        </div>

        <nav className="flex-1 overflow-y-auto py-4 px-3 space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path || (item.path !== '/predict' && location.pathname.startsWith(item.path + '/'));
            return (
              <Link
                key={item.path}
                to={item.path}
                onClick={() => onClose()}
                className={cn(
                  "flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors",
                  isActive ? "bg-blue-50 text-blue-700" : "text-slate-700 hover:bg-slate-100",
                  item.indent ? "ml-4" : ""
                )}
              >
                <Icon className={cn("flex-shrink-0 mr-3 h-5 w-5", isActive ? "text-blue-700" : "text-slate-400")} />
                {item.name}
              </Link>
            );
          })}

          {user?.role === 'admin' && (
            <div className="pt-4 mt-4 border-t border-slate-200">
              <p className="px-3 text-xs font-semibold text-slate-500 uppercase tracking-wider mb-2">Admin</p>
              <Link to="/admin" className="flex items-center px-3 py-2 text-sm font-medium text-slate-700 rounded-md hover:bg-slate-100" onClick={() => onClose()}>
                <Settings className="flex-shrink-0 mr-3 h-5 w-5 text-slate-400" />
                Admin Panel
              </Link>
            </div>
          )}
        </nav>

        <div className="border-t border-slate-200 p-4">
          <div className="flex items-center w-full">
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-slate-900 truncate">{user?.name}</p>
              <p className="text-xs text-slate-500 truncate">{user?.email}</p>
            </div>
            <button onClick={handleLogout} className="ml-2 p-2 text-slate-400 hover:text-slate-600 rounded-md hover:bg-slate-100">
              <LogOut className="h-5 w-5" />
            </button>
          </div>
        </div>
      </div>
    </>
  );
}
