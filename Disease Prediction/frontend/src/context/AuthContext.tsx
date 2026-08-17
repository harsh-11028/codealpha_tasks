import { createContext, useContext, useEffect, useState } from 'react';
import type { ReactNode } from 'react';
import type { AuthState } from '../types';
import { authApi } from '../services/api';

interface AuthContextType extends AuthState {
  login: (data: any) => Promise<void>;
  register: (data: any) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [state, setState] = useState<AuthState>({
    user: null,
    token: localStorage.getItem('dp_token'),
    isLoading: true,
    isAuthenticated: false,
  });

  useEffect(() => {
    const initAuth = async () => {
      const token = localStorage.getItem('dp_token');
      if (token) {
        try {
          const res = await authApi.me();
          setState(prev => ({ ...prev, user: res.data || res.user, isAuthenticated: true, isLoading: false }));
        } catch (error) {
          localStorage.removeItem('dp_token');
          setState({ user: null, token: null, isLoading: false, isAuthenticated: false });
        }
      } else {
        setState(prev => ({ ...prev, isLoading: false }));
      }
    };
    initAuth();
  }, []);

  const login = async (data: any) => {
    const res = await authApi.login(data);
    const payload = res.data || res;
    localStorage.setItem('dp_token', payload.access_token);
    setState({ user: payload.user, token: payload.access_token, isLoading: false, isAuthenticated: true });
  };

  const register = async (data: any) => {
    const res = await authApi.register(data);
    const payload = res.data || res;
    localStorage.setItem('dp_token', payload.access_token);
    setState({ user: payload.user, token: payload.access_token, isLoading: false, isAuthenticated: true });
  };

  const logout = () => {
    localStorage.removeItem('dp_token');
    setState({ user: null, token: null, isLoading: false, isAuthenticated: false });
  };

  return (
    <AuthContext.Provider value={{ ...state, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) throw new Error('useAuth must be used within an AuthProvider');
  return context;
};
