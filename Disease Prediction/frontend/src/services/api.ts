import axios from 'axios';
import type { HeartInput, DiabetesInput } from '../types';

const api = axios.create({
  baseURL: '/api'
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('dp_token');
  if (token && config.headers) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('dp_token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export const authApi = {
  login: (data: any) => api.post('/auth/login', data).then(res => res.data),
  register: (data: any) => api.post('/auth/register', data).then(res => res.data),
  me: () => api.get('/auth/me').then(res => res.data)
};

export const predictApi = {
  predictHeart: (data: HeartInput) => api.post('/predict/heart', data).then(res => res.data),
  predictDiabetes: (data: DiabetesInput) => api.post('/predict/diabetes', data).then(res => res.data),
  predictBreastCancer: (data: Record<string, number>) => api.post('/predict/breast-cancer', data).then(res => res.data)
};

export const historyApi = {
  getHistory: (params?: any) => api.get('/predictions', { params }).then(res => res.data),
  delete: (id: string) => api.delete(`/predictions/${id}`).then(res => res.data)
};

export const dashboardApi = {
  getStats: () => api.get('/dashboard/stats').then(res => res.data)
};

export const modelsApi = {
  getPerformance: (disease: string) => api.get(`/models/${disease}/performance`).then(res => res.data)
};

export const diseasesApi = {
  getAll: () => api.get('/diseases').then(res => res.data)
};
