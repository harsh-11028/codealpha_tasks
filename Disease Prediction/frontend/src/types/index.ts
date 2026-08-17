export interface User {
  id: string;
  name: string;
  email: string;
  role: 'user' | 'admin';
  created_at?: string;
}

export interface AuthState {
  user: User | null;
  token: string | null;
  isLoading: boolean;
  isAuthenticated: boolean;
}

export interface PredictionResult {
  success: boolean;
  disease: string;
  prediction: number;
  label: string;
  probability: number;
  model: string;
  message: string;
  prediction_id: string;
}

export interface PredictionHistory {
  id: string;
  disease: string;
  prediction: number;
  probability: number;
  model_used: string;
  created_at: string;
  input_data: Record<string, number>;
}

export interface DashboardStats {
  total_predictions: number;
  heart_predictions: number;
  diabetes_predictions: number;
  breast_cancer_predictions: number;
  positive_predictions: number;
  negative_predictions: number;
  recent_predictions: PredictionHistory[];
  disease_distribution: { name: string; value: number }[];
  prediction_trend: { date: string; count: number }[];
}

export interface ModelMetrics {
  algorithm: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  roc_auc: number;
}

export interface HeartInput {
  age: number;
  sex: number;
  cp: number;
  trestbps: number;
  chol: number;
  fbs: number;
  restecg: number;
  thalach: number;
  exang: number;
  oldpeak: number;
  slope: number;
  ca: number;
  thal: number;
}

export interface DiabetesInput {
  Pregnancies: number;
  Glucose: number;
  BloodPressure: number;
  SkinThickness: number;
  Insulin: number;
  BMI: number;
  DiabetesPedigreeFunction: number;
  Age: number;
}
