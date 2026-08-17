import { Link } from 'react-router-dom';
import { Card, CardContent } from '../../components/ui/Card';
import { Heart, Droplets, Zap, ArrowRight } from 'lucide-react';
import { Button } from '../../components/ui/Button';

export default function PredictSelect() {
  const models = [
    {
      id: 'heart',
      name: 'Heart Disease',
      description: 'Predict the presence of heart disease based on 13 clinical features including blood pressure, cholesterol, and ECG results.',
      icon: Heart,
      color: 'text-red-500',
      bgColor: 'bg-red-50',
      hoverBorder: 'hover:border-red-300',
      features: 13,
      path: '/predict/heart'
    },
    {
      id: 'diabetes',
      name: 'Diabetes',
      description: 'Assess the risk of diabetes using 8 diagnostic measurements such as glucose levels, BMI, and insulin.',
      icon: Droplets,
      color: 'text-blue-500',
      bgColor: 'bg-blue-50',
      hoverBorder: 'hover:border-blue-300',
      features: 8,
      path: '/predict/diabetes'
    },
    {
      id: 'breast-cancer',
      name: 'Breast Cancer',
      description: 'Classify breast mass as malignant or benign using 30 features computed from a digitized image of a fine needle aspirate (FNA).',
      icon: Zap,
      color: 'text-purple-600',
      bgColor: 'bg-purple-50',
      hoverBorder: 'hover:border-purple-300',
      features: 30,
      path: '/predict/breast-cancer'
    }
  ];

  return (
    <div className="max-w-5xl mx-auto space-y-8">
      <div className="text-center space-y-4">
        <h2 className="text-3xl font-bold text-slate-900">Select a Prediction Model</h2>
        <p className="text-lg text-slate-600 max-w-2xl mx-auto">Choose a disease category below to input patient data and generate an AI-powered prediction.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-8">
        {models.map((model) => {
          const Icon = model.icon;
          return (
            <Card key={model.id} className={`flex flex-col h-full transition-all duration-200 hover:shadow-lg ${model.hoverBorder}`}>
              <CardContent className="p-8 flex flex-col flex-1 text-center items-center">
                <div className={`w-20 h-20 rounded-full flex items-center justify-center mb-6 ${model.bgColor} ${model.color}`}>
                  <Icon className="w-10 h-10" />
                </div>
                <h3 className="text-2xl font-semibold text-slate-900 mb-3">{model.name}</h3>
                <p className="text-slate-600 flex-1 mb-6 text-sm">{model.description}</p>
                
                <div className="w-full mt-auto">
                  <div className="text-xs text-slate-500 mb-4 font-medium px-3 py-1 bg-slate-100 rounded-full inline-block">
                    {model.features} input features
                  </div>
                  <Link to={model.path} className="block w-full">
                    <Button className="w-full group" variant="default">
                      Start Prediction
                      <ArrowRight className="w-4 h-4 ml-2 group-hover:translate-x-1 transition-transform" />
                    </Button>
                  </Link>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>
    </div>
  );
}
