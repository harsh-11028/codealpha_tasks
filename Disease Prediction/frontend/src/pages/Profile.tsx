import { useAuth } from '../context/AuthContext';
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card';
import { User as UserIcon, Mail, Shield, Calendar } from 'lucide-react';

export default function Profile() {
  const { user } = useAuth();

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <h2 className="text-2xl font-bold text-slate-900">User Profile</h2>

      <Card>
        <CardHeader>
          <CardTitle>Account Information</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          <div className="flex items-center space-x-4 pb-6 border-b border-slate-100">
            <div className="h-20 w-20 rounded-full bg-blue-100 flex items-center justify-center text-blue-600">
              <UserIcon className="h-10 w-10" />
            </div>
            <div>
              <h3 className="text-xl font-semibold text-slate-900">{user?.name}</h3>
              <p className="text-slate-500 capitalize">{user?.role} Account</p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="flex items-start space-x-3">
              <Mail className="h-5 w-5 text-slate-400 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-slate-900">Email Address</p>
                <p className="text-sm text-slate-500">{user?.email}</p>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <Shield className="h-5 w-5 text-slate-400 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-slate-900">Access Role</p>
                <p className="text-sm text-slate-500 capitalize">{user?.role}</p>
              </div>
            </div>
            <div className="flex items-start space-x-3">
              <Calendar className="h-5 w-5 text-slate-400 mt-0.5" />
              <div>
                <p className="text-sm font-medium text-slate-900">Member Since</p>
                <p className="text-sm text-slate-500">{user?.created_at ? new Date(user.created_at).toLocaleDateString() : 'N/A'}</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
