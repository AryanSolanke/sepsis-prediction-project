import React, { useState } from 'react';
import { rpcCall } from '../api';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '../components/ui/card';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { Button } from '../components/ui/button';
import { Badge } from '../components/ui/badge';
import { Progress } from '../components/ui/progress';
import { Spinner } from '../components/ui/spinner';
import { Activity, Thermometer, Droplet, User, Heart, Wind, AlertTriangle, CheckCircle2 } from 'lucide-react';
import { cn } from '../lib/utils';

export function RiskPredictor() {
  const [formData, setFormData] = useState({
    HR: 80,
    O2Sat: 98,
    Temp: 37,
    SBP: 120,
    Resp: 16,
    WBC: 8,
    Lactate: 1.2,
    Age: 65
  });
  
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: parseFloat(value) }));
  };

  const handlePredict = async () => {
    setLoading(true);
    console.log("[ACTION_START] Risk prediction requested");
    try {
      const response = await rpcCall({ func: 'predict_sepsis', args: { patient_data: formData } });
      setResult(response);
      console.log("[FETCH_RESPONSE] Prediction successful");
    } catch (error) {
      console.error("[PARSE_ERROR] Prediction failed", error);
    } finally {
      setLoading(false);
    }
  };

  const formFields = [
    { name: 'Age', label: 'Age (Years)', icon: User, min: 18, max: 120 },
    { name: 'HR', label: 'Heart Rate (BPM)', icon: Heart, min: 40, max: 200 },
    { name: 'O2Sat', label: 'O2 Saturation (%)', icon: Wind, min: 70, max: 100 },
    { name: 'Temp', label: 'Temperature (°C)', icon: Thermometer, min: 34, max: 42 },
    { name: 'SBP', label: 'Systolic Blood Pressure', icon: Droplet, min: 60, max: 220 },
    { name: 'Resp', label: 'Resp. Rate (Breaths/min)', icon: Wind, min: 8, max: 50 },
    { name: 'WBC', label: 'WBC Count (10^9/L)', icon: Activity, min: 1, max: 50 },
    { name: 'Lactate', label: 'Lactate (mmol/L)', icon: Activity, min: 0.1, max: 20 },
  ];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 animate-in fade-in slide-in-from-right-4 duration-500">
      <Card className="border-white/10 bg-card/60 backdrop-blur-sm shadow-xl">
        <CardHeader>
          <CardTitle className="font-heading text-xl">Patient Profile Input</CardTitle>
          <CardDescription>Enter clinical vitals for real-time risk assessment.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {formFields.map((field) => (
              <div key={field.name} className="space-y-2">
                <Label htmlFor={field.name} className="flex items-center gap-2 text-xs font-medium">
                  <field.icon className="h-3 w-3 text-primary" />
                  {field.label}
                </Label>
                <Input
                  id={field.name}
                  name={field.name}
                  type="number"
                  step="0.1"
                  value={formData[field.name as keyof typeof formData]}
                  onChange={handleInputChange}
                  className="bg-white/5 border-white/10 focus:ring-primary/50"
                />
              </div>
            ))}
          </div>
        </CardContent>
        <CardFooter>
          <Button 
            onClick={handlePredict} 
            disabled={loading}
            className="w-full h-11 text-base font-semibold transition-all hover:scale-[1.01] active:scale-[0.99]"
          >
            {loading ? <Spinner className="mr-2 h-4 w-4" /> : <Activity className="mr-2 h-4 w-4" />}
            Generate Prediction
          </Button>
        </CardFooter>
      </Card>

      <div className="space-y-6">
        {!result && !loading && (
          <Card className="h-full border-dashed border-white/10 bg-white/5 flex flex-col items-center justify-center p-8 text-center space-y-4">
            <div className="rounded-full bg-primary/10 p-4">
              <Activity className="h-8 w-8 text-primary/40" />
            </div>
            <div className="space-y-2">
              <h3 className="text-lg font-medium">Awaiting Input</h3>
              <p className="text-sm text-muted-foreground max-w-[280px]">Fill in the patient vitals to the left and click 'Generate Prediction' to see the AI analysis.</p>
            </div>
          </Card>
        )}

        {loading && (
          <Card className="h-full border-white/10 bg-card/60 backdrop-blur-sm p-8 flex flex-col items-center justify-center space-y-4">
            <Spinner className="h-8 w-8 text-primary" />
            <p className="text-sm text-muted-foreground animate-pulse">Analyzing clinical markers...</p>
          </Card>
        )}

        {result && !loading && (
          <div className="space-y-6">
            <Card className={cn(
              "border-white/10 bg-card/60 backdrop-blur-sm shadow-xl overflow-hidden relative",
              result.risk_level === 'High' ? "ring-1 ring-rose-500/50" : 
              result.risk_level === 'Medium' ? "ring-1 ring-amber-500/50" : "ring-1 ring-emerald-500/50"
            )}>
              <div className={cn(
                "absolute top-0 left-0 w-full h-1",
                result.risk_level === 'High' ? "bg-rose-500" : 
                result.risk_level === 'Medium' ? "bg-amber-500" : "bg-emerald-500"
              )} />
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-lg font-heading">AI Sepsis Risk Assessment</CardTitle>
                  <Badge className={cn(
                    "px-3 py-1",
                    result.risk_level === 'High' ? "bg-rose-500/20 text-rose-500 border-rose-500/20" : 
                    result.risk_level === 'Medium' ? "bg-amber-500/20 text-amber-500 border-amber-500/20" : 
                    "bg-emerald-500/20 text-emerald-500 border-emerald-500/20"
                  )}>
                    {result.risk_level} Risk
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="space-y-8">
                <div className="text-center space-y-2">
                  <div className="text-5xl font-bold font-heading">
                    {(result.sepsis_probability * 100).toFixed(1)}%
                  </div>
                  <p className="text-sm text-muted-foreground">Sepsis Probability</p>
                  <Progress 
                    value={result.sepsis_probability * 100} 
                    className="h-2 w-full mt-4"
                  />
                </div>

                <div className="space-y-4">
                  <h4 className="text-sm font-semibold flex items-center gap-2">
                    <AlertTriangle className="h-4 w-4 text-amber-500" />
                    Top Risk Contributors
                  </h4>
                  <div className="space-y-3">
                    {result.top_contributors.map((contributor: any, i: number) => (
                      <div key={i} className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className="font-medium">{contributor.feature}</span>
                          <span className="text-muted-foreground">Value: {contributor.value}</span>
                        </div>
                        <Progress 
                          value={Math.min(contributor.contribution_score * 10, 100)} 
                          className="h-1 bg-white/5"
                        />
                      </div>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>

            <div className="flex items-start gap-4 p-4 rounded-xl bg-white/5 border border-white/10">
              <div className="p-2 rounded-lg bg-primary/10">
                <CheckCircle2 className="h-5 w-5 text-primary" />
              </div>
              <div className="space-y-1">
                <h4 className="text-sm font-semibold">Clinical Recommendation</h4>
                <p className="text-xs text-muted-foreground leading-relaxed">
                  {result.risk_level === 'High' 
                    ? "Immediate clinical evaluation and initiation of sepsis protocols (fluids, antibiotics) is strongly recommended." 
                    : result.risk_level === 'Medium'
                    ? "Increased monitoring frequency recommended. Review source of potential infection."
                    : "Patient shows stable vitals. Continue standard ICU monitoring protocols."}
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
