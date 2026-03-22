import React, { useEffect, useState } from 'react';
import { rpcCall } from '../api';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../components/ui/select';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '../components/ui/table';
import { Badge } from '../components/ui/badge';
import { Skeleton } from '../components/ui/skeleton';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { Users, LayoutGrid, AlertCircle } from 'lucide-react';

export function CohortAnalysis() {
  const [cohortFeature, setCohortFeature] = useState('AgeGroup');
  const [data, setData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchCohorts = async (feature: string) => {
    setLoading(true);
    try {
      console.log(`[FETCH_START] Cohort analysis for ${feature}`);
      const result = await rpcCall({ func: 'get_cohort_analysis', args: { cohort_feature: feature } });
      setData(result);
      console.log("[FETCH_RESPONSE] Cohort data received");
    } catch (error) {
      console.error("[PARSE_ERROR] Failed to fetch cohort data", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchCohorts(cohortFeature);
  }, [cohortFeature]);

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div className="space-y-1">
          <h2 className="text-2xl font-bold tracking-tight font-heading">Cohort Performance</h2>
          <p className="text-muted-foreground">Deep dive into model accuracy across demographic and clinical segments.</p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-muted-foreground">Segment by:</span>
          <Select value={cohortFeature} onValueChange={setCohortFeature}>
            <SelectTrigger className="w-[180px] border-white/10 bg-card/60">
              <SelectValue placeholder="Select feature" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="AgeGroup">Age Group</SelectItem>
              <SelectItem value="Gender">Gender</SelectItem>
              <SelectItem value="Unit1">Medical Unit</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-2 border-white/10 bg-card/60 backdrop-blur-sm shadow-xl">
          <CardHeader>
            <CardTitle className="text-lg font-heading flex items-center gap-2">
              <Users className="h-4 w-4 text-primary" />
              Volume & Sepsis Distribution
            </CardTitle>
          </CardHeader>
          <CardContent>
            {loading ? (
              <Skeleton className="h-[350px] w-full rounded-lg" />
            ) : (
              <div className="h-[350px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="rgba(255,255,255,0.05)" />
                    <XAxis 
                      dataKey="cohort" 
                      axisLine={false} 
                      tickLine={false} 
                      tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 12 }}
                    />
                    <YAxis 
                      axisLine={false} 
                      tickLine={false} 
                      tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 12 }}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: 'rgba(9, 9, 11, 0.95)', 
                        borderColor: 'rgba(255,255,255,0.1)',
                        borderRadius: '8px'
                      }}
                    />
                    <Legend iconType="circle" />
                    <Bar dataKey="count" name="Total Patients" fill="hsl(var(--primary))" fillOpacity={0.4} radius={[4, 4, 0, 0]} />
                    <Bar dataKey="sepsis_cases" name="Sepsis Cases" fill="hsl(var(--rose-500))" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}
          </CardContent>
        </Card>

        <Card className="border-white/10 bg-card/60 backdrop-blur-sm shadow-xl">
          <CardHeader>
            <CardTitle className="text-lg font-heading flex items-center gap-2">
              <LayoutGrid className="h-4 w-4 text-primary" />
              Segment Summary
            </CardTitle>
            <CardDescription>
              Performance breakdown for selected feature.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {loading ? (
              [1, 2, 3].map(i => <Skeleton key={i} className="h-16 w-full" />)
            ) : (
              data.map((item, i) => (
                <div key={i} className="flex items-center justify-between p-3 rounded-lg bg-white/5 border border-white/5">
                  <div className="space-y-1">
                    <p className="text-sm font-semibold">{item.cohort}</p>
                    <p className="text-xs text-muted-foreground">{item.count} Patients</p>
                  </div>
                  <div className="text-right space-y-1">
                    <p className="text-xs font-medium text-emerald-500">{(item.recall * 100).toFixed(1)}% Recall</p>
                    <p className="text-xs font-medium text-blue-500">{(item.precision * 100).toFixed(1)}% Precision</p>
                  </div>
                </div>
              ))
            )}
          </CardContent>
        </Card>
      </div>

      <Card className="border-white/10 bg-card/60 backdrop-blur-sm shadow-xl">
        <CardHeader>
          <CardTitle className="text-lg font-heading">Model Performance Metrics</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow className="hover:bg-transparent border-white/5">
                <TableHead>Cohort</TableHead>
                <TableHead>Sample Size</TableHead>
                <TableHead>Sepsis Prevalence</TableHead>
                <TableHead>Recall (Sensitivity)</TableHead>
                <TableHead>Precision (PPV)</TableHead>
                <TableHead className="text-right">Status</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {loading ? (
                [1, 2, 3].map(i => (
                  <TableRow key={i}><TableCell colSpan={6}><Skeleton className="h-8 w-full" /></TableCell></TableRow>
                ))
              ) : (
                data.map((row, i) => (
                  <TableRow key={i} className="border-white/5 hover:bg-white/5 transition-colors">
                    <TableCell className="font-medium">{row.cohort}</TableCell>
                    <TableCell>{row.count}</TableCell>
                    <TableCell>{((row.sepsis_cases / row.count) * 100).toFixed(1)}%</TableCell>
                    <TableCell>{(row.recall * 100).toFixed(1)}%</TableCell>
                    <TableCell>{(row.precision * 100).toFixed(1)}%</TableCell>
                    <TableCell className="text-right">
                      {row.recall > 0.8 ? (
                        <Badge variant="outline" className="bg-emerald-500/10 text-emerald-500 border-emerald-500/20">Optimal</Badge>
                      ) : (
                        <Badge variant="outline" className="bg-amber-500/10 text-amber-500 border-amber-500/20">Review</Badge>
                      )}
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
}
