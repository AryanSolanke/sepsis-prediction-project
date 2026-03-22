import React, { useEffect, useState } from 'react';
import { rpcCall } from '../api';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/card';
import { Badge } from '../components/ui/badge';
import { Skeleton } from '../components/ui/skeleton';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { Activity, Users, Percent, ShieldAlert, Target } from 'lucide-react';
import { cn } from '../lib/utils';

export function Overview() {
  const [stats, setStats] = useState<any>(null);
  const [features, setFeatures] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchData() {
      try {
        console.log("[FETCH_START] Overview stats and features");
        const [statsData, featuresData] = await Promise.all([
          rpcCall({ func: 'get_summary_stats' }),
          rpcCall({ func: 'get_feature_importance', args: { top_n: 8 } })
        ]);
        setStats(statsData);
        setFeatures(featuresData);
        console.log("[FETCH_RESPONSE] Overview data received");
      } catch (error) {
        console.error("[PARSE_ERROR] Failed to fetch overview data", error);
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <Skeleton key={i} className="h-32 w-full rounded-xl" />
          ))}
        </div>
        <Skeleton className="h-[400px] w-full rounded-xl" />
      </div>
    );
  }

  const statCards = [
    {
      label: 'Sepsis Prevalence',
      value: `${(stats?.sepsis_rate * 100).toFixed(1)}%`,
      icon: ShieldAlert,
      color: 'text-rose-500',
      bg: 'bg-rose-500/10',
      description: 'Active cases in dataset'
    },
    {
      label: 'Total Patients',
      value: stats?.total_patients?.toLocaleString(),
      icon: Users,
      color: 'text-blue-500',
      bg: 'bg-blue-500/10',
      description: 'Unique ICU records'
    },
    {
      label: 'Model AUC',
      value: stats?.auc?.toFixed(3),
      icon: Target,
      color: 'text-emerald-500',
      bg: 'bg-emerald-500/10',
      description: 'Discriminative power'
    },
    {
      label: 'Avg. Patient Age',
      value: stats?.avg_age?.toFixed(1),
      icon: Activity,
      color: 'text-amber-500',
      bg: 'bg-amber-500/10',
      description: 'Years'
    }
  ];

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {statCards.map((stat, i) => (
          <Card key={i} className="overflow-hidden border-white/10 bg-card/60 backdrop-blur-sm shadow-xl">
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div className={cn("rounded-xl p-3", stat.bg)}>
                  <stat.icon className={cn("h-6 w-6", stat.color)} />
                </div>
                {i === 2 && <Badge variant="outline" className="bg-emerald-500/10 text-emerald-500 border-emerald-500/20">High Performance</Badge>}
              </div>
              <div className="mt-4 space-y-1">
                <h3 className="text-sm font-medium text-muted-foreground">{stat.label}</h3>
                <div className="text-3xl font-bold tracking-tight font-heading">{stat.value}</div>
                <p className="text-xs text-muted-foreground">{stat.description}</p>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <Card className="border-white/10 bg-card/60 backdrop-blur-sm shadow-xl">
        <CardHeader>
          <CardTitle className="font-heading flex items-center gap-2 text-xl">
            <Activity className="h-5 w-5 text-primary" />
            Top Feature Importance
          </CardTitle>
          <CardDescription>
            Relative contribution of physiological markers to the model's predictive output.
          </CardDescription>
        </CardHeader>
        <CardContent className="pt-4">
          <div className="h-[350px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={features}
                layout="vertical"
                margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="rgba(255,255,255,0.05)" />
                <XAxis type="number" hide />
                <YAxis 
                  dataKey="feature" 
                  type="category" 
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 12 }}
                  width={100}
                />
                <Tooltip 
                  cursor={{ fill: 'rgba(255,255,255,0.05)' }}
                  contentStyle={{ 
                    backgroundColor: 'rgba(9, 9, 11, 0.95)', 
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderRadius: '8px',
                    fontSize: '12px'
                  }}
                  itemStyle={{ color: 'hsl(var(--primary))' }}
                />
                <Bar 
                  dataKey="importance" 
                  radius={[0, 4, 4, 0]} 
                  barSize={32}
                >
                  {features.map((_, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={`hsl(var(--chart-${(index % 5) + 1}))`}
                      fillOpacity={0.8}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
