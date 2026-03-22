import React, { useState, useEffect } from 'react';
import { LayoutDashboard, Users, Activity, Menu, X, ShieldAlert, Heart, Github, Terminal } from 'lucide-react';
import { RiPulseLine } from 'react-icons/ri';
import { SiPython, SiReact, SiScikitlearn } from 'react-icons/si';
import { Button } from './components/ui/button';
import { Badge } from './components/ui/badge';
import { Separator } from './components/ui/separator';
import { Overview } from './features/Overview';
import { CohortAnalysis } from './features/CohortAnalysis';
import { RiskPredictor } from './features/RiskPredictor';
import { cn } from './lib/utils';

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);

  useEffect(() => {
    console.log("RENDER_SUCCESS");
  }, []);

  const navItems = [
    { id: 'overview', label: 'Overview', icon: LayoutDashboard },
    { id: 'cohorts', label: 'Cohort Analysis', icon: Users },
    { id: 'predictor', label: 'Risk Predictor', icon: Activity },
  ];

  const renderContent = () => {
    switch (activeTab) {
      case 'overview': return <Overview />;
      case 'cohorts': return <CohortAnalysis />;
      case 'predictor': return <RiskPredictor />;
      default: return <Overview />;
    }
  };

  return (
    <div className="flex h-screen overflow-hidden bg-background bg-grid-fade font-sans text-foreground selection:bg-primary/30 selection:text-primary-foreground">
      {/* Sidebar - Desktop */}
      <aside 
        className={cn(
          "hidden md:flex flex-col border-r border-white/10 bg-card/40 backdrop-blur-xl transition-all duration-300 ease-in-out z-30",
          isSidebarOpen ? "w-72" : "w-20"
        )}
      >
        <div className="p-6 flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/20 ring-1 ring-primary/40 shadow-[0_0_15px_rgba(var(--primary-rgb),0.3)]">
            <RiPulseLine className="h-6 w-6 text-primary animate-pulse" />
          </div>
          {isSidebarOpen && (
            <div className="animate-in fade-in slide-in-from-left-2 duration-300">
              <h1 className="text-xl font-bold font-heading tracking-tight">SepsisGuard</h1>
              <p className="text-[10px] uppercase tracking-widest text-muted-foreground font-semibold">Diagnostic Intelligence</p>
            </div>
          )}
        </div>

        <nav className="flex-1 px-3 space-y-1">
          {navItems.map((item) => (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={cn(
                "w-full flex items-center gap-3 rounded-xl px-4 py-3 text-sm font-medium transition-all group relative",
                activeTab === item.id 
                  ? "bg-primary text-primary-foreground shadow-lg shadow-primary/20" 
                  : "text-muted-foreground hover:bg-white/5 hover:text-foreground"
              )}
            >
              <item.icon className={cn("h-5 w-5", activeTab === item.id ? "" : "group-hover:text-primary transition-colors")} />
              {isSidebarOpen && <span>{item.label}</span>}
              {!isSidebarOpen && (
                <div className="absolute left-full ml-4 px-2 py-1 bg-popover text-popover-foreground text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-50 pointer-events-none shadow-xl border border-white/10">
                  {item.label}
                </div>
              )}
            </button>
          ))}
        </nav>

        <div className="p-6">
          <Button 
            variant="ghost" 
            size="sm" 
            className="w-full justify-start text-muted-foreground"
            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
          >
            {isSidebarOpen ? <X className="mr-2 h-4 w-4" /> : <Menu className="h-4 w-4" />}
            {isSidebarOpen && "Collapse Menu"}
          </Button>
        </div>
      </aside>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col min-w-0 overflow-hidden relative">
        {/* Mobile Header */}
        <header className="md:hidden flex items-center justify-between p-4 border-b border-white/10 bg-card/60 backdrop-blur-md">
          <div className="flex items-center gap-2">
            <RiPulseLine className="h-6 w-6 text-primary" />
            <span className="font-bold font-heading">SepsisGuard</span>
          </div>
          <Button variant="ghost" size="icon" onClick={() => setIsSidebarOpen(!isSidebarOpen)}>
            <Menu className="h-6 w-6" />
          </Button>
        </header>

        {/* Page Content */}
        <div className="flex-1 overflow-y-auto custom-scrollbar">
          <div className="p-6 md:p-10 max-w-7xl mx-auto space-y-10 pb-20">
            {/* Header / Hero Area */}
            <div className="relative rounded-2xl overflow-hidden bg-slate-900 border border-white/10 shadow-2xl">
              <img 
                src="/assets/hero-icu-monitor.jpg" 
                alt="ICU Monitor Hero" 
                className="absolute inset-0 w-full h-full object-cover opacity-30 mix-blend-luminosity"
              />
              <div className="absolute inset-0 bg-gradient-to-r from-slate-950 via-slate-950/80 to-transparent" />
              <div className="relative p-8 md:p-12 space-y-4 max-w-2xl">
                <Badge variant="outline" className="bg-primary/10 text-primary border-primary/20 backdrop-blur-md">
                  ICU Diagnostic Support System
                </Badge>
                <h2 className="text-3xl md:text-4xl font-bold font-heading tracking-tight text-white">
                  Early Sepsis Detection Through AI Analysis
                </h2>
                <p className="text-slate-400 text-lg leading-relaxed">
                  Analyzing real-time physiological biomarkers and clinical trends to identify high-risk patients before symptoms escalate.
                </p>
                <div className="flex flex-wrap gap-4 pt-4">
                  <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 backdrop-blur-sm">
                    <ShieldAlert className="h-4 w-4 text-rose-500" />
                    <span className="text-xs font-medium text-slate-300">Predictive Accuracy: 84%</span>
                  </div>
                  <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-white/5 border border-white/10 backdrop-blur-sm">
                    <Heart className="h-4 w-4 text-blue-500" />
                    <span className="text-xs font-medium text-slate-300">24/7 Monitoring Enabled</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Active Feature Render */}
            <div className="min-h-[500px]">
              {renderContent()}
            </div>

            {/* Footer */}
            <footer className="pt-10 border-t border-white/5">
              <div className="flex flex-col md:flex-row items-center justify-between gap-6">
                <div className="flex flex-col items-center md:items-start gap-2">
                  <div className="flex items-center gap-2">
                    <RiPulseLine className="h-5 w-5 text-primary" />
                    <span className="font-heading font-bold">SepsisGuard AI</span>
                  </div>
                  <p className="text-xs text-muted-foreground">Professional Medical Diagnostic Interface v1.0.4</p>
                </div>
                
                <div className="flex items-center gap-8 opacity-60 hover:opacity-100 transition-opacity">
                   <div className="flex flex-col items-center gap-1">
                    <SiPython className="h-6 w-6" />
                    <span className="text-[10px] font-medium">Python</span>
                  </div>
                  <div className="flex flex-col items-center gap-1">
                    <SiScikitlearn className="h-6 w-6" />
                    <span className="text-[10px] font-medium">Scikit-learn</span>
                  </div>
                  <div className="flex flex-col items-center gap-1">
                    <SiReact className="h-6 w-6" />
                    <span className="text-[10px] font-medium">React</span>
                  </div>
                </div>

                <div className="flex items-center gap-4 text-xs text-muted-foreground">
                  <span>Privacy Policy</span>
                  <Separator orientation="vertical" className="h-3 bg-white/10" />
                  <span>Regulatory Compliance</span>
                  <Separator orientation="vertical" className="h-3 bg-white/10" />
                  <span className="flex items-center gap-1 text-primary">
                    <Terminal className="h-3 w-3" />
                    Medical AI
                  </span>
                </div>
              </div>
            </footer>
          </div>
        </div>

        {/* Mobile Nav Tabs */}
        <div className="md:hidden fixed bottom-0 left-0 right-0 h-16 bg-card/80 backdrop-blur-xl border-t border-white/10 flex items-center justify-around px-4 z-40">
          {navItems.map((item) => (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={cn(
                "flex flex-col items-center justify-center gap-1 px-4 py-1 rounded-lg transition-colors",
                activeTab === item.id ? "text-primary bg-primary/10" : "text-muted-foreground"
              )}
            >
              <item.icon className="h-5 w-5" />
              <span className="text-[10px] font-medium">{item.label}</span>
            </button>
          ))}
        </div>
      </main>
    </div>
  );
}
