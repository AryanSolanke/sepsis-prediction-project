import React, { useEffect, useState } from 'react';
import { rpcCall } from '../api';
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from '../components/ui/card';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { Button } from '../components/ui/button';
import { Badge } from '../components/ui/badge';
import { Progress } from '../components/ui/progress';
import { Spinner } from '../components/ui/spinner';
import { ScrollArea } from '../components/ui/scroll-area';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../components/ui/select';
import { Alert, AlertDescription, AlertTitle } from '../components/ui/alert';
import { Skeleton } from '../components/ui/skeleton';
import { Activity, AlertTriangle, CheckCircle2, Heart, User } from 'lucide-react';
import { cn } from '../lib/utils';

type PredictionOption = {
  label: string;
  value: number;
};

type PredictionField = {
  name: string;
  label: string;
  group: string;
  kind: 'number' | 'select';
  unit?: string | null;
  default_value: number;
  min_value: number;
  max_value: number;
  step: number;
  options: PredictionOption[];
};

type PredictionGroup = {
  id: string;
  label: string;
  description: string;
};

type PredictionSchema = {
  model_name: string;
  feature_count: number;
  primary_feature_count: number;
  measurement_flag_count: number;
  input_groups: PredictionGroup[];
  fields: PredictionField[];
};

type PredictionResult = {
  sepsis_probability: number;
  risk_level: 'Low' | 'Medium' | 'High';
  top_contributors: Array<{
    feature: string;
    value: number | null;
    contribution_score: number;
  }>;
};

const sectionIcons: Record<string, typeof Activity> = {
  patient_context: User,
  vital_signs: Heart,
  lab_markers: Activity,
};

function formatNumericValue(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return 'Missing';
  if (Number.isInteger(value)) return String(value);
  return value.toFixed(Math.abs(value) >= 100 ? 1 : 2).replace(/\.0+$/, '').replace(/(\.\d*[1-9])0+$/, '$1');
}

function humanizeFeatureName(name: string): string {
  if (name.endsWith('_Measured')) {
    return `${humanizeFeatureName(name.slice(0, -'_Measured'.length))} Measured`;
  }

  return name
    .replace(/_/g, ' ')
    .replace(/([a-z0-9])([A-Z])/g, '$1 $2')
    .trim();
}

export function RiskPredictor() {
  const [schema, setSchema] = useState<PredictionSchema | null>(null);
  const [formData, setFormData] = useState<Record<string, string>>({});
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loadingSchema, setLoadingSchema] = useState(true);
  const [loadingPrediction, setLoadingPrediction] = useState(false);
  const [schemaError, setSchemaError] = useState<string | null>(null);
  const [predictionError, setPredictionError] = useState<string | null>(null);

  useEffect(() => {
    async function loadSchema() {
      setLoadingSchema(true);
      setSchemaError(null);

      try {
        const response = await rpcCall<PredictionSchema>({ func: 'get_prediction_schema' });
        const defaultValues = Object.fromEntries(
          response.fields.map((field) => [field.name, String(field.default_value)])
        ) as Record<string, string>;

        setSchema(response);
        setFormData(defaultValues);
      } catch (error) {
        console.error('[PARSE_ERROR] Failed to load prediction schema', error);
        setSchemaError(error instanceof Error ? error.message : 'Unable to load the prediction schema.');
      } finally {
        setLoadingSchema(false);
      }
    }

    loadSchema();
  }, []);

  const fieldMap: Record<string, PredictionField> = {};
  (schema?.fields ?? []).forEach((field) => {
    fieldMap[field.name] = field;
  });

  const sections = (schema?.input_groups ?? [])
    .map((group) => ({
      ...group,
      fields: (schema?.fields ?? []).filter((field) => field.group === group.id),
    }))
    .filter((group) => group.fields.length > 0);

  const handleFieldChange = (fieldName: string, rawValue: string) => {
    setFormData((prev) => ({
      ...prev,
      [fieldName]: rawValue,
    }));
  };

  const buildPatientPayload = (): Record<string, number | null> => {
    if (!schema) return {};

    return Object.fromEntries(
      schema.fields.map((field) => {
        const rawValue = (formData[field.name] ?? '').trim();
        return [field.name, rawValue === '' ? null : Number(rawValue)];
      })
    ) as Record<string, number | null>;
  };

  const handlePredict = async () => {
    if (!schema) return;

    setLoadingPrediction(true);
    setPredictionError(null);

    try {
      const response = await rpcCall<PredictionResult>({
        func: 'predict_sepsis',
        args: { patient_data: buildPatientPayload() },
      });
      setResult(response);
      console.log('[FETCH_RESPONSE] Prediction with auto-derived flags successful');
    } catch (error) {
      console.error('[PARSE_ERROR] Prediction failed', error);
      setPredictionError(error instanceof Error ? error.message : 'Prediction failed.');
    } finally {
      setLoadingPrediction(false);
    }
  };

  const getContributorLabel = (feature: string) => fieldMap[feature]?.label || humanizeFeatureName(feature);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 animate-in fade-in slide-in-from-right-4 duration-500">
      <Card className="border-white/10 bg-card/60 backdrop-blur-sm shadow-xl">
        <CardHeader className="space-y-4">
          <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
            <div className="space-y-1">
              <CardTitle className="font-heading text-xl">Patient Feature Intake</CardTitle>
              <CardDescription>
                Defaults are prefilled from the training data. Edit the values you have, clear anything unavailable, and the backend will derive the rest automatically.
              </CardDescription>
            </div>
            {schema && (
              <div className="flex flex-wrap gap-2">
                <Badge variant="outline" className="border-primary/20 bg-primary/10 text-primary">
                  {schema.model_name}
                </Badge>
                <Badge variant="outline" className="border-white/10 bg-white/5">
                  {schema.primary_feature_count} inputs
                </Badge>
              </div>
            )}
          </div>
        </CardHeader>

        <CardContent className="space-y-6">
          {loadingSchema && (
            <div className="space-y-4">
              <Skeleton className="h-24 w-full rounded-xl" />
              <Skeleton className="h-[520px] w-full rounded-xl" />
            </div>
          )}

          {!loadingSchema && schemaError && (
            <Alert variant="destructive">
              <AlertTriangle className="h-4 w-4" />
              <AlertTitle>Schema load failed</AlertTitle>
              <AlertDescription>{schemaError}</AlertDescription>
            </Alert>
          )}

          {!loadingSchema && schema && (
            <>
              <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
                <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                  <div className="space-y-1">
                    <p className="text-sm font-semibold">Prediction Payload</p>
                    <p className="text-xs text-muted-foreground">
                      Clear any field you do not have. Missing values are sent as missing, and the backend derives the measured flags without exposing extra controls.
                    </p>
                  </div>
                  <div className="text-left md:text-right">
                    <div className="text-2xl font-bold font-heading">{schema.feature_count}</div>
                    <p className="text-xs text-muted-foreground">
                      {schema.primary_feature_count} direct inputs + {schema.measurement_flag_count} auto-derived flags
                    </p>
                  </div>
                </div>
              </div>

              <ScrollArea className="h-[64vh] pr-4">
                <div className="space-y-8 pr-2">
                  {sections.map((section) => {
                    const SectionIcon = sectionIcons[section.id] || Activity;

                    return (
                      <section key={section.id} className="space-y-4">
                        <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
                          <div className="space-y-1">
                            <div className="flex items-center gap-2">
                              <SectionIcon className="h-4 w-4 text-primary" />
                              <h3 className="text-sm font-semibold uppercase tracking-[0.2em] text-muted-foreground">
                                {section.label}
                              </h3>
                            </div>
                            <p className="text-sm text-muted-foreground">{section.description}</p>
                          </div>
                          <Badge variant="outline" className="w-fit border-white/10 bg-white/5">
                            {section.fields.length} inputs
                          </Badge>
                        </div>

                        <div className="grid grid-cols-1 2xl:grid-cols-2 gap-4">
                          {section.fields.map((field) => {
                            const currentValue = formData[field.name] ?? '';

                            return (
                              <div
                                key={field.name}
                                className="rounded-2xl border border-white/10 bg-black/10 p-4 space-y-3 shadow-inner"
                              >
                                <div className="space-y-1">
                                  <Label htmlFor={field.name} className="text-sm font-semibold">
                                    {field.label}
                                  </Label>
                                  <p className="break-words text-xs text-muted-foreground">
                                    • {field.unit || (field.kind === 'select' ? 'categorical value' : 'numeric value')}
                                  </p>
                                </div>

                                {field.kind === 'select' ? (
                                  <Select
                                    value={currentValue}
                                    onValueChange={(value) => handleFieldChange(field.name, value === '__missing__' ? '' : value)}
                                  >
                                    <SelectTrigger className="w-full border-white/10 bg-white/5">
                                      <SelectValue placeholder={`Not provided (${field.label})`} />
                                    </SelectTrigger>
                                    <SelectContent>
                                      <SelectItem value="__missing__">Not provided</SelectItem>
                                      {field.options.map((option) => (
                                        <SelectItem key={`${field.name}-${option.value}`} value={String(option.value)}>
                                          {option.label}
                                        </SelectItem>
                                      ))}
                                    </SelectContent>
                                  </Select>
                                ) : (
                                  <Input
                                    id={field.name}
                                    name={field.name}
                                    type="number"
                                    min={field.min_value}
                                    max={field.max_value}
                                    step={field.step}
                                    value={currentValue}
                                    onChange={(event) => handleFieldChange(field.name, event.target.value)}
                                    className="bg-white/5 border-white/10 focus:ring-primary/50"
                                  />
                                )}
                              </div>
                            );
                          })}
                        </div>
                      </section>
                    );
                  })}
                </div>
              </ScrollArea>

              {predictionError && (
                <Alert variant="destructive">
                  <AlertTriangle className="h-4 w-4" />
                  <AlertTitle>Prediction failed</AlertTitle>
                  <AlertDescription>{predictionError}</AlertDescription>
                </Alert>
              )}
            </>
          )}
        </CardContent>

        <CardFooter>
          <Button
            onClick={handlePredict}
            disabled={loadingPrediction || loadingSchema || !schema}
            className="w-full h-11 text-base font-semibold transition-all hover:scale-[1.01] active:scale-[0.99]"
          >
            {loadingPrediction ? <Spinner className="mr-2 h-4 w-4" /> : <Activity className="mr-2 h-4 w-4" />}
            Predict Sepsis
          </Button>
        </CardFooter>
      </Card>

      <div className="space-y-6">
        {!result && !loadingPrediction && (
          <Card className="h-full border-dashed border-white/10 bg-white/5 flex flex-col items-center justify-center p-8 text-center space-y-4">
            <div className="rounded-full bg-primary/10 p-4">
              <Activity className="h-8 w-8 text-primary/40" />
            </div>
            <div className="space-y-2">
              <h3 className="text-lg font-medium">Awaiting Feature Values</h3>
              <p className="text-sm text-muted-foreground max-w-[340px]">
                The form is prefilled with sensible defaults. Adjust the values you have, clear anything missing, and run the prediction when ready.
              </p>
            </div>
          </Card>
        )}

        {loadingPrediction && (
          <Card className="h-full border-white/10 bg-card/60 backdrop-blur-sm p-8 flex flex-col items-center justify-center space-y-4">
            <Spinner className="h-8 w-8 text-primary" />
            <p className="text-sm text-muted-foreground animate-pulse">Analyzing the patient profile...</p>
          </Card>
        )}

        {result && !loadingPrediction && (
          <div className="space-y-6">
            <Card
              className={cn(
                'border-white/10 bg-card/60 backdrop-blur-sm shadow-xl overflow-hidden relative',
                result.risk_level === 'High'
                  ? 'ring-1 ring-rose-500/50'
                  : result.risk_level === 'Medium'
                  ? 'ring-1 ring-amber-500/50'
                  : 'ring-1 ring-emerald-500/50'
              )}
            >
              <div
                className={cn(
                  'absolute top-0 left-0 w-full h-1',
                  result.risk_level === 'High'
                    ? 'bg-rose-500'
                    : result.risk_level === 'Medium'
                    ? 'bg-amber-500'
                    : 'bg-emerald-500'
                )}
              />
              <CardHeader>
                <div className="flex items-center justify-between gap-3">
                  <CardTitle className="text-lg font-heading">AI Sepsis Risk Assessment</CardTitle>
                  <Badge
                    className={cn(
                      'px-3 py-1',
                      result.risk_level === 'High'
                        ? 'bg-rose-500/20 text-rose-500 border-rose-500/20'
                        : result.risk_level === 'Medium'
                        ? 'bg-amber-500/20 text-amber-500 border-amber-500/20'
                        : 'bg-emerald-500/20 text-emerald-500 border-emerald-500/20'
                    )}
                  >
                    {result.risk_level} Risk
                  </Badge>
                </div>
                {schema && (
                  <CardDescription>
                    Based on {schema.primary_feature_count} visible inputs with {schema.measurement_flag_count} derived model flags.
                  </CardDescription>
                )}
              </CardHeader>
              <CardContent className="space-y-8">
                <div className="text-center space-y-2">
                  <div className="text-5xl font-bold font-heading">
                    {(result.sepsis_probability * 100).toFixed(1)}%
                  </div>
                  <p className="text-sm text-muted-foreground">Sepsis Probability</p>
                  <Progress value={result.sepsis_probability * 100} className="h-2 w-full mt-4" />
                </div>

                <div className="space-y-4">
                  <h4 className="text-sm font-semibold flex items-center gap-2">
                    <AlertTriangle className="h-4 w-4 text-amber-500" />
                    Top Risk Contributors
                  </h4>
                  <div className="space-y-3">
                    {result.top_contributors.map((contributor, index) => (
                      <div key={`${contributor.feature}-${index}`} className="space-y-1">
                        <div className="flex justify-between gap-3 text-xs">
                          <span className="font-medium">{getContributorLabel(contributor.feature)}</span>
                          <span className="text-muted-foreground">Value: {formatNumericValue(contributor.value)}</span>
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
                    ? 'Immediate clinical evaluation and initiation of sepsis protocols (fluids, antibiotics) is strongly recommended.'
                    : result.risk_level === 'Medium'
                    ? 'Increased monitoring frequency recommended. Review source of potential infection.'
                    : 'Patient shows stable vitals. Continue standard ICU monitoring protocols.'}
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
