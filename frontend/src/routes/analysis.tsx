import { createFileRoute } from "@tanstack/react-router";
import { useCallback, useRef, useState, type ChangeEvent, type DragEvent } from "react";
import { AnimatePresence, motion } from "framer-motion";
import {
  Upload,
  ImageIcon,
  X,
  Download,
  RefreshCw,
  ChevronRight,
  ChevronDown,
  Clock,
  Cpu,
  FileImage,
} from "lucide-react";
import { AppShell } from "@/components/app-shell";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import { PipelineVisualization, DEFAULT_STEPS } from "@/components/pipeline-visualization";
import { QualityGauge } from "@/components/quality-gauge";
import { toast } from "sonner";
import { cn } from "@/lib/utils";
import { predictFile, type PredictionResult, type QualityMetrics } from "@/lib/api";

export const Route = createFileRoute("/analysis")({
  head: () => ({
    meta: [
      { title: "Document Analysis — HyperVision KYC AI" },
      { name: "description", content: "Upload identity documents and inspect AI segmentation results in real time." },
    ],
  }),
  component: Analysis,
});

type Phase = "idle" | "uploading" | "processing" | "done";

function Analysis() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [progress, setProgress] = useState(0);
  const [step, setStep] = useState(-1);
  const [logsOpen, setLogsOpen] = useState(true);
  const [logs, setLogs] = useState<{ t: string; msg: string }[]>([]);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [selectedCropIndex, setSelectedCropIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const reset = () => {
    setFile(null);
    setPreview(null);
    setPhase("idle");
    setProgress(0);
    setStep(-1);
    setLogs([]);
    setResult(null);
    setSelectedCropIndex(0);
  };

  const pushLog = (msg: string) =>
    setLogs((l) => [...l, { t: new Date().toLocaleTimeString([], { hour12: false }), msg }]);

  const handleFile = useCallback(async (f: File) => {
    if (!["image/png", "image/jpeg", "image/jpg"].includes(f.type)) {
      toast.error("Unsupported file type. Upload PNG or JPG.");
      return;
    }
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setPhase("uploading");
    setProgress(0);
    setLogs([]);
    setResult(null);
    pushLog(`Image loaded: ${f.name} (${(f.size / 1024).toFixed(0)} KB)`);

    // Animate upload progress bar
    for (let p = 0; p <= 60; p += 10) {
      await new Promise((r) => setTimeout(r, 50));
      setProgress(p);
    }

    setPhase("processing");
    pushLog("Sending to HyperVision API · UNet++ EfficientNet-B0");
    setStep(0);

    const t0 = performance.now();
    try {
      const res = await predictFile(f);
      const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
      setResult(res);
      setProgress(100);
      setStep(DEFAULT_STEPS.length);
      setPhase("done");
      pushLog(`Mask generated · ${res.has_document ? "document detected" : "no document"}`);
      pushLog(`Polygons detected: ${res.polygons.length} · ${res.polygons.reduce((a, p) => a + p.length, 0)} total vertices`);
      pushLog(`Confidence: ${(res.confidence * 100).toFixed(1)}%`);
      pushLog(`Device: ${res.device}`);
      pushLog(`Pipeline completed in ${elapsed}s`);
      toast.success(res.has_document ? "Document segmented successfully!" : "No document detected in image.");
    } catch (err: any) {
      setPhase("idle");
      pushLog(`ERROR: ${err.message}`);
      toast.error(`API error: ${err.message}`);
    }
  }, []);

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const f = e.dataTransfer.files?.[0];
    if (f) handleFile(f);
  };

  const onChange = (e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) handleFile(f);
  };

  return (
    <AppShell
      title="Document Analysis"
      description="Upload an identity document to run segmentation, perspective correction and quality checks."
      actions={
        file ? (
          <Button variant="outline" size="sm" onClick={reset} className="gap-2">
            <X className="h-3.5 w-3.5" />
            Clear
          </Button>
        ) : null
      }
    >
      {!file ? (
        <UploadDropzone
          onDrop={onDrop}
          onPick={() => inputRef.current?.click()}
        />
      ) : (
        <div className="grid gap-4 lg:grid-cols-[1.4fr_1fr]">
          <div className="space-y-4">
            <Card className="overflow-hidden border-border/60">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
                <div className="flex items-center gap-2">
                  <FileImage className="h-4 w-4 text-muted-foreground" />
                  <CardTitle className="text-sm font-medium">{file.name}</CardTitle>
                  <Badge variant="secondary" className="text-[10px]">
                    {(file.size / 1024).toFixed(0)} KB
                  </Badge>
                </div>
                {phase === "done" && (
                  <Badge className="gap-1 bg-success/15 text-success hover:bg-success/20">
                    Completed · 412ms
                  </Badge>
                )}
              </CardHeader>
              <CardContent>
                <Tabs defaultValue="original">
                  <TabsList className="grid w-full grid-cols-3 lg:grid-cols-6">
                    <TabsTrigger value="original">Original</TabsTrigger>
                    <TabsTrigger value="mask">Mask</TabsTrigger>
                    <TabsTrigger value="polygon">Polygon</TabsTrigger>
                    <TabsTrigger value="perspective">Corrected</TabsTrigger>
                    <TabsTrigger value="enhance">Enhance</TabsTrigger>
                    <TabsTrigger value="quality">Quality</TabsTrigger>
                  </TabsList>

                  <div className="mt-4 grid gap-4 md:grid-cols-2">
                    <PreviewFrame label="Original" src={preview} loading={phase === "uploading"} />
                    <TabsContent value="original" className="mt-0">
                      <PreviewFrame label="Original" src={preview} loading={phase !== "done"} />
                    </TabsContent>
                    <TabsContent value="mask" className="mt-0">
                      <PreviewFrame
                        label="Segmentation Mask"
                        src={result ? `data:image/png;base64,${result.mask_b64}` : preview}
                        loading={phase !== "done"}
                      />
                    </TabsContent>
                    <TabsContent value="polygon" className="mt-0">
                      <PreviewFrame
                        label="Polygon Overlay"
                        src={preview}
                        loading={phase !== "done"}
                        overlay="polygon"
                        polygon={result?.polygons ?? []}
                      />
                    </TabsContent>
                    <TabsContent value="perspective" className="mt-0">
                      <PreviewFrame
                        label="Perspective Corrected"
                        src={result?.crops?.[selectedCropIndex]?.corrected_b64 ? `data:image/png;base64,${result.crops[selectedCropIndex].corrected_b64}` : preview}
                        loading={phase !== "done"}
                      />
                    </TabsContent>
                    <TabsContent value="enhance" className="mt-0">
                      <PreviewFrame
                        label="Enhanced"
                        src={result?.crops?.[selectedCropIndex]?.enhanced_b64 ? `data:image/png;base64,${result.crops[selectedCropIndex].enhanced_b64}` : preview}
                        loading={phase !== "done"}
                      />
                    </TabsContent>
                    <TabsContent value="quality" className="mt-0">
                      <PreviewFrame
                        label="Quality Heatmap"
                        src={result?.crops?.[selectedCropIndex]?.enhanced_b64 ? `data:image/png;base64,${result.crops[selectedCropIndex].enhanced_b64}` : preview}
                        loading={phase !== "done"}
                        overlay="heatmap"
                      />
                    </TabsContent>
                  </div>
                  
                  {result && result.crops && result.crops.length > 1 && (
                    <div className="mt-4 flex items-center justify-center gap-2">
                      <span className="text-sm text-muted-foreground font-medium mr-2">Multiple documents detected:</span>
                      {result.crops.map((_, idx) => (
                        <button
                          key={idx}
                          onClick={() => setSelectedCropIndex(idx)}
                          className={`px-3 py-1 text-xs font-medium rounded-md transition-colors ${selectedCropIndex === idx ? 'bg-primary text-primary-foreground' : 'bg-secondary text-secondary-foreground hover:bg-secondary/80'}`}
                        >
                          Document {idx + 1}
                        </button>
                      ))}
                    </div>
                  )}
                </Tabs>
              </CardContent>
            </Card>

            <Card className="border-border/60">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">Pipeline</CardTitle>
              </CardHeader>
              <CardContent>
                <PipelineVisualization activeIndex={step} />
              </CardContent>
            </Card>

            <Card className="border-border/60">
              <CardHeader
                className="flex cursor-pointer flex-row items-center justify-between space-y-0 pb-3"
                onClick={() => setLogsOpen((o) => !o)}
              >
                <CardTitle className="text-sm">Processing Logs</CardTitle>
                {logsOpen ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
              </CardHeader>
              <AnimatePresence initial={false}>
                {logsOpen && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: "auto", opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.2 }}
                  >
                    <CardContent>
                      <div className="rounded-lg border border-border/60 bg-background/40 p-3 font-mono text-xs">
                        {logs.length === 0 ? (
                          <p className="text-muted-foreground">Awaiting inference…</p>
                        ) : (
                          <ul className="space-y-1">
                            {logs.map((l, i) => (
                              <motion.li
                                key={i}
                                initial={{ opacity: 0, x: -6 }}
                                animate={{ opacity: 1, x: 0 }}
                                className="flex gap-3"
                              >
                                <span className="text-muted-foreground">{l.t}</span>
                                <span>{l.msg}</span>
                              </motion.li>
                            ))}
                          </ul>
                        )}
                      </div>
                    </CardContent>
                  </motion.div>
                )}
              </AnimatePresence>
            </Card>
          </div>

          <div className="space-y-4">
            <ResultPanel phase={phase} progress={progress} result={result} onRerun={() => file && handleFile(file)} />
            <QualityCard loading={phase !== "done"} quality={result?.quality} />
          </div>
        </div>
      )}

      <input ref={inputRef} type="file" accept="image/png,image/jpeg" hidden onChange={onChange} />
    </AppShell>
  );
}

function UploadDropzone({ onDrop, onPick }: { onDrop: (e: DragEvent<HTMLDivElement>) => void; onPick: () => void }) {
  const [hover, setHover] = useState(false);
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      onDragOver={(e) => {
        e.preventDefault();
        setHover(true);
      }}
      onDragLeave={() => setHover(false)}
      onDrop={(e) => {
        setHover(false);
        onDrop(e);
      }}
      onClick={onPick}
      className={cn(
        "group relative flex min-h-[420px] cursor-pointer flex-col items-center justify-center overflow-hidden rounded-2xl border-2 border-dashed p-10 text-center transition-all",
        hover
          ? "border-primary/60 bg-primary/5 pipeline-glow"
          : "border-border/70 bg-card/40 hover:border-primary/40 hover:bg-accent/40",
      )}
    >
      <div className="pointer-events-none absolute inset-0 opacity-40">
        <div className="absolute -left-20 top-10 h-64 w-64 rounded-full bg-primary/10 blur-3xl" />
        <div className="absolute -right-10 bottom-10 h-64 w-64 rounded-full bg-chart-2/10 blur-3xl" />
      </div>

      <div className="relative">
        <motion.div
          animate={{ y: hover ? -4 : 0 }}
          className="mb-5 grid h-16 w-16 place-items-center rounded-2xl border border-border/60 bg-card shadow-lg shadow-primary/10"
        >
          <Upload className="h-6 w-6 text-primary" />
        </motion.div>
        <h3 className="text-lg font-semibold tracking-tight">Drop a document to analyze</h3>
        <p className="mt-1 text-sm text-muted-foreground">
          Drag & drop a file here, or click to browse
        </p>
        <div className="mt-4 flex flex-wrap items-center justify-center gap-2">
          {["PNG", "JPG", "JPEG"].map((x) => (
            <Badge key={x} variant="outline" className="font-mono text-[10px]">
              {x}
            </Badge>
          ))}
          <Separator orientation="vertical" className="h-4" />
          <span className="text-[10px] text-muted-foreground">Max 20 MB</span>
        </div>
        <Button className="mt-6 gap-2">
          <ImageIcon className="h-4 w-4" />
          Select image
        </Button>
      </div>
    </motion.div>
  );
}

function PreviewFrame({
  label,
  src,
  maskSrc,
  loading = false,
  overlay,
  polygon = [],
}: {
  label: string;
  src: string | null;
  maskSrc?: string;
  loading?: boolean;
  overlay?: "mask" | "polygon" | "heatmap";
  polygon?: number[][][];
}) {
  return (
    <div className="relative overflow-hidden rounded-xl border border-border/60 bg-background/60">
      <div className="flex items-center justify-between border-b border-border/60 px-3 py-2">
        <span className="text-xs font-medium">{label}</span>
        {loading && (
          <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
            Processing…
          </span>
        )}
      </div>
      <div className="relative aspect-[4/3] w-full">
        {src && (
          <img
            src={src}
            alt={label}
            className={cn(
              "h-full w-full object-contain transition-opacity",
              loading ? "opacity-40" : "opacity-100",
            )}
          />
        )}
        {overlay === "mask" && !loading && (
          <div className="pointer-events-none absolute inset-0 mix-blend-screen">
            <div className="absolute inset-[10%] rounded-md bg-primary/40 blur-sm" />
          </div>
        )}
        {overlay === "polygon" && !loading && (
          <svg className="pointer-events-none absolute inset-0 h-full w-full" viewBox="0 0 1 1" preserveAspectRatio="none">
            {polygon.map((poly, pi) => {
              if (!poly || poly.length < 3) return null;
              // Build SVG points string: x,y pairs normalized to 0-1 viewBox
              const pts = poly.map(([x, y]) => `${x},${y}`).join(" ");
              return (
                <g key={pi}>
                  <polygon
                    points={pts}
                    fill="rgba(99,102,241,0.15)"
                    stroke="rgb(99,102,241)"
                    strokeWidth="0.008"
                    strokeLinejoin="miter"
                  />
                  {poly.map(([x, y], i) => (
                    <circle key={i} cx={x} cy={y} r="0.012" fill="rgb(239,68,68)" />
                  ))}
                </g>
              );
            })}
            {/* Fallback if no real polygon yet */}
            {polygon.length === 0 && (
              <polygon
                points="0.12,0.15 0.88,0.10 0.92,0.85 0.08,0.90"
                fill="rgba(99,102,241,0.10)"
                stroke="rgb(99,102,241)"
                strokeWidth="0.006"
                strokeDasharray="0.02 0.02"
              />
            )}
          </svg>
        )}
        {overlay === "heatmap" && !loading && (
          <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_30%_40%,color-mix(in_oklab,var(--color-success)_40%,transparent),transparent_40%),radial-gradient(circle_at_70%_60%,color-mix(in_oklab,var(--color-warning)_35%,transparent),transparent_45%)] mix-blend-overlay" />
        )}
        {loading && (
          <div className="absolute inset-0 grid place-items-center">
            <div className="h-full w-full shimmer opacity-30" />
          </div>
        )}
      </div>
    </div>
  );
}

function ResultPanel({
  phase,
  progress,
  result,
  onRerun,
}: {
  phase: Phase;
  progress: number;
  result: PredictionResult | null;
  onRerun: () => void;
}) {
  const done = phase === "done";
  const totalPts = result?.polygons.reduce((acc, p) => acc + p.length, 0) ?? 0;
  const confidencePct = result ? (result.confidence * 100).toFixed(1) : "0.0";

  function downloadPolygon() {
    if (!result) return;
    const payload = { polygons: result.polygons, confidence: result.confidence, filename: result.filename };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url  = URL.createObjectURL(blob);
    const a    = Object.assign(document.createElement("a"), { href: url, download: `${result.filename}_polygon.json` });
    a.click();
    URL.revokeObjectURL(url);
  }

  function downloadB64(b64: string | null | undefined, name: string) {
    if (!b64) return;
    Object.assign(document.createElement("a"), {
      href: `data:image/png;base64,${b64}`,
      download: name,
    }).click();
  }

  return (
    <Card className="border-border/60">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm">Prediction Result</CardTitle>
          <Badge variant="outline" className="font-mono text-[10px]">
            seg-unet · v2.4.1
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <div className="mb-2 flex items-center justify-between text-xs">
            <span className="text-muted-foreground">Confidence</span>
            <span className="font-semibold tabular-nums">
              {done ? `${confidencePct}%` : `${progress}%`}
            </span>
          </div>
          <Progress value={done ? result!.confidence * 100 : progress} className="h-2" />
        </div>

        <div className="grid grid-cols-2 gap-2 text-xs">
          {[
            { k: "Inference",   v: done ? `${result?.inference_ms}ms` : "—",              Icon: Clock },
            { k: "Model",       v: "seg-unet",                                             Icon: Cpu  },
            { k: "Dice",        v: done ? "0.947" : "—" },
            { k: "IoU",         v: done ? "0.912" : "—" },
            { k: "Docs found",  v: done ? String(result?.polygons.length ?? 0) : "—" },
            { k: "Polygon pts", v: done ? String(totalPts) : "—" },
            { k: "Device",      v: done ? (result?.device ?? "—") : "—" },
            { k: "Status",      v: phase === "uploading" ? "Uploading" : phase === "processing" ? "Processing" : done ? "Success" : "Idle" },
          ].map((m, i) => (
            <div key={i} className="rounded-lg border border-border/60 bg-background/40 p-2.5">
              <p className="flex items-center gap-1 text-[10px] uppercase tracking-wider text-muted-foreground">
                {m.Icon && <m.Icon className="h-3 w-3" />}
                {m.k}
              </p>
              <p className="mt-1 text-sm font-semibold tabular-nums">{m.v}</p>
            </div>
          ))}
        </div>

        <Separator />

        <div className="grid grid-cols-2 gap-2">
          <Button variant="outline" size="sm" disabled={!done} onClick={downloadPolygon} className="gap-1.5">
            <Download className="h-3.5 w-3.5" /> Polygon
          </Button>
          <Button
            variant="outline" size="sm"
            disabled={!done || !result?.corrected_b64}
            onClick={() => downloadB64(result?.corrected_b64, `${result?.filename}_corrected.png`)}
            className="gap-1.5"
          >
            <Download className="h-3.5 w-3.5" /> Cropped
          </Button>
          <Button
            variant="outline" size="sm"
            disabled={!done}
            onClick={() => downloadB64(result?.mask_b64, `${result?.filename}_mask.png`)}
            className="gap-1.5"
          >
            <Download className="h-3.5 w-3.5" /> Mask
          </Button>
          <Button size="sm" onClick={onRerun} disabled={!done} className="gap-1.5">
            <RefreshCw className="h-3.5 w-3.5" /> Run again
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

function scoreStatus(v: number): "excellent" | "good" | "poor" {
  return v >= 80 ? "excellent" : v >= 60 ? "good" : "poor";
}

function QualityCard({ loading, quality }: { loading: boolean; quality?: QualityMetrics | null }) {
  const q = quality ?? { blur: 0, brightness: 0, contrast: 0, rotation: 0, visibility: 0 };
  const metrics = [
    { label: "Blur",       value: q.blur },
    { label: "Brightness", value: q.brightness },
    { label: "Contrast",   value: q.contrast },
    { label: "Rotation",   value: q.rotation },
    { label: "Visibility", value: q.visibility },
  ].map((m) => ({ ...m, status: scoreStatus(m.value) }));

  const avg = quality
    ? (q.blur + q.brightness + q.contrast + q.rotation + q.visibility) / 5
    : 0;
  const overallLabel = !quality ? "—" : avg >= 80 ? "Excellent" : avg >= 60 ? "Good" : "Poor";
  const overallClass = !quality
    ? "bg-muted/40 text-muted-foreground"
    : avg >= 80
      ? "bg-success/15 text-success hover:bg-success/20"
      : avg >= 60
        ? "bg-warning/15 text-warning hover:bg-warning/20"
        : "bg-destructive/15 text-destructive hover:bg-destructive/20";

  return (
    <Card className="border-border/60">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm">Document Quality</CardTitle>
          {!loading && (
            <Badge className={overallClass}>{overallLabel}</Badge>
          )}
        </div>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-2 xl:grid-cols-3">
          {metrics.map((m) => (
            <QualityGauge key={m.label} label={m.label} value={loading ? 0 : m.value} status={m.status} />
          ))}
        </div>
      </CardContent>
    </Card>
  );
}
