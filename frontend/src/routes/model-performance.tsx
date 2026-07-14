import { createFileRoute } from "@tanstack/react-router";
import { AppShell } from "@/components/app-shell";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";

export const Route = createFileRoute("/model-performance")({
  head: () => ({
    meta: [
      { title: "Model Performance — HyperVision KYC AI" },
      { name: "description", content: "Analytics for production model performance, confidence and latency." },
    ],
  }),
  component: PerfPage,
});

const conf = Array.from({ length: 10 }).map((_, i) => ({
  bucket: `${i * 10}-${i * 10 + 10}`,
  count: Math.round(20 + Math.pow(i, 2.4) * 4 + (i > 8 ? 900 : 0)),
}));

const latency = Array.from({ length: 24 }).map((_, i) => ({
  hour: `${i}h`,
  p50: 380 + Math.round(Math.sin(i / 3) * 40 + Math.random() * 30),
  p95: 620 + Math.round(Math.cos(i / 4) * 60 + Math.random() * 40),
}));

const success = Array.from({ length: 14 }).map((_, i) => ({
  day: `D-${13 - i}`,
  rate: 96 + Math.random() * 3,
}));

const quality = Array.from({ length: 10 }).map((_, i) => ({
  bin: `${i * 10}`,
  n: Math.round(40 + Math.pow(i, 2) * 6 + (i > 7 ? 500 : 0)),
}));

function PerfPage() {
  return (
    <AppShell
      title="Model Performance"
      description="Registered model metrics, live inference distribution and quality drift."
    >
      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        {[
          { k: "Model", v: "seg-unet-v2.4.1", sub: "Production" },
          { k: "Trained", v: "Jul 12, 2026", sub: "MLflow" },
          { k: "Dataset", v: "hv-kyc-v9.2", sub: "412K samples" },
          { k: "Precision", v: "0.958", sub: "Val set" },
        ].map((c) => (
          <Card key={c.k} className="border-border/60">
            <CardContent className="p-5">
              <p className="text-xs uppercase tracking-wider text-muted-foreground">{c.k}</p>
              <p className="mt-1 text-lg font-semibold tracking-tight">{c.v}</p>
              <p className="text-xs text-muted-foreground">{c.sub}</p>
            </CardContent>
          </Card>
        ))}
      </section>

      <section className="mt-4 grid gap-4 md:grid-cols-4">
        {[
          { k: "Dice Score", v: "0.947" },
          { k: "IoU", v: "0.912" },
          { k: "Recall", v: "0.941" },
          { k: "F1", v: "0.949" },
        ].map((c) => (
          <Card key={c.k} className="border-border/60">
            <CardContent className="p-5">
              <p className="text-xs uppercase tracking-wider text-muted-foreground">{c.k}</p>
              <p className="mt-1 text-2xl font-semibold tabular-nums">{c.v}</p>
            </CardContent>
          </Card>
        ))}
      </section>

      <section className="mt-4 grid gap-4 lg:grid-cols-2">
        <Card className="border-border/60">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Prediction Confidence Distribution</CardTitle>
          </CardHeader>
          <CardContent className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={conf} margin={{ left: -20, right: 8, top: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                <XAxis dataKey="bucket" fontSize={10} tickLine={false} axisLine={false} />
                <YAxis fontSize={10} tickLine={false} axisLine={false} />
                <Tooltip contentStyle={{ background: "var(--color-popover)", border: "1px solid var(--color-border)", borderRadius: 8, fontSize: 12 }} />
                <Bar dataKey="count" fill="var(--color-primary)" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card className="border-border/60">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Inference Time (p50 / p95)</CardTitle>
          </CardHeader>
          <CardContent className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={latency} margin={{ left: -20, right: 8, top: 8 }}>
                <defs>
                  <linearGradient id="g1" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="var(--color-primary)" stopOpacity={0.4} />
                    <stop offset="100%" stopColor="var(--color-primary)" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="g2" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="var(--color-chart-2)" stopOpacity={0.35} />
                    <stop offset="100%" stopColor="var(--color-chart-2)" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                <XAxis dataKey="hour" fontSize={10} tickLine={false} axisLine={false} />
                <YAxis fontSize={10} tickLine={false} axisLine={false} />
                <Tooltip contentStyle={{ background: "var(--color-popover)", border: "1px solid var(--color-border)", borderRadius: 8, fontSize: 12 }} />
                <Area type="monotone" dataKey="p50" stroke="var(--color-primary)" fill="url(#g1)" strokeWidth={2} />
                <Area type="monotone" dataKey="p95" stroke="var(--color-chart-2)" fill="url(#g2)" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card className="border-border/60">
          <CardHeader className="pb-2 flex flex-row items-center justify-between">
            <CardTitle className="text-sm">Success Rate (14d)</CardTitle>
            <Badge className="bg-success/15 text-success hover:bg-success/20">Stable</Badge>
          </CardHeader>
          <CardContent className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={success} margin={{ left: -20, right: 8, top: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                <XAxis dataKey="day" fontSize={10} tickLine={false} axisLine={false} />
                <YAxis fontSize={10} tickLine={false} axisLine={false} domain={[94, 100]} />
                <Tooltip contentStyle={{ background: "var(--color-popover)", border: "1px solid var(--color-border)", borderRadius: 8, fontSize: 12 }} />
                <Line type="monotone" dataKey="rate" stroke="var(--color-success)" strokeWidth={2.5} dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card className="border-border/60">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Quality Score Distribution</CardTitle>
          </CardHeader>
          <CardContent className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={quality} margin={{ left: -20, right: 8, top: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                <XAxis dataKey="bin" fontSize={10} tickLine={false} axisLine={false} />
                <YAxis fontSize={10} tickLine={false} axisLine={false} />
                <Tooltip contentStyle={{ background: "var(--color-popover)", border: "1px solid var(--color-border)", borderRadius: 8, fontSize: 12 }} />
                <Bar dataKey="n" fill="var(--color-chart-3)" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </section>
    </AppShell>
  );
}
