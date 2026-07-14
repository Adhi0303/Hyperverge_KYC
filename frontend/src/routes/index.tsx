import { createFileRoute, Link } from "@tanstack/react-router";
import { motion } from "framer-motion";
import {
  Activity,
  Clock,
  FileCheck2,
  Gauge,
  Upload,
  History as HistoryIcon,
  ExternalLink,
  
  CheckCircle2,
} from "lucide-react";
import { AppShell } from "@/components/app-shell";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { StatCard } from "@/components/stat-card";
import { PipelineVisualization } from "@/components/pipeline-visualization";

export const Route = createFileRoute("/")({
  component: Dashboard,
});

const recent = [
  { id: "doc_a91k", type: "Aadhaar Card", conf: 0.982, time: "412ms", ago: "2m ago", status: "success" },
  { id: "doc_b2xd", type: "PAN Card", conf: 0.968, time: "389ms", ago: "5m ago", status: "success" },
  { id: "doc_c7pl", type: "Passport", conf: 0.741, time: "512ms", ago: "12m ago", status: "review" },
  { id: "doc_d3mv", type: "Driver License", conf: 0.951, time: "398ms", ago: "18m ago", status: "success" },
  { id: "doc_e8kz", type: "Voter ID", conf: 0.612, time: "624ms", ago: "27m ago", status: "review" },
];

function Dashboard() {
  return (
    <AppShell>
      <section className="relative overflow-hidden rounded-2xl border border-border/60 bg-gradient-to-br from-card via-card to-card p-6 md:p-10">
        <div className="pointer-events-none absolute -right-32 -top-32 h-96 w-96 rounded-full bg-primary/15 blur-3xl" />
        <div className="pointer-events-none absolute -bottom-40 -left-20 h-80 w-80 rounded-full bg-chart-2/15 blur-3xl" />

        <div className="relative grid gap-8 md:grid-cols-[1.4fr_1fr] md:items-center">
          <div className="min-w-0">
            <Badge variant="outline" className="mb-4 gap-1.5 border-primary/30 bg-primary/5 text-primary">
              <span className="h-1.5 w-1.5 rounded-full bg-primary" />
              Enterprise Document Intelligence
            </Badge>
            <motion.h1
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-3xl font-semibold tracking-tight md:text-5xl"
            >
              HyperVision <span className="bg-gradient-to-r from-primary to-chart-2 bg-clip-text text-transparent">KYC AI</span>
            </motion.h1>
            <p className="mt-3 max-w-xl text-sm leading-relaxed text-muted-foreground md:text-base">
              AI powered document segmentation and preprocessing pipeline,
              tracked in MLflow and battle-tested across 12M+ identity documents.
            </p>

            <div className="mt-6 flex flex-wrap gap-2">
              <Button asChild size="lg" className="gap-2 shadow-lg shadow-primary/20">
                <Link to="/analysis">
                  <Upload className="h-4 w-4" />
                  Upload Document
                </Link>
              </Button>
              <Button asChild variant="outline" size="lg" className="gap-2">
                <Link to="/history">
                  <HistoryIcon className="h-4 w-4" />
                  View History
                </Link>
              </Button>
              <Button asChild variant="ghost" size="lg" className="gap-2">
                <Link to="/experiments">
                  <ExternalLink className="h-4 w-4" />
                  MLflow Dashboard
                </Link>
              </Button>
            </div>
          </div>

          <div className="rounded-xl border border-border/60 bg-background/40 p-5 backdrop-blur">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  Live Pipeline
                </p>
                <p className="mt-1 text-sm font-semibold">seg-unet-v2.4.1</p>
              </div>
              <Badge className="gap-1 bg-success/15 text-success hover:bg-success/20">
                <CheckCircle2 className="h-3 w-3" />
                Healthy
              </Badge>
            </div>
            <div className="mt-4">
              <PipelineVisualization activeIndex={7} />
            </div>
            <div className="mt-4 grid grid-cols-3 gap-2 text-center">
              <div className="rounded-lg bg-card/60 p-2">
                <p className="text-[10px] text-muted-foreground">Latency p50</p>
                <p className="text-sm font-semibold tabular-nums">412ms</p>
              </div>
              <div className="rounded-lg bg-card/60 p-2">
                <p className="text-[10px] text-muted-foreground">Throughput</p>
                <p className="text-sm font-semibold tabular-nums">142/s</p>
              </div>
              <div className="rounded-lg bg-card/60 p-2">
                <p className="text-[10px] text-muted-foreground">Error rate</p>
                <p className="text-sm font-semibold tabular-nums text-success">0.03%</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="mt-6 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard label="Total Predictions" value="1,284,912" delta={12.4} hint="last 30d" icon={FileCheck2} accent="primary" index={0} />
        <StatCard label="Avg Confidence" value="94.7%" delta={1.2} hint="vs last week" icon={Gauge} accent="success" index={1} />
        <StatCard label="Avg Processing Time" value="412ms" delta={-6.1} hint="p50 latency" icon={Clock} accent="info" index={2} />
        <StatCard label="Registered Models" value="14" hint="3 in production" icon={Activity} accent="warning" index={3} />
      </section>

      <section className="mt-6 grid gap-4">
        <Card className="border-border/60">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-3">
            <div>
              <CardTitle className="text-base">Recent Activity</CardTitle>
              <p className="mt-0.5 text-xs text-muted-foreground">Last 5 inference runs</p>
            </div>
            <Button asChild variant="ghost" size="sm" className="text-xs">
              <Link to="/history">View all</Link>
            </Button>
          </CardHeader>
          <CardContent className="p-0">
            <div className="divide-y divide-border/60">
              {recent.map((r) => (
                <div key={r.id} className="flex items-center gap-3 px-6 py-3 transition-colors hover:bg-accent/40">
                  <div className="grid h-9 w-9 shrink-0 place-items-center rounded-lg border border-border/60 bg-card">
                    <FileCheck2 className="h-4 w-4 text-muted-foreground" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <span className="truncate text-sm font-medium">{r.type}</span>
                      <span className="font-mono text-[10px] text-muted-foreground">{r.id}</span>
                    </div>
                    <p className="text-xs text-muted-foreground">{r.ago} · {r.time}</p>
                  </div>
                  <Badge
                    variant="outline"
                    className={
                      r.status === "success"
                        ? "border-success/30 bg-success/10 text-success"
                        : "border-warning/30 bg-warning/10 text-warning"
                    }
                  >
                    {(r.conf * 100).toFixed(1)}%
                  </Badge>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

      </section>
    </AppShell>
  );
}
