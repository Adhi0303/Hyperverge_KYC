import { createFileRoute } from "@tanstack/react-router";
import { AppShell } from "@/components/app-shell";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { GitBranch, Layers, Play, ArrowRightLeft } from "lucide-react";

export const Route = createFileRoute("/experiments")({
  head: () => ({
    meta: [
      { title: "Experiments — HyperVision KYC AI" },
      { name: "description", content: "MLflow-style experiment tracking with runs, metrics and artifacts." },
    ],
  }),
  component: ExperimentsPage,
});

const runs = [
  { name: "seg-unet-v2.4.1", id: "run_a91k4c", status: "finished", dice: 0.947, iou: 0.912, loss: 0.041, epochs: 60, params: "lr=3e-4, bs=16" },
  { name: "seg-unet-v2.3.9", id: "run_x22p9m", status: "finished", dice: 0.938, iou: 0.901, loss: 0.048, epochs: 55, params: "lr=1e-4, bs=32" },
  { name: "seg-deeplab-v1.2", id: "run_c81jd2", status: "failed", dice: 0.881, iou: 0.842, loss: 0.092, epochs: 40, params: "lr=5e-4, bs=8" },
  { name: "seg-unet-aug", id: "run_kk29mm", status: "running", dice: 0.921, iou: 0.887, loss: 0.058, epochs: 32, params: "lr=3e-4, bs=16, aug=heavy" },
];

function ExperimentsPage() {
  return (
    <AppShell
      title="Experiments"
      description="Track runs, hyperparameters and artifacts across your MLflow tracking server."
      actions={
        <>
          <Button variant="outline" size="sm" className="gap-2">
            <ArrowRightLeft className="h-3.5 w-3.5" /> Compare
          </Button>
          <Button size="sm" className="gap-2">
            <Play className="h-3.5 w-3.5" /> New Run
          </Button>
        </>
      }
    >
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-2">
        {runs.map((r) => (
          <Card key={r.id} className="group relative overflow-hidden border-border/60 transition-shadow hover:shadow-lg">
            <div className="pointer-events-none absolute -right-16 -top-16 h-40 w-40 rounded-full bg-primary/10 opacity-0 blur-3xl transition-opacity group-hover:opacity-100" />
            <CardHeader className="pb-3">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <CardTitle className="truncate text-base">{r.name}</CardTitle>
                  <p className="mt-0.5 font-mono text-[10px] text-muted-foreground">{r.id}</p>
                </div>
                <Badge
                  className={
                    r.status === "finished"
                      ? "bg-success/15 text-success hover:bg-success/20"
                      : r.status === "running"
                        ? "bg-info/15 text-info hover:bg-info/20"
                        : "bg-destructive/15 text-destructive hover:bg-destructive/20"
                  }
                >
                  {r.status}
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-4 gap-2 text-center">
                {[
                  { k: "Dice", v: r.dice.toFixed(3) },
                  { k: "IoU", v: r.iou.toFixed(3) },
                  { k: "Loss", v: r.loss.toFixed(3) },
                  { k: "Epochs", v: r.epochs },
                ].map((m) => (
                  <div key={m.k} className="rounded-lg border border-border/60 bg-background/40 p-2">
                    <p className="text-[10px] uppercase tracking-wider text-muted-foreground">{m.k}</p>
                    <p className="mt-0.5 text-sm font-semibold tabular-nums">{m.v}</p>
                  </div>
                ))}
              </div>

              <div className="rounded-lg border border-border/60 bg-background/40 p-3 font-mono text-[11px] text-muted-foreground">
                <div className="flex items-center gap-1.5 text-foreground">
                  <GitBranch className="h-3 w-3" />
                  main · {r.params}
                </div>
              </div>

              <div className="flex flex-wrap gap-1.5">
                {["model.pt", "config.yaml", "metrics.json", "confusion.png"].map((a) => (
                  <Badge key={a} variant="outline" className="gap-1 font-mono text-[10px]">
                    <Layers className="h-2.5 w-2.5" />
                    {a}
                  </Badge>
                ))}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    </AppShell>
  );
}
