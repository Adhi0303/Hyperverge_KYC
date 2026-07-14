import { motion } from "framer-motion";
import type { LucideIcon } from "lucide-react";
import { ArrowUpRight, ArrowDownRight } from "lucide-react";
import { Card } from "@/components/ui/card";
import { cn } from "@/lib/utils";

interface StatCardProps {
  label: string;
  value: string;
  hint?: string;
  delta?: number;
  icon?: LucideIcon;
  accent?: "primary" | "success" | "warning" | "info";
  index?: number;
}

const accentMap: Record<NonNullable<StatCardProps["accent"]>, string> = {
  primary: "from-primary/20 to-transparent text-primary",
  success: "from-success/20 to-transparent text-success",
  warning: "from-warning/20 to-transparent text-warning",
  info: "from-info/20 to-transparent text-info",
};

export function StatCard({ label, value, hint, delta, icon: Icon, accent = "primary", index = 0 }: StatCardProps) {
  const positive = (delta ?? 0) >= 0;
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35, delay: index * 0.05 }}
    >
      <Card className="group relative overflow-hidden border-border/60 p-5">
        <div
          className={cn(
            "pointer-events-none absolute -right-10 -top-10 h-32 w-32 rounded-full bg-gradient-to-br opacity-70 blur-2xl transition-opacity group-hover:opacity-100",
            accentMap[accent],
          )}
        />
        <div className="relative flex items-start justify-between">
          <div className="min-w-0">
            <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">{label}</p>
            <p className="mt-2 text-2xl font-semibold tracking-tight md:text-3xl">{value}</p>
            {(hint || delta !== undefined) && (
              <div className="mt-2 flex items-center gap-1.5 text-xs">
                {delta !== undefined && (
                  <span
                    className={cn(
                      "inline-flex items-center gap-0.5 rounded-md px-1.5 py-0.5 font-medium",
                      positive ? "bg-success/10 text-success" : "bg-destructive/10 text-destructive",
                    )}
                  >
                    {positive ? <ArrowUpRight className="h-3 w-3" /> : <ArrowDownRight className="h-3 w-3" />}
                    {Math.abs(delta).toFixed(1)}%
                  </span>
                )}
                {hint && <span className="text-muted-foreground">{hint}</span>}
              </div>
            )}
          </div>
          {Icon && (
            <div className={cn("grid h-9 w-9 place-items-center rounded-lg border border-border/60 bg-card/50", accentMap[accent].split(" ").pop())}>
              <Icon className="h-4 w-4" />
            </div>
          )}
        </div>
      </Card>
    </motion.div>
  );
}
