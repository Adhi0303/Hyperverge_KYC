import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

interface Props {
  label: string;
  value: number; // 0..100
  unit?: string;
  status?: "excellent" | "good" | "poor";
}

const statusColor = {
  excellent: "text-success",
  good: "text-warning",
  poor: "text-destructive",
} as const;

const statusStroke = {
  excellent: "stroke-success",
  good: "stroke-warning",
  poor: "stroke-destructive",
} as const;

export function QualityGauge({ label, value, unit = "%", status = "good" }: Props) {
  const clamped = Math.max(0, Math.min(100, value));
  const radius = 34;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (clamped / 100) * circumference;

  return (
    <div className="flex flex-col items-center gap-2 rounded-xl border border-border/60 bg-card/40 p-4">
      <div className="relative h-24 w-24">
        <svg viewBox="0 0 80 80" className="h-full w-full -rotate-90">
          <circle cx="40" cy="40" r={radius} className="fill-none stroke-muted" strokeWidth="6" />
          <motion.circle
            cx="40"
            cy="40"
            r={radius}
            className={cn("fill-none", statusStroke[status])}
            strokeWidth="6"
            strokeLinecap="round"
            strokeDasharray={circumference}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset: offset }}
            transition={{ duration: 0.9, ease: "easeOut" }}
          />
        </svg>
        <div className="absolute inset-0 grid place-items-center">
          <div className="text-center">
            <div className={cn("text-lg font-semibold tabular-nums", statusColor[status])}>
              {clamped.toFixed(0)}
              <span className="text-[10px] text-muted-foreground">{unit}</span>
            </div>
          </div>
        </div>
      </div>
      <div className="text-center">
        <p className="text-xs font-medium">{label}</p>
        <p className={cn("text-[10px] uppercase tracking-wide", statusColor[status])}>{status}</p>
      </div>
    </div>
  );
}
