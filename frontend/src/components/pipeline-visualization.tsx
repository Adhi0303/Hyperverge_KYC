import { motion } from "framer-motion";
import { Check, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

export interface PipelineStep {
  key: string;
  label: string;
}

export const DEFAULT_STEPS: PipelineStep[] = [
  { key: "upload", label: "Upload" },
  { key: "preprocess", label: "Preprocessing" },
  { key: "segment", label: "Segmentation" },
  { key: "polygon", label: "Polygon Extraction" },
  { key: "perspective", label: "Perspective Correction" },
  { key: "quality", label: "Quality Assessment" },
  { key: "done", label: "Completed" },
];

interface Props {
  steps?: PipelineStep[];
  activeIndex: number; // -1 = idle, steps.length = done
}

export function PipelineVisualization({ steps = DEFAULT_STEPS, activeIndex }: Props) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      {steps.map((s, i) => {
        const state = i < activeIndex ? "done" : i === activeIndex ? "active" : "idle";
        return (
          <div key={s.key} className="flex items-center gap-2">
            <motion.div
              layout
              className={cn(
                "flex items-center gap-2 rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors",
                state === "done" && "border-success/40 bg-success/10 text-success",
                state === "active" && "border-primary/50 bg-primary/10 text-primary pipeline-glow",
                state === "idle" && "border-border/60 bg-card/40 text-muted-foreground",
              )}
            >
              <span
                className={cn(
                  "grid h-4 w-4 place-items-center rounded-full text-[10px] font-semibold",
                  state === "done" && "bg-success text-success-foreground",
                  state === "active" && "bg-primary text-primary-foreground",
                  state === "idle" && "bg-muted text-muted-foreground",
                )}
              >
                {state === "done" ? (
                  <Check className="h-2.5 w-2.5" />
                ) : state === "active" ? (
                  <Loader2 className="h-2.5 w-2.5 animate-spin" />
                ) : (
                  i + 1
                )}
              </span>
              {s.label}
            </motion.div>
            {i < steps.length - 1 && (
              <div className="relative h-px w-6 overflow-hidden bg-border">
                {i < activeIndex && (
                  <motion.div
                    initial={{ x: "-100%" }}
                    animate={{ x: 0 }}
                    transition={{ duration: 0.4 }}
                    className="absolute inset-0 bg-success"
                  />
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
