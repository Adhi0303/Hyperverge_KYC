import { createFileRoute } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { AppShell } from "@/components/app-shell";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ChevronLeft, ChevronRight, Download, Eye, Search, Filter } from "lucide-react";

export const Route = createFileRoute("/history")({
  head: () => ({
    meta: [
      { title: "History — HyperVision KYC AI" },
      { name: "description", content: "Full inference history with search, filters and detailed metadata." },
    ],
  }),
  component: HistoryPage,
});

const TYPES = ["Aadhaar", "PAN", "Passport", "License", "Voter ID"];
const STATUSES = ["success", "review", "failed"] as const;

const rows = Array.from({ length: 38 }).map((_, i) => {
  const c = 0.6 + Math.random() * 0.4;
  return {
    id: `doc_${(Math.random().toString(36).slice(2, 7))}`,
    type: TYPES[i % TYPES.length],
    conf: c,
    time: Math.floor(280 + Math.random() * 500),
    quality: Math.floor(60 + Math.random() * 40),
    status: c > 0.9 ? "success" : c > 0.75 ? "review" : "failed",
    ts: new Date(Date.now() - i * 1000 * 60 * 37).toLocaleString(),
  };
});

function HistoryPage() {
  const [q, setQ] = useState("");
  const [status, setStatus] = useState("all");
  const [page, setPage] = useState(1);
  const perPage = 10;

  const filtered = useMemo(
    () =>
      rows.filter(
        (r) =>
          (status === "all" || r.status === status) &&
          (q === "" || r.id.includes(q.toLowerCase()) || r.type.toLowerCase().includes(q.toLowerCase())),
      ),
    [q, status],
  );
  const pages = Math.max(1, Math.ceil(filtered.length / perPage));
  const paged = filtered.slice((page - 1) * perPage, page * perPage);

  return (
    <AppShell
      title="History"
      description="Every inference the workspace has ever produced, searchable and exportable."
      actions={
        <Button variant="outline" size="sm" className="gap-2">
          <Download className="h-3.5 w-3.5" />
          Export CSV
        </Button>
      }
    >
      <Card className="border-border/60">
        <div className="flex flex-col gap-3 border-b border-border/60 p-4 sm:flex-row sm:items-center">
          <div className="relative flex-1 min-w-0">
            <Search className="pointer-events-none absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
            <Input
              placeholder="Search by ID or document type…"
              value={q}
              onChange={(e) => setQ(e.target.value)}
              className="pl-9"
            />
          </div>
          <Select value={status} onValueChange={setStatus}>
            <SelectTrigger className="sm:w-44">
              <Filter className="mr-2 h-3.5 w-3.5" />
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All statuses</SelectItem>
              {STATUSES.map((s) => (
                <SelectItem key={s} value={s} className="capitalize">
                  {s}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow className="hover:bg-transparent">
                <TableHead className="w-12">Preview</TableHead>
                <TableHead>Document</TableHead>
                <TableHead>Timestamp</TableHead>
                <TableHead className="text-right">Confidence</TableHead>
                <TableHead className="text-right">Time</TableHead>
                <TableHead className="text-right">Quality</TableHead>
                <TableHead>Status</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {paged.map((r) => (
                <TableRow key={r.id}>
                  <TableCell>
                    <div className="h-9 w-9 rounded-md bg-gradient-to-br from-primary/15 to-chart-2/15" />
                  </TableCell>
                  <TableCell>
                    <div className="font-medium">{r.type}</div>
                    <div className="font-mono text-[10px] text-muted-foreground">{r.id}</div>
                  </TableCell>
                  <TableCell className="text-xs text-muted-foreground">{r.ts}</TableCell>
                  <TableCell className="text-right font-mono tabular-nums">{(r.conf * 100).toFixed(1)}%</TableCell>
                  <TableCell className="text-right font-mono tabular-nums">{r.time}ms</TableCell>
                  <TableCell className="text-right font-mono tabular-nums">{r.quality}</TableCell>
                  <TableCell>
                    <Badge
                      variant="outline"
                      className={
                        r.status === "success"
                          ? "border-success/30 bg-success/10 text-success"
                          : r.status === "review"
                            ? "border-warning/30 bg-warning/10 text-warning"
                            : "border-destructive/30 bg-destructive/10 text-destructive"
                      }
                    >
                      {r.status}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-right">
                    <Button variant="ghost" size="icon" className="h-7 w-7">
                      <Eye className="h-3.5 w-3.5" />
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
              {paged.length === 0 && (
                <TableRow>
                  <TableCell colSpan={8} className="h-32 text-center text-sm text-muted-foreground">
                    No results match your filters.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </div>

        <div className="flex items-center justify-between border-t border-border/60 p-3 text-xs text-muted-foreground">
          <span>
            Showing {(page - 1) * perPage + 1}–{Math.min(page * perPage, filtered.length)} of {filtered.length}
          </span>
          <div className="flex items-center gap-1">
            <Button variant="ghost" size="icon" className="h-7 w-7" disabled={page === 1} onClick={() => setPage((p) => p - 1)}>
              <ChevronLeft className="h-3.5 w-3.5" />
            </Button>
            <span className="tabular-nums">
              {page} / {pages}
            </span>
            <Button variant="ghost" size="icon" className="h-7 w-7" disabled={page === pages} onClick={() => setPage((p) => p + 1)}>
              <ChevronRight className="h-3.5 w-3.5" />
            </Button>
          </div>
        </div>
      </Card>
    </AppShell>
  );
}
