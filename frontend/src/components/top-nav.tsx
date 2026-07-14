import { Bell, Moon, Sun, Search } from "lucide-react";
import { SidebarTrigger } from "@/components/ui/sidebar";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Input } from "@/components/ui/input";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { useTheme } from "@/components/theme-provider";

export function TopNav() {
  const { theme, toggle } = useTheme();

  return (
    <header className="sticky top-0 z-30 flex h-14 shrink-0 items-center gap-2 border-b border-border/60 bg-background/70 px-3 backdrop-blur-xl md:px-5">
      <SidebarTrigger className="-ml-1" />
      <Separator orientation="vertical" className="mr-1 h-5" />

      <div className="hidden min-w-0 items-center gap-2 md:flex">
        <span className="truncate text-sm font-semibold">hyperverge-ml-workspace</span>
        <Badge variant="outline" className="gap-1 text-[10px] font-medium">
          <span className="h-1.5 w-1.5 rounded-full bg-success" />
          Production
        </Badge>
        <Badge variant="secondary" className="text-[10px] font-mono">
          seg-unet-v2.4.1
        </Badge>
      </div>

      <div className="relative ml-auto hidden w-72 md:block">
        <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
        <Input
          placeholder="Search runs, documents, models…"
          className="h-8 pl-8 text-xs"
        />
      </div>

      <div className="ml-auto flex items-center gap-1 md:ml-2">
        <Button variant="ghost" size="icon" className="h-8 w-8" onClick={toggle} aria-label="Toggle theme">
          {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </Button>

        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="icon" className="relative h-8 w-8">
              <Bell className="h-4 w-4" />
              <span className="absolute right-1.5 top-1.5 h-1.5 w-1.5 rounded-full bg-destructive" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-80">
            <DropdownMenuLabel>Notifications</DropdownMenuLabel>
            <DropdownMenuSeparator />
            {[
              { t: "Model registered", d: "seg-unet-v2.4.1 promoted to production", ts: "2m ago" },
              { t: "Training job complete", d: "Run #4821 · Dice 0.947", ts: "1h ago" },
              { t: "Quality drift detected", d: "Blur score below threshold", ts: "3h ago" },
            ].map((n, i) => (
              <DropdownMenuItem key={i} className="flex-col items-start gap-0.5 py-2">
                <div className="text-sm font-medium">{n.t}</div>
                <div className="text-xs text-muted-foreground">{n.d}</div>
                <div className="text-[10px] text-muted-foreground">{n.ts}</div>
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>

        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" className="h-8 gap-2 pl-1 pr-2">
              <Avatar className="h-6 w-6">
                <AvatarFallback className="bg-gradient-to-br from-primary to-chart-2 text-[10px] text-primary-foreground">
                  SA
                </AvatarFallback>
              </Avatar>
              <span className="hidden text-xs font-medium md:inline">suriyaadhi</span>
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuLabel>My Account</DropdownMenuLabel>
            <DropdownMenuSeparator />
            <DropdownMenuItem>Profile</DropdownMenuItem>
            <DropdownMenuItem>API Keys</DropdownMenuItem>
            <DropdownMenuItem>Billing</DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem>Sign out</DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </header>
  );
}
