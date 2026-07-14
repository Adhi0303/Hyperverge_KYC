import type { ReactNode } from "react";
import { motion } from "framer-motion";
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar";
import { AppSidebar } from "./app-sidebar";
import { TopNav } from "./top-nav";

interface Props {
  children: ReactNode;
  title?: string;
  description?: string;
  actions?: ReactNode;
}

export function AppShell({ children, title, description, actions }: Props) {
  return (
    <SidebarProvider>
      <div className="flex min-h-screen w-full">
        <AppSidebar />
        <SidebarInset className="min-w-0">
          <TopNav />
          <main className="flex-1 px-4 py-6 md:px-8 md:py-8">
            {(title || actions) && (
              <div className="mx-auto mb-6 flex max-w-[1400px] flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
                <div className="min-w-0">
                  {title && (
                    <motion.h1
                      initial={{ opacity: 0, y: 6 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.3 }}
                      className="truncate text-2xl font-semibold tracking-tight md:text-[28px]"
                    >
                      {title}
                    </motion.h1>
                  )}
                  {description && (
                    <p className="mt-1 max-w-2xl text-sm text-muted-foreground">{description}</p>
                  )}
                </div>
                {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
              </div>
            )}
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.35, ease: "easeOut" }}
              className="mx-auto max-w-[1400px]"
            >
              {children}
            </motion.div>
          </main>
        </SidebarInset>
      </div>
    </SidebarProvider>
  );
}
