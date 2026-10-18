import { Button } from "./ui/button";
import { Badge } from "./ui/badge";
import { Github, Languages, Moon, Sun, TreeDeciduous } from "lucide-react";

type Theme = "light" | "dark";

interface HeaderProps {
  language: "en" | "fr";
  toggleLanguage: () => void;
  title: string;
  subtitle: string;
  languageToggle: string;
  theme: Theme;
  toggleTheme: () => void;
}

const REPO_URL = "https://github.com/TirthPatel6104/BA-growth-tool";

export function Header({
  language,
  toggleLanguage,
  title,
  subtitle,
  languageToggle,
  theme,
  toggleTheme,
}: HeaderProps) {
  const demoLabel = language === "fr" ? "Démo publique · données synthétiques" : "Public demo · synthetic data";

  return (
    <header className="bg-gradient-to-r from-green-800 to-emerald-700 text-white shadow-lg">
      <div className="container mx-auto max-w-7xl px-4 py-4 sm:py-6">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex min-w-0 items-center gap-3 sm:gap-4">
            <div className="shrink-0 rounded-lg bg-white/10 p-2.5 backdrop-blur-sm sm:p-3">
              <TreeDeciduous className="h-7 w-7 sm:h-8 sm:w-8" />
            </div>
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <h1 className="truncate text-lg font-semibold text-white sm:text-2xl">{title}</h1>
                <Badge
                  variant="outline"
                  className="border-white/40 bg-white/10 text-[11px] font-medium uppercase tracking-wide text-white"
                >
                  {demoLabel}
                </Badge>
              </div>
              <p className="mt-0.5 text-xs text-green-100 sm:text-sm">{subtitle}</p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <Button
              asChild
              variant="outline"
              size="sm"
              className="hidden border-white/30 bg-white/10 text-white hover:bg-white/20 hover:text-white sm:inline-flex"
            >
              <a href={REPO_URL} target="_blank" rel="noreferrer" aria-label="Source code on GitHub">
                <Github className="h-4 w-4" />
                <span className="ml-2">GitHub</span>
              </a>
            </Button>
            <Button
              onClick={toggleTheme}
              variant="outline"
              size="sm"
              className="border-white/30 bg-white/10 text-white hover:bg-white/20 hover:text-white"
              aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
              title={theme === "dark" ? "Light mode" : "Dark mode"}
            >
              {theme === "dark" ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
            </Button>
            <Button
              onClick={toggleLanguage}
              variant="outline"
              size="sm"
              className="border-white/30 bg-white/10 text-white hover:bg-white/20 hover:text-white"
            >
              <Languages className="h-4 w-4" />
              <span className="ml-2">{languageToggle}</span>
            </Button>
          </div>
        </div>
      </div>
    </header>
  );
}
