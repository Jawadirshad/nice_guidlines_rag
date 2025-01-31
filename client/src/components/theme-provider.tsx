// components/theme-provider.tsx

import { createContext, useContext, useState, useEffect } from "react";

type Theme = "light" | "dark";

interface ThemeContextType {
  theme: Theme;
  setTheme: (theme: Theme) => void;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export const ThemeProvider: React.FC<{ defaultTheme?: Theme; storageKey?: string }> = ({
  defaultTheme = "light",
  storageKey = "theme",
  children,
}) => {
  const [theme, setThemeState] = useState<Theme>(defaultTheme);

  useEffect(() => {
    // Check localStorage for saved theme
    const savedTheme = localStorage.getItem(storageKey) as Theme | null;
    if (savedTheme) {
      setThemeState(savedTheme);
    } else {
      // Fallback to system preference if no saved theme
      const systemPreference = window.matchMedia("(prefers-color-scheme: dark)").matches
        ? "dark"
        : "light";
      setThemeState(systemPreference);
    }
  }, [storageKey]);

  useEffect(() => {
    // Apply the theme to document element
    document.documentElement.classList.toggle("dark", theme === "dark");
    localStorage.setItem(storageKey, theme); // Save the selected theme to localStorage
  }, [theme, storageKey]);

  const setTheme = (theme: Theme) => {
    setThemeState(theme);
  };

  return (
    <ThemeContext.Provider value={{ theme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};

export const useTheme = (): ThemeContextType => {
  const context = useContext(ThemeContext);
  if (!context) throw new Error("useTheme must be used within a ThemeProvider");
  return context;
};
