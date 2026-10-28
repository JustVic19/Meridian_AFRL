import type { ReactNode } from "react";

import { labTitle } from "../lib/site";
import "./globals.css";

export const metadata = {
  title: labTitle,
  description: "Research skeleton. Not investment advice.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
