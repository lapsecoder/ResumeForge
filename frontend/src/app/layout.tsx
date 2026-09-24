import type { Metadata } from "next";
import type { ReactNode } from "react";
import "./globals.css";

export const metadata: Metadata = {
  title: "ResumeForge",
  description:
    "Analyze your resume — upload a PDF, DOCX, or TXT resume, see the parsed structure, and keep your data private. Nothing is permanently stored.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}