import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MathTutor — Personal AI Math Coach",
  description: "A private, step-by-step AI math tutor with persistent study notes and conversation history.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
