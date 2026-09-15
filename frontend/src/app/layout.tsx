// ─── app/layout ─────────────────────────────────────────────────────────────
// Next.js App Router root layout. Wraps every page with:
//   • `<html lang="en">` and a dark `<body>`.
//   • The two Geist font families (loaded by Next from Google Fonts at
//     build time, then served self-hosted to keep things local-first).
//   • The TanStack Query client provider (see ./providers.tsx).
//
import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import Providers from "./providers";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Memoria",
  description: "AI-powered media management",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
