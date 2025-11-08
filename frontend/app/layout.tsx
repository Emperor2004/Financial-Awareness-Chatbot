import type React from "react"
import type { Metadata, Viewport } from "next"
import { GeistSans } from "geist/font/sans"
import { GeistMono } from "geist/font/mono"
import { Analytics } from "@vercel/analytics/next"
import { Suspense } from "react"
import { AuthProvider } from "@/components/auth-provider"
import { PWARegister } from "@/components/pwa-register"
import "./globals.css"

export const metadata: Metadata = {
  title: "FIU-Sahayak - Official FIU-IND Assistant",
  description:
    "Official FIU-IND AI assistant for financial regulations, PMLA compliance, AML guidance, and financial fraud awareness.",
  generator: "v0.app",
  manifest: "/manifest.json",
  icons: {
    icon: [
      { url: "/icon-192.png", sizes: "192x192", type: "image/png" },
      { url: "/icon-512.png", sizes: "512x512", type: "image/png" },
    ],
    shortcut: "/icon-192.png",
    apple: "/icon-192.png",
  },
}

export const viewport: Viewport = {
  themeColor: "#0f172a",
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body className={`font-sans ${GeistSans.variable} ${GeistMono.variable}`}>
        <AuthProvider>
          <PWARegister />
          <Suspense fallback={null}>{children}</Suspense>
        </AuthProvider>
        <Analytics />
      </body>
    </html>
  )
}
