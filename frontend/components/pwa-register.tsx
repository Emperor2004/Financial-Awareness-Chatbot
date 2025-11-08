"use client"

import { useEffect } from "react"

const SERVICE_WORKER_PATH = "/sw.js"

export function PWARegister() {
  useEffect(() => {
    if (typeof window === "undefined") return
    if (!("serviceWorker" in navigator)) return
    if (process.env.NODE_ENV !== "production") return

    const registerServiceWorker = async () => {
      try {
        await navigator.serviceWorker.register(SERVICE_WORKER_PATH, { scope: "/" })
      } catch (error) {
        if (process.env.NODE_ENV !== "production") {
          console.error("Service worker registration failed:", error)
        }
      }
    }

    registerServiceWorker()
  }, [])

  return null
}

