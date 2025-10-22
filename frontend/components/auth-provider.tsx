"use client"

import { createContext, useContext, useEffect, useState } from "react"
import { useRouter, usePathname } from "next/navigation"

interface User {
  email: string
  name: string
  loginTime: string
}

interface AuthContextType {
  user: User | null
  login: (email: string, password: string) => Promise<boolean>
  signup: (name: string, email: string, password: string) => Promise<boolean>
  logout: () => void
  isAuthenticated: boolean
  isLoading: boolean
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const router = useRouter()
  const pathname = usePathname()

  useEffect(() => {
    // Check for existing session on mount
    const session = localStorage.getItem("fiu-session")
    if (session) {
      try {
        const userData = JSON.parse(session)
        setUser(userData)
      } catch (error) {
        localStorage.removeItem("fiu-session")
      }
    }
    setIsLoading(false)
  }, [])

  const login = async (email: string, password: string): Promise<boolean> => {
    try {
      // In production, this would call your backend API
      const users = JSON.parse(localStorage.getItem("fiu-users") || "[]")
      const foundUser = users.find((u: any) => u.email === email && u.password === password)

      if (foundUser) {
        const userData: User = {
          email: foundUser.email,
          name: foundUser.name,
          loginTime: new Date().toISOString()
        }
        localStorage.setItem("fiu-session", JSON.stringify(userData))
        setUser(userData)
        return true
      }
      return false
    } catch (error) {
      console.error("Login error:", error)
      return false
    }
  }

  const signup = async (name: string, email: string, password: string): Promise<boolean> => {
    try {
      // In production, this would call your backend API
      const users = JSON.parse(localStorage.getItem("fiu-users") || "[]")
      
      if (users.some((u: any) => u.email === email)) {
        return false // User already exists
      }

      const newUser = {
        id: Date.now().toString(),
        name,
        email,
        password, // In production, hash on backend
        createdAt: new Date().toISOString()
      }

      users.push(newUser)
      localStorage.setItem("fiu-users", JSON.stringify(users))
      return true
    } catch (error) {
      console.error("Signup error:", error)
      return false
    }
  }

  const logout = () => {
    localStorage.removeItem("fiu-session")
    setUser(null)
    router.push("/login")
  }

  const value: AuthContextType = {
    user,
    login,
    signup,
    logout,
    isAuthenticated: !!user,
    isLoading
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error("useAuth must be used within an AuthProvider")
  }
  return context
}



