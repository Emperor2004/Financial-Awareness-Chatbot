"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet"
import { Send, AlertTriangle, Bot, User, Loader2, History, Download, Trash2 } from "lucide-react"

interface Message {
  id: string
  content: string
  sender: "user" | "bot"
  timestamp: Date
}

interface ChatSession {
  id: string
  messages: Message[]
  createdAt: Date
  lastUpdated: Date
}

export function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      content:
        "Hello! I'm the FIU-IND chatbot. I can help you with questions about financial fraud, AML regulations, scam tactics, or guide you through reporting suspicious activities. How can I assist you today?",
      sender: "bot",
      timestamp: new Date(),
    },
  ])
  const [inputValue, setInputValue] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [currentSessionId, setCurrentSessionId] = useState<string>("")
  const [chatHistory, setChatHistory] = useState<ChatSession[]>([])
  const scrollAreaRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    // Generate session ID if not exists
    if (!currentSessionId) {
      const sessionId = `session_${Date.now()}`
      setCurrentSessionId(sessionId)
    }

    // Load chat history from localStorage
    const savedHistory = localStorage.getItem("fiu-chat-history")
    if (savedHistory) {
      try {
        const parsedHistory = JSON.parse(savedHistory).map((session: any) => ({
          ...session,
          createdAt: new Date(session.createdAt),
          lastUpdated: new Date(session.lastUpdated),
          messages: session.messages.map((msg: any) => ({
            ...msg,
            timestamp: new Date(msg.timestamp),
          })),
        }))
        setChatHistory(parsedHistory)
      } catch (error) {
        console.error("Error loading chat history:", error)
      }
    }
  }, [currentSessionId])

  useEffect(() => {
    if (currentSessionId && messages.length > 1) {
      // Don't save initial welcome message only
      const currentSession: ChatSession = {
        id: currentSessionId,
        messages,
        createdAt: new Date(),
        lastUpdated: new Date(),
      }

      const updatedHistory = chatHistory.filter((session) => session.id !== currentSessionId)
      updatedHistory.unshift(currentSession)

      // Keep only last 10 sessions
      const limitedHistory = updatedHistory.slice(0, 10)

      setChatHistory(limitedHistory)
      localStorage.setItem("fiu-chat-history", JSON.stringify(limitedHistory))
    }
  }, [messages, currentSessionId, chatHistory])

  // Auto-scroll to bottom when new messages are added
  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollContainer = scrollAreaRef.current.querySelector("[data-radix-scroll-area-viewport]")
      if (scrollContainer) {
        scrollContainer.scrollTop = scrollContainer.scrollHeight
      }
    }
  }, [messages])

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputValue.trim(),
      sender: "user",
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInputValue("")
    setIsLoading(true)

    // Simulate bot response (will be replaced with actual AI integration)
    setTimeout(() => {
      const botResponse: Message = {
        id: (Date.now() + 1).toString(),
        content: getBotResponse(userMessage.content),
        sender: "bot",
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, botResponse])
      setIsLoading(false)
    }, 1500)
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const clearChat = () => {
    const newSessionId = `session_${Date.now()}`
    setCurrentSessionId(newSessionId)
    setMessages([
      {
        id: "1",
        content:
          "Hello! I'm the FIU-IND chatbot. I can help you with questions about financial fraud, AML regulations, scam tactics, or guide you through reporting suspicious activities. How can I assist you today?",
        sender: "bot",
        timestamp: new Date(),
      },
    ])
  }

  const loadChatSession = (session: ChatSession) => {
    setCurrentSessionId(session.id)
    setMessages(session.messages)
  }

  const clearAllHistory = () => {
    setChatHistory([])
    localStorage.removeItem("fiu-chat-history")
  }

  const exportChatHistory = () => {
    if (messages.length <= 1) return

    const chatText = messages
      .map((msg) => {
        const time = msg.timestamp.toLocaleString()
        const sender = msg.sender === "user" ? "You" : "FIU-IND Bot"
        return `[${time}] ${sender}: ${msg.content}`
      })
      .join("\n\n")

    const blob = new Blob([chatText], { type: "text/plain" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `FIU-IND-Chat-${new Date().toISOString().split("T")[0]}.txt`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  return (
    <div className="flex flex-col h-full max-w-4xl mx-auto w-full">
      {/* Warning Banner */}
      <Card className="m-4 bg-muted/50 border-destructive/20">
        <CardContent className="pt-4">
          <div className="flex items-start gap-3">
            <AlertTriangle className="h-5 w-5 text-destructive mt-0.5 flex-shrink-0" />
            <div className="text-sm">
              <p className="font-medium text-destructive mb-1">Security Notice</p>
              <p className="text-muted-foreground">
                Do not share personal or financial information. This chatbot provides educational information only. For
                urgent matters, contact our helpline at 1800-XXX-XXXX.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Chat Messages */}
      <div className="flex-1 px-4">
        <ScrollArea className="h-full" ref={scrollAreaRef}>
          <div className="space-y-4 pb-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex gap-3 ${message.sender === "user" ? "justify-end" : "justify-start"}`}
              >
                {message.sender === "bot" && (
                  <div className="flex-shrink-0">
                    <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center">
                      <Bot className="h-4 w-4 text-primary-foreground" />
                    </div>
                  </div>
                )}

                <div className={`max-w-[70%] ${message.sender === "user" ? "order-1" : ""}`}>
                  <Card className={`${message.sender === "user" ? "bg-primary text-primary-foreground" : "bg-card"}`}>
                    <CardContent className="p-3">
                      <p className="text-sm leading-relaxed">{message.content}</p>
                      <p
                        className={`text-xs mt-2 ${
                          message.sender === "user" ? "text-primary-foreground/70" : "text-muted-foreground"
                        }`}
                      >
                        {message.timestamp.toLocaleTimeString([], {
                          hour: "2-digit",
                          minute: "2-digit",
                        })}
                      </p>
                    </CardContent>
                  </Card>
                </div>

                {message.sender === "user" && (
                  <div className="flex-shrink-0 order-2">
                    <div className="w-8 h-8 rounded-full bg-secondary flex items-center justify-center">
                      <User className="h-4 w-4 text-secondary-foreground" />
                    </div>
                  </div>
                )}
              </div>
            ))}

            {/* Loading indicator */}
            {isLoading && (
              <div className="flex gap-3 justify-start">
                <div className="flex-shrink-0">
                  <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center">
                    <Bot className="h-4 w-4 text-primary-foreground" />
                  </div>
                </div>
                <Card className="bg-card">
                  <CardContent className="p-3">
                    <div className="flex items-center gap-2">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      <span className="text-sm text-muted-foreground">Typing...</span>
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}
          </div>
        </ScrollArea>
      </div>

      {/* Input Area */}
      <div className="border-t border-border bg-card/50 p-4">
        <div className="flex gap-2 max-w-4xl mx-auto">
          <div className="flex-1">
            <Input
              ref={inputRef}
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask about financial fraud, AML regulations, or report suspicious activity..."
              disabled={isLoading}
              className="min-h-[44px]"
            />
          </div>
          <Button onClick={handleSendMessage} disabled={!inputValue.trim() || isLoading} size="lg">
            <Send className="h-4 w-4" />
            <span className="sr-only">Send message</span>
          </Button>
          <Button onClick={clearChat} variant="outline" size="lg">
            Clear
          </Button>

          <Sheet>
            <SheetTrigger asChild>
              <Button variant="outline" size="lg">
                <History className="h-4 w-4" />
                <span className="sr-only">Chat History</span>
              </Button>
            </SheetTrigger>
            <SheetContent className="w-[400px] sm:w-[540px]">
              <SheetHeader>
                <SheetTitle>Chat History</SheetTitle>
                <SheetDescription>View and manage your previous conversations</SheetDescription>
              </SheetHeader>

              <div className="mt-6 space-y-4">
                {/* Current Session Export */}
                {messages.length > 1 && (
                  <div className="space-y-2">
                    <h4 className="text-sm font-medium">Current Session</h4>
                    <div className="flex gap-2">
                      <Button onClick={exportChatHistory} variant="outline" size="sm">
                        <Download className="h-4 w-4 mr-2" />
                        Export Current Chat
                      </Button>
                    </div>
                  </div>
                )}

                {/* Previous Sessions */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <h4 className="text-sm font-medium">Previous Sessions</h4>
                    {chatHistory.length > 0 && (
                      <Button onClick={clearAllHistory} variant="outline" size="sm">
                        <Trash2 className="h-4 w-4 mr-2" />
                        Clear All
                      </Button>
                    )}
                  </div>

                  <ScrollArea className="h-[400px]">
                    <div className="space-y-2">
                      {chatHistory.length === 0 ? (
                        <p className="text-sm text-muted-foreground">No previous sessions</p>
                      ) : (
                        chatHistory.map((session) => (
                          <Card
                            key={session.id}
                            className="cursor-pointer hover:bg-accent/50 transition-colors"
                            onClick={() => loadChatSession(session)}
                          >
                            <CardHeader className="p-3">
                              <div className="space-y-1">
                                <p className="text-sm font-medium">
                                  {session.lastUpdated.toLocaleDateString()} at{" "}
                                  {session.lastUpdated.toLocaleTimeString([], {
                                    hour: "2-digit",
                                    minute: "2-digit",
                                  })}
                                </p>
                                <p className="text-xs text-muted-foreground">{session.messages.length} messages</p>
                                {session.messages.length > 1 && (
                                  <p className="text-xs text-muted-foreground truncate">
                                    Last: {session.messages[session.messages.length - 1].content.substring(0, 50)}...
                                  </p>
                                )}
                              </div>
                            </CardHeader>
                          </Card>
                        ))
                      )}
                    </div>
                  </ScrollArea>
                </div>
              </div>
            </SheetContent>
          </Sheet>
        </div>
      </div>
    </div>
  )
}

// Temporary bot response function (will be replaced with AI integration)
function getBotResponse(userMessage: string): string {
  const message = userMessage.toLowerCase()

  if (message.includes("fraud") || message.includes("scam")) {
    return "Financial fraud is a serious concern. Common types include phishing emails, fake investment schemes, and identity theft. Always verify the authenticity of financial communications and never share personal banking details over unsecured channels. Would you like to know about specific types of fraud or how to report suspicious activity?"
  }

  if (message.includes("report") || message.includes("suspicious")) {
    return "To report suspicious financial activity, you can: 1) Contact our helpline at 1800-XXX-XXXX, 2) File a complaint through the official FIU-IND portal, or 3) Visit your nearest bank branch. Please provide as much detail as possible including dates, amounts, and any documentation you have."
  }

  if (message.includes("aml") || message.includes("money laundering")) {
    return "Anti-Money Laundering (AML) regulations require financial institutions to monitor and report suspicious transactions. Key requirements include customer due diligence, transaction monitoring, and reporting of cash transactions above specified thresholds. Are you looking for information about compliance requirements or reporting procedures?"
  }

  if (message.includes("hello") || message.includes("hi")) {
    return "Hello! I'm here to help you with financial fraud awareness and AML guidance. You can ask me about different types of scams, how to protect yourself, reporting procedures, or AML regulations. What would you like to know?"
  }

  return "Thank you for your question. I can help you with information about financial fraud prevention, AML regulations, scam awareness, and reporting procedures. Could you please provide more specific details about what you'd like to know? For urgent matters or detailed assistance, please contact our helpline at 1800-XXX-XXXX."
}
