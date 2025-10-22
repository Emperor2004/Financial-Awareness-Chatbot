"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet"
import { Send, AlertTriangle, Bot, User, Loader2, History, Download, Trash2 } from "lucide-react"
import { useAuth } from "@/components/auth-provider"

// Simple markdown-to-JSX converter for basic formatting
const formatMessage = (text: string) => {
  // Split by lines
  const lines = text.split('\n')
  const elements: React.ReactNode[] = []
  let inList = false
  let listItems: string[] = []

  const processLine = (line: string, idx: number) => {
    // Headers (###)
    if (line.startsWith('### ')) {
      if (inList) {
        elements.push(<ul key={`list-${idx}`} className="list-disc pl-5 my-2">{listItems.map((item, i) => <li key={i} className="mb-1">{processInline(item)}</li>)}</ul>)
        listItems = []
        inList = false
      }
      elements.push(<h3 key={idx} className="font-semibold text-base mt-3 mb-2">{line.replace('### ', '')}</h3>)
    }
    // Bullet points (- or *)
    else if (line.match(/^[\s]*[-*]\s/)) {
      const content = line.replace(/^[\s]*[-*]\s/, '')
      listItems.push(content)
      inList = true
    }
    // Numbered lists
    else if (line.match(/^[\s]*\d+\.\s/)) {
      if (inList && listItems.length > 0) {
        elements.push(<ul key={`list-${idx}`} className="list-disc pl-5 my-2">{listItems.map((item, i) => <li key={i} className="mb-1">{processInline(item)}</li>)}</ul>)
        listItems = []
      }
      const content = line.replace(/^[\s]*\d+\.\s/, '')
      listItems.push(content)
      inList = true
    }
    // Horizontal rule
    else if (line.trim() === '---') {
      if (inList) {
        elements.push(<ul key={`list-${idx}`} className="list-disc pl-5 my-2">{listItems.map((item, i) => <li key={i} className="mb-1">{processInline(item)}</li>)}</ul>)
        listItems = []
        inList = false
      }
      elements.push(<hr key={idx} className="my-3 border-border" />)
    }
    // Empty line
    else if (line.trim() === '') {
      if (inList && listItems.length > 0) {
        elements.push(<ul key={`list-${idx}`} className="list-disc pl-5 my-2">{listItems.map((item, i) => <li key={i} className="mb-1">{processInline(item)}</li>)}</ul>)
        listItems = []
        inList = false
      }
      elements.push(<br key={idx} />)
    }
    // Regular paragraph
    else {
      if (inList) {
        elements.push(<ul key={`list-${idx}`} className="list-disc pl-5 my-2">{listItems.map((item, i) => <li key={i} className="mb-1">{processInline(item)}</li>)}</ul>)
        listItems = []
        inList = false
      }
      elements.push(<p key={idx} className="mb-2">{processInline(line)}</p>)
    }
  }

  const processInline = (text: string): React.ReactNode => {
    const parts: React.ReactNode[] = []
    let remaining = text
    let key = 0

    while (remaining.length > 0) {
      // Bold (**text**)
      const boldMatch = remaining.match(/\*\*([^*]+)\*\*/)
      if (boldMatch && boldMatch.index !== undefined) {
        if (boldMatch.index > 0) {
          parts.push(remaining.substring(0, boldMatch.index))
        }
        parts.push(<strong key={key++} className="font-semibold">{boldMatch[1]}</strong>)
        remaining = remaining.substring(boldMatch.index + boldMatch[0].length)
      }
      // Italic (*text*)
      else if (remaining.match(/\*([^*]+)\*/)) {
        const italicMatch = remaining.match(/\*([^*]+)\*/)!
        if (italicMatch.index! > 0) {
          parts.push(remaining.substring(0, italicMatch.index!))
        }
        parts.push(<em key={key++} className="italic">{italicMatch[1]}</em>)
        remaining = remaining.substring(italicMatch.index! + italicMatch[0].length)
      }
      else {
        parts.push(remaining)
        break
      }
    }

    return parts.length > 0 ? parts : text
  }

  lines.forEach((line, idx) => processLine(line, idx))
  
  // Handle remaining list items
  if (inList && listItems.length > 0) {
    elements.push(<ul key="list-final" className="list-disc pl-5 my-2">{listItems.map((item, i) => <li key={i} className="mb-1">{processInline(item)}</li>)}</ul>)
  }

  return <div className="space-y-1">{elements}</div>
}

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
  const { user } = useAuth()
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      content:
        "**Welcome to FIU-Sahayak!** 🇮🇳\n\nI'm your official AI assistant for the Financial Intelligence Unit of India (FIU-IND). I can help you understand:\n\n- Prevention of Money Laundering Act (PMLA)\n- Anti-Money Laundering (AML) regulations\n- Suspicious Transaction Reporting (STR)\n- FIU-IND compliance requirements\n- Financial fraud prevention\n\n### Important Information:\n\n⚠️ **I provide information only, not legal or financial advice**\n\n🔒 **Never share personal information** like PAN, Aadhaar, account numbers, or passwords\n\n📚 **My responses are based on official FIU-IND documents and regulations**\n\n❓ **For complex legal matters or technical support**, please contact FIU-IND directly\n\nHow can I assist you today?",
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

    // Load user-specific chat history from localStorage
    if (user?.email) {
      const userHistoryKey = `fiu-chat-history-${user.email}`
      const savedHistory = localStorage.getItem(userHistoryKey)
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
      } else {
        // Clear history if no user-specific data exists
        setChatHistory([])
      }
    }
  }, [user?.email, currentSessionId])

  useEffect(() => {
    if (currentSessionId && messages.length > 1 && user?.email) {
      // Don't save initial welcome message only
      const currentSession: ChatSession = {
        id: currentSessionId,
        messages,
        createdAt: new Date(),
        lastUpdated: new Date(),
      }

      // Update history without causing infinite loop
      setChatHistory(prev => {
        const updatedHistory = prev.filter((session) => session.id !== currentSessionId)
        updatedHistory.unshift(currentSession)
        
        // Keep only last 10 sessions
        const limitedHistory = updatedHistory.slice(0, 10)
        
        // Save to user-specific localStorage
        const userHistoryKey = `fiu-chat-history-${user.email}`
        localStorage.setItem(userHistoryKey, JSON.stringify(limitedHistory))
        
        return limitedHistory
      })
    }
  }, [messages, currentSessionId, user?.email])

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

    const userInput = inputValue.trim()

    // Check for potential PII (basic patterns)
    const piiPatterns = [
      /\b[A-Z]{5}[0-9]{4}[A-Z]\b/g, // PAN pattern
      /\b\d{12}\b/g, // Aadhaar pattern (12 digits)
      /\b\d{10,16}\b/g, // Account/card numbers
    ]

    let hasPotentialPII = false
    for (const pattern of piiPatterns) {
      if (pattern.test(userInput)) {
        hasPotentialPII = true
        break
      }
    }

    // Warn user if PII detected
    if (hasPotentialPII) {
      const warningMessage: Message = {
        id: Date.now().toString(),
        content: "⚠️ **Security Warning**\n\nIt appears your message may contain personal information (PAN, Aadhaar, account number, etc.).\n\n**Please do not share:**\n- PAN Card numbers\n- Aadhaar numbers\n- Bank account details\n- Credit/debit card numbers\n- Passwords or PINs\n\nI provide general information only and do not need your personal details. Would you like to rephrase your question?",
        sender: "bot",
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, warningMessage])
      setInputValue("")
      return
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      content: userInput,
      sender: "user",
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInputValue("")
    setIsLoading(true)

    try {
      // Call the backend API
      const response = await fetch('http://localhost:5000/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMessage.content,
          session_id: currentSessionId
        })
      })

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`)
      }

      const data = await response.json()
      
      // Format response with sources at the end
      let responseContent = data.response
      
      // Add sources section if available - formatted professionally without relevance scores
      if (data.sources && data.sources.length > 0) {
        responseContent += '\n\n---\n\n### 📚 Reference Sources\n\n'
        responseContent += '*The information provided is based on the following official documents:*\n\n'
        
        data.sources.slice(0, 3).forEach((source: any, idx: number) => {
          // Clean up document name (remove .txt extension, replace underscores)
          const docName = source.document.replace('.txt', '').replace(/_/g, ' ')
          const docType = source.doc_type || 'Official Document'
          
          // Format: Just document name and type (no relevance %)
          responseContent += `**${idx + 1}. ${docName}**\n`
          responseContent += `   - Type: ${docType}\n\n`
        })
        
        responseContent += '*For authoritative information, please refer to the official FIU-IND website at www.fiuindia.gov.in*'
      }

      const botResponse: Message = {
        id: (Date.now() + 1).toString(),
        content: responseContent,
        sender: "bot",
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, botResponse])
    } catch (error) {
      console.error('Error calling API:', error)
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: "I apologize, but I'm having trouble connecting to the server. Please make sure the backend is running on http://localhost:5000. Error: " + (error as Error).message,
        sender: "bot",
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
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
          "**Welcome to FIU-Sahayak!** 🇮🇳\n\nI'm your official AI assistant for the Financial Intelligence Unit of India (FIU-IND). I can help you understand:\n\n- Prevention of Money Laundering Act (PMLA)\n- Anti-Money Laundering (AML) regulations\n- Suspicious Transaction Reporting (STR)\n- FIU-IND compliance requirements\n- Financial fraud prevention\n\n### Important Information:\n\n⚠️ **I provide information only, not legal or financial advice**\n\n🔒 **Never share personal information** like PAN, Aadhaar, account numbers, or passwords\n\n📚 **My responses are based on official FIU-IND documents and regulations**\n\n❓ **For complex legal matters or technical support**, please contact FIU-IND directly\n\nHow can I assist you today?",
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
    if (user?.email) {
      const userHistoryKey = `fiu-chat-history-${user.email}`
      localStorage.removeItem(userHistoryKey)
    }
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
                      <div className="text-sm leading-relaxed">
                        {message.sender === "bot" ? formatMessage(message.content) : <p>{message.content}</p>}
                      </div>
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
