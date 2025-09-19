import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Shield, MessageCircle, AlertTriangle, Phone } from "lucide-react"
import Link from "next/link"

export default function HomePage() {
  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Shield className="h-8 w-8 text-primary" />
              <div>
                <h1 className="text-xl font-bold text-foreground">FIU-IND</h1>
                <p className="text-sm text-muted-foreground">Financial Intelligence Unit - India</p>
              </div>
            </div>
            <Select defaultValue="en">
              <SelectTrigger className="w-32">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="en">English</SelectItem>
                <SelectItem value="hi">हिंदी</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-12">
        <div className="max-w-4xl mx-auto text-center space-y-8">
          {/* Hero Section */}
          <div className="space-y-4">
            <h1 className="text-4xl md:text-5xl font-bold text-balance text-foreground">
              FIU-IND Financial Fraud Awareness Chatbot
            </h1>
            <p className="text-xl text-muted-foreground text-pretty max-w-2xl mx-auto">
              Ask questions about financial fraud, AML regulations, scam tactics, or report suspicious activity
              securely.
            </p>
          </div>

          {/* Features Grid */}
          <div className="grid md:grid-cols-3 gap-6 my-12">
            <Card className="text-center">
              <CardHeader>
                <MessageCircle className="h-12 w-12 text-primary mx-auto mb-2" />
                <CardTitle>Ask Questions</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Get instant answers about financial fraud prevention, AML regulations, and security best practices.
                </CardDescription>
              </CardContent>
            </Card>

            <Card className="text-center">
              <CardHeader>
                <AlertTriangle className="h-12 w-12 text-destructive mx-auto mb-2" />
                <CardTitle>Report Fraud</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Securely report suspicious financial activities and get guidance on next steps.
                </CardDescription>
              </CardContent>
            </Card>

            <Card className="text-center">
              <CardHeader>
                <Shield className="h-12 w-12 text-primary mx-auto mb-2" />
                <CardTitle>Stay Protected</CardTitle>
              </CardHeader>
              <CardContent>
                <CardDescription>
                  Learn about the latest scam tactics and how to protect yourself and your finances.
                </CardDescription>
              </CardContent>
            </Card>
          </div>

          {/* CTA Section */}
          <div className="space-y-6">
            <Link href="/chat">
              <Button size="lg" className="text-lg px-8 py-6">
                Start Chatting
                <MessageCircle className="ml-2 h-5 w-5" />
              </Button>
            </Link>

            {/* Warning Message */}
            <Card className="bg-muted/50 border-destructive/20">
              <CardContent className="pt-6">
                <div className="flex items-start gap-3">
                  <AlertTriangle className="h-5 w-5 text-destructive mt-0.5 flex-shrink-0" />
                  <div className="text-sm text-left">
                    <p className="font-medium text-destructive mb-1">Important Notice</p>
                    <p className="text-muted-foreground">
                      Please do not share personal or financial information in this chat. This chatbot provides
                      educational information only. For urgent matters, contact our helpline directly.
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-border bg-card/50 mt-16">
        <div className="container mx-auto px-4 py-8">
          <div className="grid md:grid-cols-3 gap-8">
            <div>
              <h3 className="font-semibold text-foreground mb-3">FIU-IND</h3>
              <p className="text-sm text-muted-foreground">
                Financial Intelligence Unit of India - Combating money laundering and terrorist financing.
              </p>
            </div>

            <div>
              <h3 className="font-semibold text-foreground mb-3">Quick Links</h3>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li>
                  <a href="#" className="hover:text-primary transition-colors">
                    About FIU-IND
                  </a>
                </li>
                <li>
                  <a href="#" className="hover:text-primary transition-colors">
                    AML Guidelines
                  </a>
                </li>
                <li>
                  <a href="#" className="hover:text-primary transition-colors">
                    Report Fraud
                  </a>
                </li>
                <li>
                  <a href="#" className="hover:text-primary transition-colors">
                    Resources
                  </a>
                </li>
              </ul>
            </div>

            <div>
              <h3 className="font-semibold text-foreground mb-3">Contact & Help</h3>
              <div className="space-y-2 text-sm text-muted-foreground">
                <div className="flex items-center gap-2">
                  <Phone className="h-4 w-4" />
                  <span>Helpline: 1800-XXX-XXXX</span>
                </div>
                <p>Email: help@fiuindia.gov.in</p>
                <p>Available 24/7 for urgent matters</p>
              </div>
            </div>
          </div>

          <div className="border-t border-border mt-8 pt-6 text-center text-sm text-muted-foreground">
            <p>&copy; 2024 Financial Intelligence Unit - India. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  )
}
