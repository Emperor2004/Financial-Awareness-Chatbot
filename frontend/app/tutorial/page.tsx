"use client";

import React from 'react';
import DemoVideo from '@/components/demo-video';
import { Button } from '@/components/ui/button';
import { Play, Clock, Star, BookOpen } from 'lucide-react';

export default function TutorialPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      {/* Hero Section */}
      <div className="bg-gradient-to-r from-blue-600 to-indigo-700 text-white py-16">
        <div className="container mx-auto px-4 text-center">
          <h1 className="text-5xl font-bold mb-6">FIU-Sahayak Demo Hub</h1>
          <p className="text-xl mb-8 max-w-3xl mx-auto">
            Watch the latest walkthrough video showcasing onboarding, live chat, and compliance reporting features of the Financial Awareness Chatbot.
          </p>
          <div className="flex flex-wrap justify-center gap-4 mb-8">
            <div className="flex items-center space-x-2 bg-white/20 px-4 py-2 rounded-full">
              <Play className="h-5 w-5" />
              <span>1 Featured Demo</span>
            </div>
            <div className="flex items-center space-x-2 bg-white/20 px-4 py-2 rounded-full">
              <Clock className="h-5 w-5" />
              <span>5 Minute Overview</span>
            </div>
          </div>
          <Button size="lg" className="bg-white text-blue-600 hover:bg-gray-100">
            Watch the Demo
          </Button>
        </div>
      </div>

      {/* Demo Section */}
      <div className="py-16 bg-white">
        <div className="container mx-auto px-4">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Video Tutorial Demo</h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Experience our interactive video tutorial system with working video player controls.
            </p>
          </div>
          
          <div className="max-w-4xl mx-auto mb-16">
            <DemoVideo
              title="FIU-Sahayak Walkthrough"
              description="Recorded product demo highlighting the onboarding flow, live chat experience, and key compliance reporting features."
              videoUrl="/videos/financial-demo.mp4"
              duration="5:01"
              category="Demo"
              thumbnail="/placeholder.jpg"
            />
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="py-16 bg-white">
        <div className="container mx-auto px-4">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold mb-4">Why Our Tutorials?</h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Our video tutorials are designed by financial compliance experts to help you master FIU-Sahayak effectively.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            <div className="text-center">
              <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Star className="h-8 w-8 text-blue-600" />
              </div>
              <h3 className="text-lg font-semibold mb-2">Expert Content</h3>
              <p className="text-muted-foreground">
                Created by financial compliance professionals with real-world experience
              </p>
            </div>

            <div className="text-center">
              <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Clock className="h-8 w-8 text-green-600" />
              </div>
              <h3 className="text-lg font-semibold mb-2">Time-Efficient</h3>
              <p className="text-muted-foreground">
                Concise, focused content that gets you up to speed quickly
              </p>
            </div>

            <div className="text-center">
              <div className="w-16 h-16 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Play className="h-8 w-8 text-purple-600" />
              </div>
              <h3 className="text-lg font-semibold mb-2">Interactive Learning</h3>
              <p className="text-muted-foreground">
                Hands-on demonstrations with real examples and scenarios
              </p>
            </div>

            <div className="text-center">
              <div className="w-16 h-16 bg-orange-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <BookOpen className="h-8 w-8 text-orange-600" />
              </div>
              <h3 className="text-lg font-semibold mb-2">Comprehensive Coverage</h3>
              <p className="text-muted-foreground">
                From basic usage to advanced compliance procedures
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div className="py-16 bg-gradient-to-r from-blue-600 to-indigo-700 text-white">
        <div className="container mx-auto px-4 text-center">
          <h2 className="text-3xl font-bold mb-4">Ready to Master FIU-Sahayak?</h2>
          <p className="text-xl mb-8 max-w-2xl mx-auto">
            Start with our comprehensive tutorial library and become proficient in using the Financial Awareness Chatbot.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Button size="lg" className="bg-white text-blue-600 hover:bg-gray-100">
              Browse All Tutorials
            </Button>
            <Button size="lg" variant="outline" className="border-white text-white hover:bg-white/10">
              Try the Chatbot
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
