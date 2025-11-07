"use client";

import React, { useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Play, Pause, RotateCcw, Volume2, VolumeX, Maximize, Settings } from 'lucide-react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';

interface TutorialVideoProps {
  title: string;
  description: string;
  videoUrl: string;
  duration?: string;
  category?: string;
  thumbnail?: string;
}

export const TutorialVideo: React.FC<TutorialVideoProps> = ({
  title,
  description,
  videoUrl,
  duration: videoDuration = "5:30",
  category = "Tutorial",
  thumbnail
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isMuted, setIsMuted] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [hasError, setHasError] = useState(false);

  const togglePlay = () => {
    const video = videoRef.current;
    if (!video) return;

    if (video.paused || video.ended) {
      void video.play();
    } else {
      video.pause();
    }
  };

  const toggleMute = () => {
    const video = videoRef.current;
    if (!video) return;

    const shouldMute = !video.muted;
    video.muted = shouldMute;
    setIsMuted(shouldMute);
  };

  const resetVideo = () => {
    const video = videoRef.current;
    if (!video) return;

    video.pause();
    video.currentTime = 0;
    setCurrentTime(0);
    setIsPlaying(false);
  };

  const toggleFullscreen = () => {
    const video = videoRef.current;
    if (!video) return;

    if (document.fullscreenElement) {
      void document.exitFullscreen();
    } else {
      void video.requestFullscreen();
    }
  };

  const formatTime = (value: number) => {
    if (!Number.isFinite(value) || value < 0) {
      return '0:00';
    }

    const minutes = Math.floor(value / 60);
    const seconds = Math.floor(value % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  const handleVideoError = () => {
    setHasError(true);
  };

  const handleVideoLoad = () => {
    setHasError(false);
  };

  return (
    <Card className="w-full max-w-4xl mx-auto">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-2xl font-bold">{title}</CardTitle>
            <CardDescription className="text-lg mt-2">{description}</CardDescription>
          </div>
          <Badge variant="secondary" className="text-sm">
            {category}
          </Badge>
        </div>
      </CardHeader>
      <CardContent>
        <div className="relative bg-black rounded-lg overflow-hidden">
          {/* Video Player */}
          <div className="relative aspect-video">
            {hasError ? (
              <div className="w-full h-full flex items-center justify-center bg-gray-900 text-white">
                <div className="text-center">
                  <div className="text-4xl mb-4">🎥</div>
                  <h3 className="text-lg font-semibold mb-2">Video Unavailable</h3>
                  <p className="text-sm text-gray-300 mb-4">
                    This tutorial video is currently being prepared.
                  </p>
                  <div className="text-xs text-gray-400">
                    Duration: {videoDuration} | Category: {category}
                  </div>
                </div>
              </div>
            ) : (
              <video
                ref={videoRef}
                className="w-full h-full object-cover"
                poster={thumbnail}
                controls
                preload="metadata"
                onPlay={() => setIsPlaying(true)}
                onPause={() => setIsPlaying(false)}
                onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
                onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)}
                onError={handleVideoError}
                onLoadStart={handleVideoLoad}
                onVolumeChange={(e) => setIsMuted(e.currentTarget.muted)}
              >
                <source src={videoUrl} type="video/mp4" />
                Your browser does not support the video tag.
              </video>
            )}
            
            {/* Custom Controls Overlay */}
            {!hasError && (
              <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-4">
                <div className="flex items-center justify-between text-white">
                  <div className="flex items-center space-x-2">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={togglePlay}
                      className="text-white hover:bg-white/20"
                    >
                      {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={resetVideo}
                      className="text-white hover:bg-white/20"
                    >
                      <RotateCcw className="h-4 w-4" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={toggleMute}
                      className="text-white hover:bg-white/20"
                    >
                      {isMuted ? <VolumeX className="h-4 w-4" /> : <Volume2 className="h-4 w-4" />}
                    </Button>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-sm">
                      {formatTime(currentTime)} / {duration > 0 ? formatTime(duration) : '--:--'}
                    </span>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={toggleFullscreen}
                      className="text-white hover:bg-white/20"
                    >
                      <Maximize className="h-4 w-4" />
                    </Button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
        
        {/* Video Info */}
        <div className="mt-4 flex items-center justify-between text-sm text-muted-foreground">
          <span>Duration: {videoDuration}</span>
          <span>Category: {category}</span>
        </div>
      </CardContent>
    </Card>
  );
};

export default TutorialVideo;
