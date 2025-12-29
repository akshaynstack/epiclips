"use client";

import React, { useRef, useEffect, useState, useCallback } from "react";
import { useEditorStore, Face, CropRegion } from "@/lib/editorStore";
import { Play, Pause, RotateCcw } from "lucide-react";

interface PreviewCanvasProps {
    className?: string;
}

export function PreviewCanvas({ className }: PreviewCanvasProps) {
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);

    const {
        videoUrl,
        faces,
        cropRegions,
        selectedFaceId,
        trackingEnabled,
        layoutPreset,
        currentTime,
        isPlaying,
        setCurrentTime,
        setPlaying,
        selectFace,
        updateCropRegion,
        getAspectRatio,
    } = useEditorStore();

    const [scale, setScale] = useState(1);
    const [dragging, setDragging] = useState<{ faceId: string; startX: number; startY: number; startRegion: CropRegion } | null>(null);
    const [resizing, setResizing] = useState<{ faceId: string; handle: string; startX: number; startY: number; startRegion: CropRegion } | null>(null);

    // Calculate scale to fit video in container
    useEffect(() => {
        if (!containerRef.current || !videoRef.current) return;

        const updateScale = () => {
            const container = containerRef.current!;
            const video = videoRef.current!;

            if (video.videoWidth === 0) return;

            const containerWidth = container.clientWidth;
            const containerHeight = container.clientHeight;

            const scaleX = containerWidth / video.videoWidth;
            const scaleY = containerHeight / video.videoHeight;

            setScale(Math.min(scaleX, scaleY, 1));
        };

        updateScale();
        window.addEventListener("resize", updateScale);
        return () => window.removeEventListener("resize", updateScale);
    }, [videoUrl]);

    // Draw overlay on canvas
    const drawOverlay = useCallback(() => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!video || !canvas) return;

        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        // Match canvas size to video display size
        canvas.width = video.videoWidth * scale;
        canvas.height = video.videoHeight * scale;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw semi-transparent overlay for non-cropped areas
        ctx.fillStyle = "rgba(0, 0, 0, 0.4)";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw crop regions
        cropRegions.forEach((region) => {
            const face = faces.find(f => f.id === region.faceId);
            const isSelected = selectedFaceId === region.faceId;

            // Clear the crop area (show original video)
            ctx.clearRect(
                region.x * scale,
                region.y * scale,
                region.width * scale,
                region.height * scale
            );

            // Draw border
            ctx.strokeStyle = isSelected ? "#a855f7" : "#ffffff";
            ctx.lineWidth = isSelected ? 3 : 2;
            ctx.setLineDash(isSelected ? [] : [5, 5]);
            ctx.strokeRect(
                region.x * scale,
                region.y * scale,
                region.width * scale,
                region.height * scale
            );
            ctx.setLineDash([]);

            // Draw resize handles if selected
            if (isSelected) {
                const handleSize = 10;
                ctx.fillStyle = "#a855f7";

                // Corner handles
                const corners = [
                    { x: region.x, y: region.y, cursor: "nw-resize" },
                    { x: region.x + region.width, y: region.y, cursor: "ne-resize" },
                    { x: region.x, y: region.y + region.height, cursor: "sw-resize" },
                    { x: region.x + region.width, y: region.y + region.height, cursor: "se-resize" },
                ];

                corners.forEach(corner => {
                    ctx.fillRect(
                        corner.x * scale - handleSize / 2,
                        corner.y * scale - handleSize / 2,
                        handleSize,
                        handleSize
                    );
                });
            }

            // Draw speaker label
            if (face) {
                ctx.fillStyle = isSelected ? "#a855f7" : "rgba(255, 255, 255, 0.9)";
                ctx.font = "bold 14px Inter, sans-serif";

                const labelY = (region.y + region.height) * scale + 20;
                const labelX = (region.x + region.width / 2) * scale;

                ctx.textAlign = "center";
                ctx.fillText(face.label, labelX, labelY);
            }
        });

        // Draw face detection boxes (if tracking enabled)
        if (trackingEnabled) {
            faces.forEach((face) => {
                ctx.strokeStyle = face.isActiveSpeaker ? "#22c55e" : "rgba(255, 255, 255, 0.5)";
                ctx.lineWidth = 1;
                ctx.setLineDash([3, 3]);
                ctx.strokeRect(
                    face.bbox.x * scale,
                    face.bbox.y * scale,
                    face.bbox.width * scale,
                    face.bbox.height * scale
                );
                ctx.setLineDash([]);
            });
        }
    }, [faces, cropRegions, selectedFaceId, trackingEnabled, scale]);

    // Redraw on state changes
    useEffect(() => {
        drawOverlay();
    }, [drawOverlay]);

    // Video time sync
    useEffect(() => {
        const video = videoRef.current;
        if (!video) return;

        const handleTimeUpdate = () => {
            setCurrentTime(video.currentTime);
            drawOverlay();
        };

        video.addEventListener("timeupdate", handleTimeUpdate);
        return () => video.removeEventListener("timeupdate", handleTimeUpdate);
    }, [setCurrentTime, drawOverlay]);

    // Play/pause control
    useEffect(() => {
        const video = videoRef.current;
        if (!video) return;

        if (isPlaying) {
            video.play();
        } else {
            video.pause();
        }
    }, [isPlaying]);

    // Mouse handlers for drag/resize
    const handleMouseDown = (e: React.MouseEvent) => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const rect = canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) / scale;
        const y = (e.clientY - rect.top) / scale;

        // Check if clicking on a resize handle
        if (selectedFaceId) {
            const region = cropRegions.find(r => r.faceId === selectedFaceId);
            if (region) {
                const handleSize = 15 / scale;
                const corners = [
                    { handle: "nw", x: region.x, y: region.y },
                    { handle: "ne", x: region.x + region.width, y: region.y },
                    { handle: "sw", x: region.x, y: region.y + region.height },
                    { handle: "se", x: region.x + region.width, y: region.y + region.height },
                ];

                for (const corner of corners) {
                    if (Math.abs(x - corner.x) < handleSize && Math.abs(y - corner.y) < handleSize) {
                        setResizing({
                            faceId: selectedFaceId,
                            handle: corner.handle,
                            startX: x,
                            startY: y,
                            startRegion: { ...region },
                        });
                        return;
                    }
                }
            }
        }

        // Check if clicking inside a crop region
        for (const region of cropRegions) {
            if (
                x >= region.x && x <= region.x + region.width &&
                y >= region.y && y <= region.y + region.height
            ) {
                selectFace(region.faceId);
                setDragging({
                    faceId: region.faceId,
                    startX: x,
                    startY: y,
                    startRegion: { ...region },
                });
                return;
            }
        }

        // Click outside - deselect
        selectFace(null);
    };

    const handleMouseMove = (e: React.MouseEvent) => {
        const canvas = canvasRef.current;
        const video = videoRef.current;
        if (!canvas || !video) return;

        const rect = canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) / scale;
        const y = (e.clientY - rect.top) / scale;

        if (dragging) {
            const dx = x - dragging.startX;
            const dy = y - dragging.startY;

            const newX = Math.max(0, Math.min(video.videoWidth - dragging.startRegion.width, dragging.startRegion.x + dx));
            const newY = Math.max(0, Math.min(video.videoHeight - dragging.startRegion.height, dragging.startRegion.y + dy));

            updateCropRegion(dragging.faceId, { x: newX, y: newY });
        }

        if (resizing) {
            const dx = x - resizing.startX;
            const dy = y - resizing.startY;
            const region = resizing.startRegion;

            let newRegion = { ...region };

            switch (resizing.handle) {
                case "se":
                    newRegion.width = Math.max(50, region.width + dx);
                    newRegion.height = Math.max(50, region.height + dy);
                    break;
                case "sw":
                    newRegion.x = Math.max(0, region.x + dx);
                    newRegion.width = Math.max(50, region.width - dx);
                    newRegion.height = Math.max(50, region.height + dy);
                    break;
                case "ne":
                    newRegion.y = Math.max(0, region.y + dy);
                    newRegion.width = Math.max(50, region.width + dx);
                    newRegion.height = Math.max(50, region.height - dy);
                    break;
                case "nw":
                    newRegion.x = Math.max(0, region.x + dx);
                    newRegion.y = Math.max(0, region.y + dy);
                    newRegion.width = Math.max(50, region.width - dx);
                    newRegion.height = Math.max(50, region.height - dy);
                    break;
            }

            updateCropRegion(resizing.faceId, newRegion);
        }
    };

    const handleMouseUp = () => {
        setDragging(null);
        setResizing(null);
    };

    const togglePlayback = () => {
        setPlaying(!isPlaying);
    };

    if (!videoUrl) {
        return (
            <div className={`flex items-center justify-center bg-zinc-900 rounded-lg ${className}`}>
                <p className="text-zinc-500">No video loaded</p>
            </div>
        );
    }

    return (
        <div className={`flex flex-col gap-4 ${className}`}>
            {/* Video container */}
            <div
                ref={containerRef}
                className="relative bg-black rounded-lg overflow-hidden flex items-center justify-center"
                style={{ minHeight: "400px" }}
            >
                <video
                    ref={videoRef}
                    src={videoUrl}
                    className="max-w-full max-h-full"
                    style={{ transform: `scale(${scale})`, transformOrigin: "center" }}
                    onLoadedMetadata={drawOverlay}
                />
                <canvas
                    ref={canvasRef}
                    className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 pointer-events-auto cursor-crosshair"
                    onMouseDown={handleMouseDown}
                    onMouseMove={handleMouseMove}
                    onMouseUp={handleMouseUp}
                    onMouseLeave={handleMouseUp}
                />
            </div>

            {/* Playback controls */}
            <div className="flex items-center justify-center gap-4">
                <button
                    onClick={togglePlayback}
                    className="p-2 rounded-full bg-zinc-800 hover:bg-zinc-700 transition-colors"
                >
                    {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
                </button>

                <div className="flex-1 max-w-md">
                    <input
                        type="range"
                        min={0}
                        max={videoRef.current?.duration || 100}
                        value={currentTime}
                        onChange={(e) => {
                            const time = parseFloat(e.target.value);
                            setCurrentTime(time);
                            if (videoRef.current) {
                                videoRef.current.currentTime = time;
                            }
                        }}
                        className="w-full h-2 bg-zinc-700 rounded-lg appearance-none cursor-pointer"
                    />
                </div>

                <span className="text-sm text-zinc-400 min-w-[80px] text-right">
                    {formatTime(currentTime)} / {formatTime(videoRef.current?.duration || 0)}
                </span>
            </div>
        </div>
    );
}

function formatTime(seconds: number): string {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, "0")}`;
}
