"use client";

import React, { useRef, useEffect, useState, useCallback } from "react";
import { X, Play, Pause, Eye, EyeOff, Check, Plus, Clock, Trash2 } from "lucide-react";
import { LayoutKeyframe } from "@/lib/store";

interface Face {
    id: string;
    label: string;
    bbox: { x: number; y: number; width: number; height: number };
    color: string;
    isSelected: boolean;
}

interface VisualEditorProps {
    isOpen: boolean;
    onClose: () => void;
    videoUrl: string;
    onApply: (settings: LayoutSettings) => void;
}

interface LayoutSettings {
    layoutPreset: "9:16" | "1:1" | "16:9" | "4:3" | "fill";
    selectedSpeakers: string[];
    trackFaces: boolean;
    podcastMode: boolean;  // When true, auto-detect active speaker from multiple boxes
    customRegions: {
        id: string;
        label: string;
        x: number;
        y: number;
        width: number;
        height: number;
    }[];
    keyframes: LayoutKeyframe[];  // Timeline keyframes for multi-layout rendering
}

const LAYOUT_PRESETS = [
    { id: "9:16" as const, label: "9:16", desc: "TikTok/Reels" },
    { id: "1:1" as const, label: "1:1", desc: "Square" },
    { id: "16:9" as const, label: "16:9", desc: "YouTube" },
    { id: "4:3" as const, label: "4:3", desc: "Classic" },
    { id: "fill" as const, label: "Fill", desc: "Full frame" },
];

const SPEAKER_COLORS = ["#a855f7", "#22c55e", "#3b82f6", "#f59e0b", "#ef4444"];

export function VisualEditor({ isOpen, onClose, videoUrl, onApply }: VisualEditorProps) {
    const videoRef = useRef<HTMLVideoElement>(null);
    const containerRef = useRef<HTMLDivElement>(null);

    const [isPlaying, setIsPlaying] = useState(false);
    const [currentTime, setCurrentTime] = useState(0);
    const [duration, setDuration] = useState(0);
    const [layoutPreset, setLayoutPreset] = useState<"9:16" | "1:1" | "16:9" | "4:3" | "fill">("9:16");
    const [trackFaces, setTrackFaces] = useState(true);
    const [podcastMode, setPodcastMode] = useState(false);  // Auto-detect active speaker
    const [videoLoaded, setVideoLoaded] = useState(false);
    const [keyframes, setKeyframes] = useState<LayoutKeyframe[]>([]);  // Timeline keyframes
    const [selectedKeyframeIndex, setSelectedKeyframeIndex] = useState<number | null>(null);

    // Video dimensions for scaling
    const [videoSize, setVideoSize] = useState({ width: 0, height: 0 });
    const [displaySize, setDisplaySize] = useState({ width: 0, height: 0, offsetX: 0, offsetY: 0 });

    // Demo faces - positioned to cover typical podcast/talking head video (% of video dimensions)
    const [faces, setFaces] = useState<Face[]>([
        { id: "face_1", label: "Speaker 1", bbox: { x: 60, y: 25, width: 25, height: 45 }, color: SPEAKER_COLORS[0], isSelected: true },
        { id: "face_2", label: "Speaker 2", bbox: { x: 10, y: 30, width: 25, height: 45 }, color: SPEAKER_COLORS[1], isSelected: true },
    ]);

    const [dragging, setDragging] = useState<string | null>(null);
    const [dragStart, setDragStart] = useState<{ x: number; y: number } | null>(null);
    const [resizeMode, setResizeMode] = useState<"none" | "tl" | "tr" | "bl" | "br">("none");
    const [hoveredHandle, setHoveredHandle] = useState<"none" | "tl" | "tr" | "bl" | "br" | "box">("none");

    const HANDLE_SIZE = 3; // Percentage of video width for handle hit area

    // Calculate display size when video loads
    useEffect(() => {
        const video = videoRef.current;
        const container = containerRef.current;
        if (!video || !container || !videoLoaded) return;

        const updateDisplaySize = () => {
            const containerRect = container.getBoundingClientRect();
            const videoAspect = video.videoWidth / video.videoHeight;
            const containerAspect = containerRect.width / containerRect.height;

            let displayWidth, displayHeight, offsetX, offsetY;

            if (videoAspect > containerAspect) {
                // Video is wider - letterbox top/bottom
                displayWidth = containerRect.width;
                displayHeight = containerRect.width / videoAspect;
                offsetX = 0;
                offsetY = (containerRect.height - displayHeight) / 2;
            } else {
                // Video is taller - letterbox left/right
                displayHeight = containerRect.height;
                displayWidth = containerRect.height * videoAspect;
                offsetX = (containerRect.width - displayWidth) / 2;
                offsetY = 0;
            }

            setVideoSize({ width: video.videoWidth, height: video.videoHeight });
            setDisplaySize({ width: displayWidth, height: displayHeight, offsetX, offsetY });
        };

        updateDisplaySize();
        window.addEventListener("resize", updateDisplaySize);
        return () => window.removeEventListener("resize", updateDisplaySize);
    }, [videoLoaded]);

    // Video time updates
    useEffect(() => {
        const video = videoRef.current;
        if (!video) return;

        const handleTimeUpdate = () => setCurrentTime(video.currentTime);
        const handleLoadedMetadata = () => {
            setDuration(video.duration);
            setVideoLoaded(true);
        };
        const handleCanPlay = () => {
            setVideoLoaded(true);
        };

        // Check if video is already loaded
        if (video.readyState >= 2) {
            setDuration(video.duration);
            setVideoLoaded(true);
        }

        video.addEventListener("timeupdate", handleTimeUpdate);
        video.addEventListener("loadedmetadata", handleLoadedMetadata);
        video.addEventListener("canplay", handleCanPlay);

        return () => {
            video.removeEventListener("timeupdate", handleTimeUpdate);
            video.removeEventListener("loadedmetadata", handleLoadedMetadata);
            video.removeEventListener("canplay", handleCanPlay);
        };
    }, [isOpen]);

    // Convert percentage coords to pixel position
    const toPixelCoords = (face: Face) => {
        return {
            x: displaySize.offsetX + (face.bbox.x / 100) * displaySize.width,
            y: displaySize.offsetY + (face.bbox.y / 100) * displaySize.height,
            width: (face.bbox.width / 100) * displaySize.width,
            height: (face.bbox.height / 100) * displaySize.height,
        };
    };

    // Convert pixel to percentage coords
    const toPercentCoords = (x: number, y: number) => {
        return {
            x: ((x - displaySize.offsetX) / displaySize.width) * 100,
            y: ((y - displaySize.offsetY) / displaySize.height) * 100,
        };
    };

    // Play/pause
    const togglePlayback = () => {
        const video = videoRef.current;
        if (!video) return;
        if (isPlaying) video.pause();
        else video.play();
        setIsPlaying(!isPlaying);
    };

    // Toggle speaker selection
    const toggleSpeaker = (faceId: string) => {
        setFaces(prev => prev.map(f =>
            f.id === faceId ? { ...f, isSelected: !f.isSelected } : f
        ));
    };

    // Update speaker label
    const updateLabel = (faceId: string, label: string) => {
        setFaces(prev => prev.map(f =>
            f.id === faceId ? { ...f, label } : f
        ));
    };

    // Get container-relative coordinates from mouse event
    const getRelativeCoords = (e: React.MouseEvent<HTMLDivElement>) => {
        const container = containerRef.current;
        if (!container) return null;
        const rect = container.getBoundingClientRect();
        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top,
        };
    };

    // Check if point is on a resize handle
    const getHandleAtPoint = (px: number, py: number): { faceId: string; handle: "tl" | "tr" | "bl" | "br" } | null => {
        const handlePx = (HANDLE_SIZE / 100) * displaySize.width;
        for (const face of faces) {
            if (!face.isSelected) continue;
            const coords = toPixelCoords(face);

            if (Math.abs(px - coords.x) < handlePx && Math.abs(py - coords.y) < handlePx)
                return { faceId: face.id, handle: "tl" };
            if (Math.abs(px - (coords.x + coords.width)) < handlePx && Math.abs(py - coords.y) < handlePx)
                return { faceId: face.id, handle: "tr" };
            if (Math.abs(px - coords.x) < handlePx && Math.abs(py - (coords.y + coords.height)) < handlePx)
                return { faceId: face.id, handle: "bl" };
            if (Math.abs(px - (coords.x + coords.width)) < handlePx && Math.abs(py - (coords.y + coords.height)) < handlePx)
                return { faceId: face.id, handle: "br" };
        }
        return null;
    };

    // Check if point is inside a face box
    const getFaceAtPoint = (px: number, py: number) => {
        for (const face of faces) {
            if (!face.isSelected) continue;
            const coords = toPixelCoords(face);
            if (px >= coords.x && px <= coords.x + coords.width &&
                py >= coords.y && py <= coords.y + coords.height) {
                return face.id;
            }
        }
        return null;
    };

    // Mouse handlers
    const handleMouseDown = (e: React.MouseEvent<HTMLDivElement>) => {
        const coords = getRelativeCoords(e);
        if (!coords) return;

        const handle = getHandleAtPoint(coords.x, coords.y);
        if (handle) {
            setDragging(handle.faceId);
            setResizeMode(handle.handle);
            setDragStart(coords);
            return;
        }

        const faceId = getFaceAtPoint(coords.x, coords.y);
        if (faceId) {
            const face = faces.find(f => f.id === faceId);
            if (face) {
                const pixelCoords = toPixelCoords(face);
                setDragging(faceId);
                setResizeMode("none");
                setDragStart({
                    x: coords.x - pixelCoords.x,
                    y: coords.y - pixelCoords.y
                });
            }
        }
    };

    const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
        const coords = getRelativeCoords(e);
        if (!coords) return;

        // Update hover cursor when not dragging
        if (!dragging) {
            const handle = getHandleAtPoint(coords.x, coords.y);
            if (handle) setHoveredHandle(handle.handle);
            else if (getFaceAtPoint(coords.x, coords.y)) setHoveredHandle("box");
            else setHoveredHandle("none");
            return;
        }

        // Dragging logic
        if (!dragStart) return;

        const MIN_SIZE = 5; // Minimum 5% of video

        setFaces(prev => prev.map(f => {
            if (f.id !== dragging) return f;

            if (resizeMode !== "none") {
                // Resize mode - convert to percentage
                const pct = toPercentCoords(coords.x, coords.y);
                let newX = f.bbox.x, newY = f.bbox.y;
                let newW = f.bbox.width, newH = f.bbox.height;

                if (resizeMode === "br") {
                    newW = Math.max(MIN_SIZE, pct.x - f.bbox.x);
                    newH = Math.max(MIN_SIZE, pct.y - f.bbox.y);
                } else if (resizeMode === "bl") {
                    const right = f.bbox.x + f.bbox.width;
                    newX = Math.min(pct.x, right - MIN_SIZE);
                    newW = right - newX;
                    newH = Math.max(MIN_SIZE, pct.y - f.bbox.y);
                } else if (resizeMode === "tr") {
                    const bottom = f.bbox.y + f.bbox.height;
                    newY = Math.min(pct.y, bottom - MIN_SIZE);
                    newH = bottom - newY;
                    newW = Math.max(MIN_SIZE, pct.x - f.bbox.x);
                } else if (resizeMode === "tl") {
                    const right = f.bbox.x + f.bbox.width;
                    const bottom = f.bbox.y + f.bbox.height;
                    newX = Math.min(pct.x, right - MIN_SIZE);
                    newY = Math.min(pct.y, bottom - MIN_SIZE);
                    newW = right - newX;
                    newH = bottom - newY;
                }

                // Clamp to 0-100
                newX = Math.max(0, Math.min(newX, 100 - MIN_SIZE));
                newY = Math.max(0, Math.min(newY, 100 - MIN_SIZE));
                newW = Math.min(newW, 100 - newX);
                newH = Math.min(newH, 100 - newY);

                return { ...f, bbox: { x: newX, y: newY, width: newW, height: newH } };
            } else {
                // Drag mode
                const pct = toPercentCoords(
                    coords.x - dragStart.x + displaySize.offsetX,
                    coords.y - dragStart.y + displaySize.offsetY
                );
                const newX = Math.max(0, Math.min(pct.x, 100 - f.bbox.width));
                const newY = Math.max(0, Math.min(pct.y, 100 - f.bbox.height));
                return { ...f, bbox: { ...f.bbox, x: newX, y: newY } };
            }
        }));
    };

    const handleMouseUp = () => {
        setDragging(null);
        setDragStart(null);
        setResizeMode("none");
    };

    // Handle apply
    const handleApply = () => {
        const selectedFaces = faces.filter(f => f.isSelected);
        onApply({
            layoutPreset,
            selectedSpeakers: selectedFaces.map(f => f.id),
            trackFaces,
            podcastMode,  // Auto-detect active speaker for podcasts
            customRegions: selectedFaces.map(f => ({
                id: f.id,
                label: f.label,
                x: f.bbox.x,
                y: f.bbox.y,
                width: f.bbox.width,
                height: f.bbox.height,
            })),
            keyframes,  // Pass timeline keyframes
        });

        onClose();
    };

    // Add keyframe at current video time
    const addKeyframe = () => {
        const timestampMs = Math.floor(currentTime * 1000);
        const selectedFaces = faces.filter(f => f.isSelected);

        // Create keyframe with current layout config
        const newKeyframe: LayoutKeyframe = {
            timestampMs,
            layoutType: selectedFaces.length >= 2 ? 'podcast' : 'split_screen',
            faceBbox: selectedFaces.length === 1 ? {
                x: selectedFaces[0].bbox.x / 100,
                y: selectedFaces[0].bbox.y / 100,
                width: selectedFaces[0].bbox.width / 100,
                height: selectedFaces[0].bbox.height / 100,
            } : undefined,
            podcastBboxes: selectedFaces.length >= 2 ? selectedFaces.slice(0, 2).map(f => ({
                x: f.bbox.x / 100,
                y: f.bbox.y / 100,
                width: f.bbox.width / 100,
                height: f.bbox.height / 100,
            })) : undefined,
        };

        setKeyframes(prev => [...prev, newKeyframe].sort((a, b) => a.timestampMs - b.timestampMs));
        setSelectedKeyframeIndex(keyframes.length);  // Select the new keyframe
    };

    // Remove keyframe
    const removeKeyframe = (index: number) => {
        setKeyframes(prev => prev.filter((_, i) => i !== index));
        setSelectedKeyframeIndex(null);
    };

    // Format time for keyframe display
    const formatKeyframeTime = (ms: number) => {
        const secs = Math.floor(ms / 1000);
        const mins = Math.floor(secs / 60);
        const remainingSecs = secs % 60;
        return `${mins}:${remainingSecs.toString().padStart(2, "0")}`;
    };

    const formatTime = (seconds: number) => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, "0")}`;
    };

    const getCursor = () => {
        if (dragging) return "grabbing";
        if (hoveredHandle === "tl") return "nw-resize";
        if (hoveredHandle === "tr") return "ne-resize";
        if (hoveredHandle === "bl") return "sw-resize";
        if (hoveredHandle === "br") return "se-resize";
        if (hoveredHandle === "box") return "grab";
        return "default";
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 bg-black/95 flex flex-col">
            {/* Header */}
            <div className="flex items-center justify-between px-6 py-4 border-b border-white/10 bg-black/50">
                <h2 className="text-xl font-bold text-white">Layout Editor</h2>
                <button onClick={onClose} className="p-2 hover:bg-white/10 rounded-lg transition-colors">
                    <X className="w-5 h-5" />
                </button>
            </div>

            {/* Main content */}
            <div className="flex flex-1 overflow-hidden">
                {/* Left: Video Preview */}
                <div className="flex-1 p-6 flex flex-col min-w-0">
                    <div
                        ref={containerRef}
                        className="relative flex-1 bg-black rounded-2xl overflow-hidden"
                        style={{ cursor: getCursor() }}
                        onMouseDown={handleMouseDown}
                        onMouseMove={handleMouseMove}
                        onMouseUp={handleMouseUp}
                        onMouseLeave={handleMouseUp}
                    >
                        <video
                            ref={videoRef}
                            src={videoUrl}
                            className="w-full h-full object-contain"
                        />

                        {/* Face overlays - positioned absolutely based on video display area */}
                        {videoLoaded && faces.filter(f => f.isSelected).map((face) => {
                            const coords = toPixelCoords(face);
                            return (
                                <div
                                    key={face.id}
                                    className="absolute pointer-events-none"
                                    style={{
                                        left: coords.x,
                                        top: coords.y,
                                        width: coords.width,
                                        height: coords.height,
                                        border: `3px solid ${face.color}`,
                                        backgroundColor: `${face.color}20`,
                                    }}
                                >
                                    {/* Label */}
                                    <div
                                        className="absolute -top-6 left-0 px-2 py-0.5 text-xs font-bold text-white rounded"
                                        style={{ backgroundColor: face.color }}
                                    >
                                        {face.label}
                                    </div>
                                    {/* Corner handles */}
                                    <div className="absolute -top-1.5 -left-1.5 w-3 h-3 rounded-sm" style={{ backgroundColor: face.color }} />
                                    <div className="absolute -top-1.5 -right-1.5 w-3 h-3 rounded-sm" style={{ backgroundColor: face.color }} />
                                    <div className="absolute -bottom-1.5 -left-1.5 w-3 h-3 rounded-sm" style={{ backgroundColor: face.color }} />
                                    <div className="absolute -bottom-1.5 -right-1.5 w-3 h-3 rounded-sm" style={{ backgroundColor: face.color }} />
                                </div>
                            );
                        })}
                    </div>

                    {/* Playback controls */}
                    <div className="flex items-center gap-4 mt-4">
                        <button
                            onClick={togglePlayback}
                            className="p-3 bg-white/10 rounded-xl hover:bg-white/20 transition-colors"
                        >
                            {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
                        </button>

                        <input
                            type="range"
                            min={0}
                            max={duration || 100}
                            value={currentTime}
                            onChange={(e) => {
                                const time = parseFloat(e.target.value);
                                if (videoRef.current) videoRef.current.currentTime = time;
                            }}
                            className="flex-1 h-1.5 bg-white/20 rounded-lg appearance-none cursor-pointer accent-purple-500"
                        />

                        <span className="text-sm text-white/60 min-w-[80px] text-right">
                            {formatTime(currentTime)} / {formatTime(duration)}
                        </span>
                    </div>
                </div>

                {/* Right: Settings Panel */}
                <div className="w-72 border-l border-white/10 p-5 overflow-y-auto bg-[#111]">
                    {/* Layout Presets */}
                    <div className="mb-6">
                        <h3 className="text-xs font-bold text-white/50 uppercase tracking-wider mb-3">
                            Aspect Ratio
                        </h3>
                        <div className="grid grid-cols-3 gap-2">
                            {LAYOUT_PRESETS.map((preset) => (
                                <button
                                    key={preset.id}
                                    onClick={() => setLayoutPreset(preset.id)}
                                    className={`p-2.5 rounded-lg border text-center transition-all ${layoutPreset === preset.id
                                        ? "bg-purple-600/30 border-purple-500 text-white"
                                        : "bg-white/5 border-white/10 text-white/70 hover:bg-white/10"
                                        }`}
                                >
                                    <div className="font-bold text-sm">{preset.label}</div>
                                    <div className="text-[9px] text-white/40">{preset.desc}</div>
                                </button>
                            ))}
                        </div>
                    </div>

                    {/* Speakers */}
                    <div className="mb-6">
                        <h3 className="text-xs font-bold text-white/50 uppercase tracking-wider mb-3">
                            Speakers
                        </h3>
                        <div className="space-y-2">
                            {faces.map((face) => (
                                <div
                                    key={face.id}
                                    onClick={() => toggleSpeaker(face.id)}
                                    className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-all ${face.isSelected
                                        ? "bg-white/10 border-white/20"
                                        : "bg-white/5 border-white/10 opacity-60"
                                        }`}
                                >
                                    <div
                                        className="w-3 h-3 rounded-full flex-shrink-0"
                                        style={{ backgroundColor: face.color }}
                                    />
                                    <input
                                        type="text"
                                        value={face.label}
                                        onChange={(e) => updateLabel(face.id, e.target.value)}
                                        onClick={(e) => e.stopPropagation()}
                                        className="flex-1 bg-transparent border-none outline-none text-sm text-white"
                                    />
                                    <div className={`w-5 h-5 rounded border-2 flex items-center justify-center transition-all ${face.isSelected
                                        ? "bg-purple-600 border-purple-600"
                                        : "border-white/30 bg-transparent"
                                        }`}>
                                        {face.isSelected && <Check className="w-3 h-3 text-white" />}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>

                    {/* Face Tracking */}
                    <div className="mb-6">
                        <div
                            onClick={() => setTrackFaces(!trackFaces)}
                            className="flex items-center justify-between p-4 bg-white/5 rounded-lg border border-white/10 cursor-pointer hover:bg-white/10 transition-all"
                        >
                            <div className="flex items-center gap-3">
                                {trackFaces ? (
                                    <Eye className="w-5 h-5 text-green-400" />
                                ) : (
                                    <EyeOff className="w-5 h-5 text-white/40" />
                                )}
                                <div>
                                    <div className="font-medium text-sm text-white">Face Tracking</div>
                                    <div className="text-[10px] text-white/40">Follow speaker movement</div>
                                </div>
                            </div>
                            <div className={`w-11 h-6 rounded-full transition-all relative ${trackFaces ? "bg-green-600" : "bg-white/20"
                                }`}>
                                <div className={`w-5 h-5 bg-white rounded-full absolute top-0.5 transition-all shadow ${trackFaces ? "left-5" : "left-0.5"
                                    }`} />
                            </div>
                        </div>
                    </div>

                    {/* Timeline Keyframes */}
                    <div className="mb-6">
                        <div className="flex items-center justify-between mb-3">
                            <h3 className="text-xs font-bold text-white/50 uppercase tracking-wider">
                                Timeline Keyframes
                            </h3>
                            <button
                                onClick={addKeyframe}
                                className="flex items-center gap-1 px-2 py-1 bg-purple-600/30 hover:bg-purple-600/50 text-purple-300 text-[10px] font-bold rounded-md transition-all"
                            >
                                <Plus className="w-3 h-3" />
                                Add at {formatTime(currentTime)}
                            </button>
                        </div>

                        {keyframes.length === 0 ? (
                            <div className="p-4 bg-white/5 rounded-lg border border-white/10 text-center">
                                <Clock className="w-6 h-6 text-white/20 mx-auto mb-2" />
                                <p className="text-xs text-white/40">
                                    No keyframes yet. Add keyframes to create layout transitions at specific times.
                                </p>
                            </div>
                        ) : (
                            <div className="space-y-1.5 max-h-40 overflow-y-auto">
                                {keyframes.map((kf, idx) => (
                                    <div
                                        key={idx}
                                        onClick={() => {
                                            setSelectedKeyframeIndex(idx);
                                            if (videoRef.current) {
                                                videoRef.current.currentTime = kf.timestampMs / 1000;
                                            }
                                        }}
                                        className={`flex items-center gap-2 p-2 rounded-lg border cursor-pointer transition-all ${selectedKeyframeIndex === idx
                                                ? "bg-purple-600/30 border-purple-500"
                                                : "bg-white/5 border-white/10 hover:bg-white/10"
                                            }`}
                                    >
                                        <Clock className="w-3 h-3 text-purple-400 flex-shrink-0" />
                                        <div className="flex-1 min-w-0">
                                            <div className="text-xs font-mono text-white">
                                                {formatKeyframeTime(kf.timestampMs)}
                                            </div>
                                            <div className="text-[9px] text-white/40 truncate">
                                                {kf.layoutType === 'podcast' ? 'Podcast (2 speakers)' :
                                                    kf.layoutType === 'split_screen' ? 'Split Screen' :
                                                        'Talking Head'}
                                            </div>
                                        </div>
                                        <button
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                removeKeyframe(idx);
                                            }}
                                            className="p-1 hover:bg-red-500/30 rounded transition-colors"
                                        >
                                            <Trash2 className="w-3 h-3 text-red-400" />
                                        </button>
                                    </div>
                                ))}
                            </div>
                        )}

                        {keyframes.length > 0 && (
                            <p className="text-[9px] text-white/30 mt-2">
                                Click keyframe to seek. Each keyframe defines layout until next keyframe.
                            </p>
                        )}
                    </div>
                </div>
            </div>

            {/* Footer */}
            <div className="flex items-center justify-end gap-4 px-6 py-4 border-t border-white/10 bg-black/50">
                <button
                    onClick={onClose}
                    className="px-6 py-2.5 rounded-lg bg-white/10 hover:bg-white/20 transition-colors font-medium text-white"
                >
                    Cancel
                </button>
                <button
                    onClick={handleApply}
                    className="px-8 py-2.5 rounded-lg bg-purple-600 hover:bg-purple-500 transition-colors font-bold text-white"
                >
                    Apply Layout
                </button>
            </div>
        </div>
    );
}
