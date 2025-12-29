"use client";

import { useState, useCallback, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
    Plus,
    ArrowLeft,
    X,
    FileVideo,
    Loader2,
    Sparkles,
    AlertCircle,
    Film,
    Play,
    Clock,
    ArrowRight,
    Download,
    Upload,
    Type,
    MessageSquare,
    Crop
} from "lucide-react";
import Link from "next/link";
import { useAppStore, CaptionStyle } from "@/lib/store";
import { VisualEditor } from "@/components/editor";

const EpiclipsLogo = ({ size = "w-7 h-7" }: { size?: string }) => (
    <div className={`${size} relative group/logo flex items-center justify-center`}>
        <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="w-full h-full text-white transition-transform duration-500 group-hover/logo:scale-110"
        >
            <path d="M6 5h12" />
            <path d="M4 12h10" />
            <path d="M12 19h8" />
        </svg>
    </div>
);

export default function EditorPage() {
    const {
        videoFile,
        videoUrl,
        stage,
        transcriptionStatus,
        progress,
        logs,
        clips,
        error,
        captionStyle,
        setVideo,
        setCaptionStyle,
        setLayoutOptions,
        reset,
        processVideo
    } = useAppStore();

    // Auto-scroll to logs
    useEffect(() => {
        const el = document.getElementById("logs-end");
        if (el) el.scrollIntoView({ behavior: "smooth" });
    }, [logs]);

    // Auto-scroll to clips when complete
    useEffect(() => {
        if (stage === "complete" && clips.length > 0) {
            setTimeout(() => {
                const clipsSection = document.getElementById("clips-section");
                if (clipsSection) {
                    clipsSection.scrollIntoView({ behavior: "smooth", block: "start" });
                }
            }, 500);
        }
    }, [stage, clips.length]);

    const [isDragging, setIsDragging] = useState(false);

    // Layout mode: auto (AI decides) or custom (user selects)
    const [layoutMode, setLayoutMode] = useState<"auto" | "custom">("auto");
    const [layoutPreset, setLayoutPreset] = useState<"9:16" | "1:1" | "16:9" | "4:3" | "fill">("9:16");
    const [trackFaces, setTrackFaces] = useState(true);
    const [showVisualEditor, setShowVisualEditor] = useState(false);

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith("video/")) {
            setVideo(file);
        }
    };

    const stageLabels: Record<string, string> = {
        idle: "Ready to process",
        loading: "Initializing engine...",
        transcribing: "AI Transcription",
        planning: "AI Viral Planning",
        detecting: "Layout Detection",
        rendering: "High-Speed Rendering",
        complete: "Done!",
        error: "Failed"
    };

    const formatTime = (seconds: number) => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, "0")}`;
    };

    if (!videoFile) {
        return (
            <div className="min-h-screen bg-black text-white selection:bg-white/30">
                {/* Header */}
                <header className="px-8 py-5 flex items-center justify-between border-b border-white/5 bg-black/50 backdrop-blur-xl sticky top-0 z-50">
                    <div className="flex items-center gap-8">
                        <Link href="/" className="group flex items-center gap-2 text-slate-400 hover:text-white transition-colors">
                            <ArrowLeft className="w-4 h-4 group-hover:-translate-x-1 transition-transform" />
                            <span className="text-sm font-medium">Home</span>
                        </Link>
                        <div className="h-4 w-px bg-white/10" />
                        <div className="flex items-center gap-3">
                            <EpiclipsLogo />
                            <span className="text-sm font-bold tracking-tight">Epiclips <span className="text-slate-500 font-medium ml-1">Editor</span></span>
                        </div>
                    </div>
                </header>


                <div className="flex flex-col items-center justify-center pt-24 px-4">
                    <div
                        className={`max-w-2xl w-full p-20 border-2 border-dashed rounded-[40px] transition-all duration-500 group relative overflow-hidden ${isDragging ? 'border-white/40 bg-white/[0.03] scale-[1.02]' : 'border-white/10 bg-white/[0.01] hover:border-white/20'
                            }`}
                        onDragOver={handleDragOver}
                        onDragLeave={handleDragLeave}
                        onDrop={handleDrop}
                    >
                        <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-purple-500/5 opacity-0 group-hover:opacity-100 transition-opacity" />

                        <div className="relative z-10 flex flex-col items-center gap-8 text-center">
                            <div className="w-20 h-20 bg-white/5 rounded-3xl flex items-center justify-center border border-white/10 shadow-2xl group-hover:scale-110 transition-transform duration-500">
                                <Plus className="w-10 h-10 text-white" />
                            </div>
                            <div className="space-y-3">
                                <h1 className="text-4xl font-bold tracking-tight">Upload source video</h1>
                                <p className="text-slate-400 text-lg max-w-sm">We'll identify the most viral segments and format them for social media.</p>
                            </div>

                            <label className="block cursor-pointer">
                                <div className="py-4 px-10 bg-white text-black rounded-2xl font-bold hover:scale-105 active:scale-95 transition-all shadow-xl">
                                    Select MP4 File
                                </div>
                                <input
                                    type="file"
                                    className="hidden"
                                    accept="video/mp4"
                                    onChange={(e) => e.target.files?.[0] && setVideo(e.target.files[0])}
                                />
                            </label>
                            <p className="text-white/20 text-xs font-medium tracking-widest uppercase">Support for MP4 / MOV up to 2GB</p>
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-black text-white selection:bg-white/30">
            {/* Header */}
            <header className="px-8 py-5 flex items-center justify-between border-b border-white/5 bg-black/50 backdrop-blur-xl sticky top-0 z-50">
                <div className="flex items-center gap-8">
                    <Link href="/" className="group flex items-center gap-2 text-slate-400 hover:text-white transition-colors">
                        <ArrowLeft className="w-4 h-4 group-hover:-translate-x-1 transition-transform" />
                        <span className="text-sm font-medium">Home</span>
                    </Link>
                    <div className="h-4 w-px bg-white/10" />
                    <div className="flex items-center gap-3">
                        <EpiclipsLogo />
                        <span className="text-sm font-bold tracking-tight">Epiclips <span className="text-slate-500 font-medium ml-1">Editor</span></span>
                    </div>
                </div>
            </header>


            <main className="max-w-[1700px] mx-auto px-8 py-12">
                <div className="grid lg:grid-cols-[1fr_450px] gap-12 items-start">

                    {/* LEFT AREA */}
                    <div className="space-y-8 animate-in fade-in slide-in-from-bottom-5 duration-700">
                        <div className="relative aspect-video bg-[#050505] rounded-[40px] overflow-hidden border border-white/5 ring-1 ring-white/5 shadow-2xl group box-content">
                            {videoUrl && <video src={videoUrl} className="w-full h-full object-contain" controls />}

                            {/* Processing Overlay */}
                            {stage !== "idle" && stage !== "complete" && stage !== "error" && (
                                <div className="absolute inset-0 z-20 bg-black/80 backdrop-blur-md flex flex-col items-center justify-center p-20 text-center animate-in fade-in duration-500">
                                    <div className="relative w-32 h-32 mb-10">
                                        <div className="absolute inset-0 border-4 border-white/5 rounded-full" />
                                        <div
                                            className="absolute inset-0 border-4 border-blue-500 rounded-full border-t-transparent animate-spin"
                                            style={{ animationDuration: '1.5s' }}
                                        />
                                        <Sparkles className="absolute inset-0 m-auto text-white w-12 h-12" />
                                    </div>

                                    <h2 className="text-4xl font-bold mb-3">{stageLabels[stage]}</h2>
                                    <p className="text-slate-400 text-lg mb-10 font-medium">
                                        {transcriptionStatus || "Optimizing for social media..."}
                                    </p>

                                    {/* Progress Bar Component */}
                                    <div className="w-full max-w-md space-y-4">
                                        <div className="flex justify-between items-end">
                                            <span className="text-xs font-bold uppercase tracking-[0.2em] text-white/30">System Status</span>
                                            <span className="text-lg font-mono font-bold text-white">{Math.round(progress)}%</span>
                                        </div>
                                        <div className="h-2.5 w-full bg-white/5 rounded-full overflow-hidden p-[2px]">
                                            <motion.div
                                                className="h-full bg-gradient-to-r from-blue-600 via-indigo-400 to-purple-500 rounded-full shadow-[0_0_20px_rgba(59,130,246,0.3)]"
                                                initial={{ width: 0 }}
                                                animate={{ width: `${progress}%` }}
                                                transition={{ type: "spring", bounce: 0, duration: 1 }}
                                            />
                                        </div>
                                        <div className="flex justify-between text-[10px] uppercase tracking-widest font-bold text-white/20">
                                            <span>Hardware: Server</span>
                                            <span>AI: Whisper</span>
                                        </div>
                                    </div>
                                </div>
                            )}

                            {stage === "error" && (
                                <div className="absolute inset-0 z-20 bg-black/90 backdrop-blur-xl flex flex-col items-center justify-center p-20 text-center">
                                    <div className="w-20 h-20 bg-red-500/10 rounded-full flex items-center justify-center mb-6">
                                        <AlertCircle className="w-10 h-10 text-red-500" />
                                    </div>
                                    <h2 className="text-3xl font-bold text-red-500 mb-2">Processing Failed</h2>
                                    <p className="text-slate-400 max-w-sm mb-10">{error}</p>
                                    <button onClick={reset} className="px-10 py-4 bg-red-500 text-white rounded-2xl font-bold hover:bg-red-600 transition-colors">Retry Process</button>
                                </div>
                            )}
                        </div>

                        {/* Caption Style Selector - Right below video */}
                        {stage === "idle" && (
                            <div className="space-y-4 animate-in fade-in slide-in-from-bottom-3 duration-500">
                                <h3 className="text-[11px] font-black uppercase tracking-[0.2em] text-slate-500">Caption Style</h3>
                                <div className="grid grid-cols-5 gap-3">
                                    {[
                                        { id: 'opus' as CaptionStyle, name: 'Opus', preview: 'Aa', desc: 'Green Pop' },
                                        { id: 'bold' as CaptionStyle, name: 'Bold', preview: 'Aa', desc: 'Large & Clear' },
                                        { id: 'minimal' as CaptionStyle, name: 'Minimal', preview: 'Aa', desc: 'Subtle' },
                                        { id: 'highlight' as CaptionStyle, name: 'Highlight', preview: 'Aa', desc: 'Yellow Pop' },
                                        { id: 'none' as CaptionStyle, name: 'None', preview: '—', desc: 'No Captions' },
                                    ].map((style) => (
                                        <button
                                            key={style.id}
                                            onClick={() => setCaptionStyle(style.id)}
                                            className={`p-4 rounded-2xl border transition-all text-center group ${captionStyle === style.id
                                                ? 'bg-white/10 border-white/30 ring-2 ring-white/20'
                                                : 'bg-white/[0.02] border-white/5 hover:bg-white/5 hover:border-white/10'
                                                }`}
                                        >
                                            <div className={`text-2xl font-bold mb-2 ${style.id === 'opus' ? 'text-green-400 drop-shadow-[0_2px_4px_rgba(0,0,0,0.8)]' :
                                                style.id === 'bold' ? 'text-white drop-shadow-[0_2px_4px_rgba(0,0,0,0.8)]' :
                                                    style.id === 'minimal' ? 'text-white/70 text-xl' :
                                                        style.id === 'highlight' ? 'text-yellow-400 drop-shadow-[0_2px_4px_rgba(0,0,0,0.8)]' :
                                                            'text-slate-600'
                                                }`}>
                                                {style.preview}
                                            </div>
                                            <div className="text-xs font-bold text-white/80">{style.name}</div>
                                            <div className="text-[10px] text-slate-500">{style.desc}</div>
                                        </button>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Layout Mode Selector - Below Caption Style */}
                        {stage === "idle" && (
                            <div className="space-y-4 animate-in fade-in slide-in-from-bottom-3 duration-500 delay-100">
                                <h3 className="text-[11px] font-black uppercase tracking-[0.2em] text-slate-500">Layout Mode</h3>

                                {/* Auto/Custom Toggle */}
                                <div className="flex gap-3">
                                    <button
                                        onClick={() => setLayoutMode("auto")}
                                        className={`flex-1 p-4 rounded-2xl border transition-all flex items-center justify-center gap-3 ${layoutMode === "auto"
                                            ? "bg-blue-600/20 border-blue-500/50 ring-2 ring-blue-500/30"
                                            : "bg-white/[0.02] border-white/5 hover:bg-white/5"
                                            }`}
                                    >
                                        <Sparkles className={`w-5 h-5 ${layoutMode === "auto" ? "text-blue-400" : "text-slate-500"}`} />
                                        <div className="text-left">
                                            <div className={`font-bold ${layoutMode === "auto" ? "text-white" : "text-slate-300"}`}>Auto</div>
                                            <div className="text-[10px] text-slate-500">AI detects layout & speakers</div>
                                        </div>
                                    </button>

                                    <button
                                        onClick={() => setLayoutMode("custom")}
                                        className={`flex-1 p-4 rounded-2xl border transition-all flex items-center justify-center gap-3 ${layoutMode === "custom"
                                            ? "bg-purple-600/20 border-purple-500/50 ring-2 ring-purple-500/30"
                                            : "bg-white/[0.02] border-white/5 hover:bg-white/5"
                                            }`}
                                    >
                                        <Crop className={`w-5 h-5 ${layoutMode === "custom" ? "text-purple-400" : "text-slate-500"}`} />
                                        <div className="text-left">
                                            <div className={`font-bold ${layoutMode === "custom" ? "text-white" : "text-slate-300"}`}>Custom</div>
                                            <div className="text-[10px] text-slate-500">Manual layout selection</div>
                                        </div>
                                    </button>
                                </div>

                                {/* Custom Mode Options */}
                                {layoutMode === "custom" && (
                                    <div className="p-5 rounded-2xl bg-white/[0.02] border border-white/5 animate-in fade-in slide-in-from-top-2 duration-300">
                                        <button
                                            onClick={() => setShowVisualEditor(true)}
                                            className="w-full p-4 rounded-xl bg-purple-600 hover:bg-purple-500 transition-all flex items-center justify-center gap-3 font-bold"
                                        >
                                            <Crop className="w-5 h-5" />
                                            Open Layout Editor
                                        </button>
                                        <p className="text-center text-xs text-slate-500 mt-3">
                                            Select speakers, adjust crop regions, and set aspect ratio
                                        </p>

                                        {/* Show current settings */}
                                        <div className="mt-4 pt-4 border-t border-white/10 flex items-center justify-between text-sm">
                                            <span className="text-slate-500">Aspect Ratio:</span>
                                            <span className="text-white font-medium">{layoutPreset}</span>
                                        </div>
                                        <div className="mt-2 flex items-center justify-between text-sm">
                                            <span className="text-slate-500">Face Tracking:</span>
                                            <span className={`font-medium ${trackFaces ? "text-green-400" : "text-slate-400"}`}>
                                                {trackFaces ? "On" : "Off"}
                                            </span>
                                        </div>
                                    </div>
                                )}
                            </div>
                        )}

                        {/* Video Info & Start Button */}
                        <div className="flex items-center justify-between p-10 rounded-[35px] bg-[#080808] border border-white/5 shadow-xl ring-1 ring-white/5">
                            <div className="flex items-center gap-6">
                                <div className="w-16 h-16 rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center text-slate-400 shadow-inner">
                                    <FileVideo className="w-8 h-8" />
                                </div>
                                <div className="space-y-1">
                                    <p className="font-bold text-2xl tracking-tight">{videoFile.name}</p>
                                    <p className="text-slate-500 font-medium">{(videoFile.size / (1024 * 1024)).toFixed(1)} MB • Source Material</p>
                                </div>
                            </div>

                            {stage === "idle" ? (
                                <button
                                    onClick={processVideo}
                                    className="px-12 py-5 bg-white text-black rounded-2xl font-bold text-xl hover:scale-[1.03] active:scale-95 transition-all shadow-[0_15px_30px_rgba(255,255,255,0.15)] flex items-center gap-3"
                                >
                                    Process with AI
                                    <Sparkles className="w-5 h-5 fill-current" />
                                </button>
                            ) : (
                                <button onClick={reset} className="px-8 py-4 bg-white/5 text-white/40 rounded-2xl font-bold hover:bg-white/10 transition-all border border-white/5">
                                    Remove
                                </button>
                            )}
                        </div>
                    </div>

                    {/* RIGHT SIDEBAR - Logs Only */}
                    <div className="space-y-5 sticky top-36">
                        <div className="flex items-center justify-between">
                            <h2 className="text-[11px] font-black uppercase tracking-[0.3em] text-slate-500">Live Engine Logs</h2>
                            {stage !== "idle" && (
                                <div className="flex items-center gap-2">
                                    <div className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
                                    <span className="text-[10px] font-bold text-green-500/80 uppercase tracking-widest">Active</span>
                                </div>
                            )}
                        </div>
                        <div className="rounded-[30px] bg-[#050505] border border-white/5 h-[500px] overflow-y-auto p-6 font-mono text-[11px] space-y-2 scrollbar-hide ring-1 ring-white/5">
                            {logs.length === 0 ? (
                                <div className="h-full flex flex-col items-center justify-center opacity-20 contrast-50 gap-4">
                                    <Loader2 className="w-6 h-6 animate-spin" />
                                    <p className="italic">Kernel waiting for task...</p>
                                </div>
                            ) : (
                                logs.map((log, i) => {
                                    const firstBracketEnd = log.indexOf(']');
                                    const timestamp = log.substring(0, firstBracketEnd + 1);
                                    const message = log.substring(firstBracketEnd + 1);
                                    return (
                                        <div key={i} className="text-slate-400 border-l-2 border-white/5 pl-3 py-0.5 animate-in slide-in-from-left-2 duration-300">
                                            <span className="text-slate-600 font-bold mr-3">{timestamp}</span>
                                            <span className="text-slate-300">{message}</span>
                                        </div>
                                    );
                                })
                            )}
                            <div id="logs-end" />
                        </div>
                    </div>
                </div>

                {/* CLIPS SECTION - Below main content */}
                {clips.length > 0 && (
                    <div id="clips-section" className="mt-16 animate-in fade-in slide-in-from-bottom-8 duration-700">
                        {/* Confetti-style celebration header */}
                        <div className="text-center mb-12 relative">
                            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                                <div className="w-96 h-96 bg-gradient-to-r from-blue-500/10 via-purple-500/10 to-pink-500/10 rounded-full blur-3xl" />
                            </div>
                            <motion.div
                                initial={{ scale: 0.8, opacity: 0 }}
                                animate={{ scale: 1, opacity: 1 }}
                                transition={{ type: "spring", bounce: 0.5 }}
                                className="relative"
                            >
                                <span className="text-6xl mb-4 block">🎉</span>
                                <h2 className="text-4xl font-black tracking-tight mb-3">Your Clips Are Ready!</h2>
                                <p className="text-slate-400 text-lg">{clips.length} viral segments extracted and optimized</p>
                            </motion.div>
                        </div>

                        {/* 3-column grid of clips */}
                        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                            {clips.map((clip, i) => (
                                <motion.div
                                    key={i}
                                    initial={{ opacity: 0, y: 30 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: i * 0.15, type: "spring", bounce: 0.3 }}
                                    className="group bg-[#0a0a0a] rounded-[30px] border border-white/5 overflow-hidden hover:border-blue-500/30 transition-all duration-300 hover:shadow-[0_0_40px_rgba(59,130,246,0.15)]"
                                >
                                    {/* Video Preview */}
                                    <div className="relative aspect-[9/16] bg-black">
                                        <video
                                            src={clip.url}
                                            className="w-full h-full object-cover"
                                            controls
                                            playsInline
                                            preload="metadata"
                                        />
                                        <div className="absolute top-4 left-4 px-3 py-1.5 rounded-full bg-black/70 backdrop-blur-md text-[10px] font-black border border-white/10 flex items-center gap-2 z-10">
                                            <Clock className="w-3 h-3 text-blue-400" />
                                            {formatTime(clip.end_time - clip.start_time)}
                                        </div>
                                        <div className="absolute top-4 right-4 px-3 py-1.5 rounded-full bg-gradient-to-r from-green-500 to-emerald-500 text-black text-[10px] font-black shadow-xl z-10 uppercase tracking-tight">
                                            ✓ Ready
                                        </div>
                                    </div>

                                    {/* Clip Info */}
                                    <div className="p-6 space-y-4">
                                        <p className="text-sm font-medium text-slate-300 line-clamp-2 leading-relaxed min-h-[2.5rem]">
                                            "{clip.summary}"
                                        </p>
                                        <div className="flex items-center justify-between pt-4 border-t border-white/5">
                                            <span className="text-[10px] font-black text-slate-600 uppercase tracking-widest">
                                                Clip #{i + 1}
                                            </span>
                                            <a
                                                href={clip.url}
                                                download={`epiclips-${i + 1}.mp4`}
                                                className="px-4 py-2 bg-white text-black rounded-xl text-xs font-bold hover:scale-105 active:scale-95 transition-all flex items-center gap-2 shadow-lg"
                                            >
                                                <Download className="w-3.5 h-3.5" />
                                                Download
                                            </a>
                                        </div>
                                    </div>
                                </motion.div>
                            ))}
                        </div>

                        {/* Download All Button */}
                        <div className="mt-10 flex justify-center">
                            <button className="px-12 py-5 bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 text-white rounded-3xl font-bold text-lg flex items-center gap-4 hover:scale-[1.03] active:scale-[0.98] transition-all shadow-2xl shadow-blue-500/20">
                                <Download className="w-6 h-6" />
                                Download All {clips.length} Clips
                            </button>
                        </div>
                    </div>
                )}
            </main>

            {/* Visual Editor Modal */}
            {videoUrl && (
                <VisualEditor
                    isOpen={showVisualEditor}
                    onClose={() => setShowVisualEditor(false)}
                    videoUrl={videoUrl}
                    onApply={(settings) => {
                        // Update local state
                        setLayoutPreset(settings.layoutPreset);
                        setTrackFaces(settings.trackFaces);
                        setShowVisualEditor(false);

                        // Save to store for processing (enables skip of layout detection)
                        setLayoutOptions({
                            mode: "custom",
                            preset: settings.layoutPreset,
                            customRegions: settings.customRegions,
                            trackFaces: settings.trackFaces,
                            keyframes: settings.keyframes || [],  // Pass timeline keyframes
                        });
                    }}
                />
            )}
        </div>
    );
}
