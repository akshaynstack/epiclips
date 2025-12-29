import { create } from "zustand";

export interface WordChunk {
    text: string;
    timestamp: [number, number];
}

export type CaptionStyle = "bold" | "minimal" | "highlight" | "opus" | "none";

// Custom layout region (% of video dimensions)
export interface CustomRegion {
    id: string;
    label: string;
    x: number;
    y: number;
    width: number;
    height: number;
}

// Timeline keyframe for multi-layout clips (Advanced Custom Mode)
// Allows users to set different layouts at specific timestamps
export interface LayoutKeyframe {
    timestampMs: number;  // Relative to clip start
    layoutType: 'split_screen' | 'talking_head' | 'podcast';
    faceBbox?: { x: number; y: number; width: number; height: number };  // Normalized 0-1
    podcastBboxes?: { x: number; y: number; width: number; height: number }[];  // For podcast mode
}

export interface LayoutOptions {
    mode: "auto" | "custom";
    preset: "9:16" | "1:1" | "16:9" | "4:3" | "fill";
    customRegions: CustomRegion[];
    trackFaces: boolean;
    podcastMode: boolean;  // When true, use multiple speaker boxes and auto-detect active speaker
    keyframes: LayoutKeyframe[];  // Timeline keyframes for multi-layout rendering
}


interface AppState {
    videoFile: File | null;
    videoUrl: string | null;
    stage: "idle" | "loading" | "transcribing" | "planning" | "detecting" | "rendering" | "complete" | "error";
    transcriptionStatus: string;
    progress: number;
    logs: string[];
    clips: any[];
    error: string | null;

    // Caption system
    captionStyle: CaptionStyle;
    transcriptChunks: WordChunk[];

    // Layout options
    layoutOptions: LayoutOptions;

    setVideo: (file: File) => void;
    setCaptionStyle: (style: CaptionStyle) => void;
    setLayoutOptions: (options: Partial<LayoutOptions>) => void;
    addLog: (msg: string) => void;
    reset: () => void;
    processVideo: () => Promise<void>;
}

const defaultLayoutOptions: LayoutOptions = {
    mode: "auto",
    preset: "9:16",
    customRegions: [],
    trackFaces: true,
    podcastMode: false,
    keyframes: [],  // Empty by default - populated when using timeline keyframes feature
};


export const useAppStore = create<AppState>((set, get) => ({
    videoFile: null,
    videoUrl: null,
    stage: "idle",
    transcriptionStatus: "",
    progress: 0,
    logs: [],
    clips: [],
    error: null,
    captionStyle: "opus",
    transcriptChunks: [],
    layoutOptions: defaultLayoutOptions,


    addLog: (msg) => {
        set((state) => ({
            logs: [...state.logs, `[${new Date().toLocaleTimeString()}] ${msg}`]
        }));
    },

    setVideo: (file) => {
        set({
            videoFile: file,
            videoUrl: URL.createObjectURL(file),
            stage: "idle",
            progress: 0,
            clips: [],
            error: null,
            transcriptChunks: []
        });
    },

    setCaptionStyle: (style) => {
        set({ captionStyle: style });
    },

    setLayoutOptions: (options) => {
        set((state) => ({
            layoutOptions: { ...state.layoutOptions, ...options }
        }));
    },

    reset: () => {
        const { videoUrl } = get();
        if (videoUrl) URL.revokeObjectURL(videoUrl);
        set({
            videoFile: null,
            videoUrl: null,
            stage: "idle",
            progress: 0,
            clips: [],
            error: null,
            logs: [],
            transcriptChunks: [],
            captionStyle: "opus"
        });
    },

    processVideo: async () => {
        const { videoFile, videoUrl, addLog, captionStyle, layoutOptions } = get();
        if (!videoFile || !videoUrl) return;

        const isCustomMode = layoutOptions.mode === "custom" && layoutOptions.customRegions.length > 0;

        const API_URL = 'http://localhost:8000';

        try {
            set({ stage: "loading", progress: 0, logs: [] });
            addLog(`🚀 Starting Backend-Assisted Process for: ${videoFile.name}`);

            // 0. API Health Check
            addLog(`[API] Connecting to Epirium Genesis backend at ${API_URL}...`);
            try {
                const health = await fetch(`${API_URL}/health`);
                if (!health.ok) throw new Error();
                addLog(`[API] ✓ Backend is running and healthy`);
            } catch (e) {
                set({ stage: "error", error: "Backend API not found" });
                addLog(`[API] ✗ Cannot find backend at ${API_URL}`);
                addLog(`[API] ACTION REQUIRED: python run_genesis.py`);
                return;
            }

            // 1. Transcription (Backend Whisper)
            set({ stage: "transcribing", progress: 5, transcriptionStatus: "Uploading to local Whisper..." });
            addLog("Step 1: Local Whisper Transcription (Faster-Whisper Tiny)");

            const transFormData = new FormData();
            transFormData.append("video", videoFile);

            const transRes = await fetch(`${API_URL}/ai-clipping/transcribe-direct`, {
                method: "POST",
                body: transFormData
            });

            if (!transRes.ok) throw new Error("Transcription failed on backend");
            const transcriptData = await transRes.json();

            set({ transcriptChunks: transcriptData.chunks, progress: 30 });
            addLog(`✓ Transcription complete (${transcriptData.chunks.length} words detected)`);
            addLog(`[AI] Transcript: "${transcriptData.transcript.substring(0, 100)}..."`);

            // 2. Planning (Semantic AI)
            set({ stage: "planning", progress: 35 });
            addLog("Step 2: AI Semantic Analysis & Planning");

            const planRes = await fetch(`${API_URL}/ai-clipping/plan-direct`, {
                method: "POST",
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    transcript: transcriptData.transcript,
                    chunks: transcriptData.chunks
                })
            });

            if (!planRes.ok) throw new Error("AI Planning failed on backend");
            const planData = await planRes.json();
            addLog(`✓ Analysis complete. Identified ${planData.clips?.length || 0} viral segments.`);

            // 3. Layout Detection (SKIP in Custom Mode)
            let layoutData;

            if (isCustomMode) {
                // Skip detection - use user-provided regions
                set({ stage: "detecting", progress: 45 });

                // Check if podcast mode is enabled (multiple speakers with auto-detection)
                if (layoutOptions.podcastMode && layoutOptions.customRegions.length >= 2) {
                    addLog("Step 3: ⚡ Using Podcast Mode (auto-detect active speaker)");

                    // Convert all regions to normalized format for backend
                    const speakerBoxes = layoutOptions.customRegions.map(region => ({
                        x: region.x / 100,
                        y: region.y / 100,
                        width: region.width / 100,
                        height: region.height / 100,
                    }));

                    layoutData = {
                        layout_type: "podcast",
                        confidence: 1.0,
                        facecam_bbox: null,
                        podcast_speaker_boxes: speakerBoxes,
                    };
                    addLog(`✓ Podcast mode: ${layoutOptions.customRegions.length} speaker boxes configured`);
                } else {
                    addLog("Step 3: ⚡ Using Custom Layout (skipping AI detection)");

                    // Single speaker mode - use first custom region
                    const region = layoutOptions.customRegions[0];
                    layoutData = {
                        layout_type: "custom",
                        confidence: 1.0,
                        facecam_bbox: region ? {
                            x: region.x / 100,
                            y: region.y / 100,
                            width: region.width / 100,
                            height: region.height / 100,
                        } : null
                    };
                    addLog(`✓ Custom layout applied (${layoutOptions.customRegions.length} region(s))`);
                }

            } else {
                // Auto mode - run AI detection
                set({ stage: "detecting", progress: 45 });
                addLog("Step 3: Layout Detection (Facecam analysis)");

                const layoutFormData = new FormData();
                layoutFormData.append("video", videoFile);

                const layoutRes = await fetch(`${API_URL}/ai-clipping/detect-layout-direct`, {
                    method: "POST",
                    body: layoutFormData
                });

                if (!layoutRes.ok) throw new Error("Layout detection failed on backend");
                layoutData = await layoutRes.json();
                addLog(`✓ Layout: ${layoutData.layout_type} (Confidence: ${Math.round(layoutData.confidence * 100)}%)`);
            }

            // 4. Rendering (Native FFmpeg)
            set({ stage: "rendering", progress: 50 });
            addLog(`Step 4: Native FFmpeg Rendering (${planData.clips.length} clips)`);

            const renderedClips = [];
            for (const [i, clipPlan] of planData.clips.entries()) {
                const clipNum = i + 1;
                addLog(`Rendering Clip ${clipNum}/${planData.clips.length}: "${clipPlan.title}"`);

                // Get words for this clip
                const start = clipPlan.start;
                const end = clipPlan.end;
                const clipChunks = transcriptData.chunks.filter((c: any) =>
                    c.timestamp[0] >= start && c.timestamp[0] < end
                );

                const renderFormData = new FormData();
                renderFormData.append("video", videoFile);
                renderFormData.append("start_time", start.toString());
                renderFormData.append("end_time", end.toString());
                renderFormData.append("layout_type", layoutData.layout_type);
                renderFormData.append("face_bbox", JSON.stringify(layoutData.facecam_bbox));
                renderFormData.append("chunks", JSON.stringify(clipChunks));
                renderFormData.append("caption_style", captionStyle);

                // Pass podcast speaker boxes if in podcast mode
                if (layoutData.podcast_speaker_boxes) {
                    renderFormData.append("podcast_speaker_boxes", JSON.stringify(layoutData.podcast_speaker_boxes));
                }

                // Pass timeline keyframes if configured (Advanced Custom Mode)
                if (layoutOptions.keyframes && layoutOptions.keyframes.length > 0) {
                    renderFormData.append("layout_keyframes", JSON.stringify(layoutOptions.keyframes));
                    addLog(`⏱️ Using ${layoutOptions.keyframes.length} timeline keyframe(s)`);
                }


                const renderRes = await fetch(`${API_URL}/ai-clipping/render-clip-direct`, {
                    method: "POST",
                    body: renderFormData
                });

                if (!renderRes.ok) {
                    addLog(`✗ Clip ${clipNum} failed to render`);
                    continue;
                }

                const clipBlob = await renderRes.blob();
                renderedClips.push({
                    ...clipPlan,
                    blob: clipBlob,
                    url: URL.createObjectURL(clipBlob)
                });

                const stepProgress = 50 + ((clipNum / planData.clips.length) * 50);
                set({ progress: Math.min(99, stepProgress) });
                addLog(`✓ Clip ${clipNum} ready`);
            }

            if (renderedClips.length === 0) throw new Error("All clip renders failed");

            addLog(`🎉 MISSION COMPLETE! ${renderedClips.length} clips generated and ready for review.`);
            set({ stage: "complete", clips: renderedClips, progress: 100 });

        } catch (err: any) {
            addLog(`❌ Process Failed: ${err.message}`);
            set({ stage: "error", error: err.message });
        }
    }
}));
