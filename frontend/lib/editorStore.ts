import { create } from "zustand";

// Layout preset types matching OpusClip
export type LayoutPreset = "9:16" | "1:1" | "16:9" | "4:3" | "9:8" | "original" | "custom";

// Face detection result from backend
export interface Face {
    id: string;
    label: string;
    bbox: {
        x: number;
        y: number;
        width: number;
        height: number;
    };
    trackId: number;
    isActiveSpeaker: boolean;
    confidence: number;
}

// Crop region for each face/speaker
export interface CropRegion {
    faceId: string;
    x: number;
    y: number;
    width: number;
    height: number;
    locked: boolean; // Lock aspect ratio
}

// Speaker information
export interface Speaker {
    id: string;
    label: string;
    faceId: string;
    color: string;
}

// Timeline segment
export interface TimelineSegment {
    id: string;
    startMs: number;
    endMs: number;
    layoutType: "fill" | "split" | "custom";
    cropRegions: CropRegion[];
}

interface EditorState {
    // Video info
    clipId: string | null;
    videoUrl: string | null;
    videoDuration: number;
    videoWidth: number;
    videoHeight: number;

    // Playback
    currentTime: number;
    isPlaying: boolean;

    // Layout
    layoutPreset: LayoutPreset;
    customAspect: { width: number; height: number };
    outputWidth: number;
    outputHeight: number;

    // Face tracking
    faces: Face[];
    cropRegions: CropRegion[];
    selectedFaceId: string | null;
    trackingEnabled: boolean;

    // Speakers
    speakers: Speaker[];

    // Timeline
    segments: TimelineSegment[];
    thumbnailSpriteUrl: string | null;

    // UI state
    isLoading: boolean;
    isDirty: boolean; // Has unsaved changes
    error: string | null;

    // Actions - Video
    setClip: (clipId: string, videoUrl: string, width: number, height: number, duration: number) => void;
    setCurrentTime: (time: number) => void;
    setPlaying: (playing: boolean) => void;

    // Actions - Layout
    setLayoutPreset: (preset: LayoutPreset) => void;
    setCustomAspect: (width: number, height: number) => void;

    // Actions - Faces
    setFaces: (faces: Face[]) => void;
    selectFace: (faceId: string | null) => void;
    setTrackingEnabled: (enabled: boolean) => void;

    // Actions - Crop
    updateCropRegion: (faceId: string, region: Partial<CropRegion>) => void;
    setCropRegions: (regions: CropRegion[]) => void;

    // Actions - Speakers
    setSpeakerLabel: (speakerId: string, label: string) => void;

    // Actions - Segments
    setSegments: (segments: TimelineSegment[]) => void;

    // Actions - Persistence
    applyChanges: () => Promise<void>;
    resetChanges: () => void;

    // Computed helpers
    getAspectRatio: () => number;
    getOutputDimensions: () => { width: number; height: number };
}

// Aspect ratio map
const ASPECT_RATIOS: Record<LayoutPreset, { width: number; height: number } | null> = {
    "9:16": { width: 9, height: 16 },
    "1:1": { width: 1, height: 1 },
    "16:9": { width: 16, height: 9 },
    "4:3": { width: 4, height: 3 },
    "9:8": { width: 9, height: 8 },
    "original": null,
    "custom": null,
};

// Default speaker colors
const SPEAKER_COLORS = ["#a855f7", "#3b82f6", "#22c55e", "#f97316", "#ec4899"];

export const useEditorStore = create<EditorState>((set, get) => ({
    // Initial state
    clipId: null,
    videoUrl: null,
    videoDuration: 0,
    videoWidth: 1920,
    videoHeight: 1080,

    currentTime: 0,
    isPlaying: false,

    layoutPreset: "9:16",
    customAspect: { width: 9, height: 16 },
    outputWidth: 1080,
    outputHeight: 1920,

    faces: [],
    cropRegions: [],
    selectedFaceId: null,
    trackingEnabled: true,

    speakers: [],

    segments: [],
    thumbnailSpriteUrl: null,

    isLoading: false,
    isDirty: false,
    error: null,

    // Actions
    setClip: (clipId, videoUrl, width, height, duration) => {
        set({
            clipId,
            videoUrl,
            videoWidth: width,
            videoHeight: height,
            videoDuration: duration,
            currentTime: 0,
            isPlaying: false,
            faces: [],
            cropRegions: [],
            speakers: [],
            segments: [],
            isDirty: false,
            error: null,
        });
    },

    setCurrentTime: (time) => set({ currentTime: time }),
    setPlaying: (playing) => set({ isPlaying: playing }),

    setLayoutPreset: (preset) => {
        const dimensions = get().getOutputDimensions();
        set({
            layoutPreset: preset,
            outputWidth: dimensions.width,
            outputHeight: dimensions.height,
            isDirty: true,
        });
    },

    setCustomAspect: (width, height) => {
        set({
            customAspect: { width, height },
            isDirty: true,
        });
    },

    setFaces: (faces) => {
        const { speakers } = get();

        // Auto-create speakers for new faces
        const newSpeakers = [...speakers];
        faces.forEach((face, i) => {
            if (!newSpeakers.find(s => s.faceId === face.id)) {
                newSpeakers.push({
                    id: `speaker_${face.id}`,
                    label: `Speaker ${newSpeakers.length + 1}`,
                    faceId: face.id,
                    color: SPEAKER_COLORS[newSpeakers.length % SPEAKER_COLORS.length],
                });
            }
        });

        // Auto-create crop regions for faces
        const { cropRegions, videoWidth, videoHeight, outputWidth, outputHeight } = get();
        const newCropRegions = faces.map(face => {
            const existing = cropRegions.find(r => r.faceId === face.id);
            if (existing) return existing;

            // Default: center face in a square region
            const size = Math.max(face.bbox.width, face.bbox.height) * 1.5;
            return {
                faceId: face.id,
                x: Math.max(0, face.bbox.x + face.bbox.width / 2 - size / 2),
                y: Math.max(0, face.bbox.y + face.bbox.height / 2 - size / 2),
                width: size,
                height: size,
                locked: true,
            };
        });

        set({
            faces,
            speakers: newSpeakers,
            cropRegions: newCropRegions,
        });
    },

    selectFace: (faceId) => set({ selectedFaceId: faceId }),
    setTrackingEnabled: (enabled) => set({ trackingEnabled: enabled, isDirty: true }),

    updateCropRegion: (faceId, region) => {
        const { cropRegions } = get();
        const updated = cropRegions.map(r =>
            r.faceId === faceId ? { ...r, ...region } : r
        );
        set({ cropRegions: updated, isDirty: true });
    },

    setCropRegions: (regions) => set({ cropRegions: regions, isDirty: true }),

    setSpeakerLabel: (speakerId, label) => {
        const { speakers } = get();
        const updated = speakers.map(s =>
            s.id === speakerId ? { ...s, label } : s
        );
        set({ speakers: updated, isDirty: true });
    },

    setSegments: (segments) => set({ segments, isDirty: true }),

    applyChanges: async () => {
        const state = get();
        if (!state.clipId) return;

        set({ isLoading: true, error: null });

        try {
            const response = await fetch(`http://localhost:8000/clips/${state.clipId}/layout`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    layout_type: state.layoutPreset,
                    aspect_ratio: `${state.customAspect.width}:${state.customAspect.height}`,
                    crop_regions: state.cropRegions.map(r => ({
                        face_id: r.faceId,
                        x: Math.round(r.x),
                        y: Math.round(r.y),
                        width: Math.round(r.width),
                        height: Math.round(r.height),
                    })),
                    track_faces: state.trackingEnabled,
                    speakers: state.speakers.map(s => ({
                        id: s.id,
                        label: s.label,
                        face_id: s.faceId,
                    })),
                }),
            });

            if (!response.ok) throw new Error("Failed to save changes");

            set({ isDirty: false, isLoading: false });
        } catch (error: any) {
            set({ error: error.message, isLoading: false });
        }
    },

    resetChanges: () => {
        // TODO: Reload from server
        set({ isDirty: false });
    },

    // Helpers
    getAspectRatio: () => {
        const { layoutPreset, customAspect, videoWidth, videoHeight } = get();

        if (layoutPreset === "original") {
            return videoWidth / videoHeight;
        }

        if (layoutPreset === "custom") {
            return customAspect.width / customAspect.height;
        }

        const preset = ASPECT_RATIOS[layoutPreset];
        return preset ? preset.width / preset.height : 9 / 16;
    },

    getOutputDimensions: () => {
        const aspectRatio = get().getAspectRatio();

        // Standard output width
        const baseWidth = 1080;
        const height = Math.round(baseWidth / aspectRatio);

        return { width: baseWidth, height };
    },
}));
