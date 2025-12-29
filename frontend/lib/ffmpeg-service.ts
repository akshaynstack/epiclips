import { FFmpeg } from "@ffmpeg/ffmpeg";
import { fetchFile, toBlobURL } from "@ffmpeg/util";
import { CaptionStyle } from "./store";

let ffmpeg: FFmpeg | null = null;
let coreURL: string | null = null;
let wasmURL: string | null = null;

interface WordChunk {
    text: string;
    timestamp: [number, number];
}

/**
 * Loads FFmpeg.wasm or returns existing instance.
 */
export async function loadFFmpeg(): Promise<FFmpeg> {
    if (ffmpeg) return ffmpeg;

    ffmpeg = new FFmpeg();

    ffmpeg.on("log", ({ message }) => {
        console.log(`[FFmpeg] ${message}`);
    });

    if (!coreURL) {
        coreURL = await toBlobURL("/ffmpeg/ffmpeg-core.js", "text/javascript");
        wasmURL = await toBlobURL("/ffmpeg/ffmpeg-core.wasm", "application/wasm");
    }

    await ffmpeg.load({
        coreURL: coreURL,
        wasmURL: wasmURL!,
    });

    return ffmpeg;
}

/**
 * Terminates and reloads FFmpeg to fully reset the WASM heap.
 */
export async function resetFFmpeg(): Promise<void> {
    if (ffmpeg) {
        try {
            ffmpeg.terminate();
        } catch (e) {
            console.warn("FFmpeg terminate failed:", e);
        }
        ffmpeg = null;
    }
    await new Promise(resolve => setTimeout(resolve, 100));
}

async function getFFmpeg(): Promise<FFmpeg> {
    return await loadFFmpeg();
}

/**
 * Generate simple drawtext filter for captions.
 * Chunks should already have timestamps relative to clip start.
 */
function generateSimpleDrawtextFilter(
    chunks: WordChunk[],
    style: CaptionStyle
): string {
    if (!chunks || chunks.length === 0 || style === 'none') {
        return '';
    }

    // Style configurations (PREMIUM LOOK)
    const styles: Record<CaptionStyle, { fontsize: number; fontcolor: string; borderw: number }> = {
        bold: { fontsize: 84, fontcolor: 'white', borderw: 4 },
        minimal: { fontsize: 56, fontcolor: 'white', borderw: 2 },
        highlight: { fontsize: 80, fontcolor: 'yellow', borderw: 4 },
        opus: { fontsize: 75, fontcolor: '#00FF00', borderw: 5 },
        none: { fontsize: 0, fontcolor: '', borderw: 0 }
    };

    const s = styles[style] || styles.bold;

    // Group words into lines (3-4 words per line for big punchy captions)
    const WORDS_PER_LINE = 4;
    const lines: { text: string; start: number; end: number }[] = [];

    for (let i = 0; i < chunks.length; i += WORDS_PER_LINE) {
        const group = chunks.slice(i, i + WORDS_PER_LINE);
        if (group.length === 0) continue;

        const text = group.map(c => c.text.trim().toUpperCase()).join(' '); // UPPERCASE for impact
        const start = Math.max(0, group[0].timestamp[0]);
        const end = Math.max(0.1, group[group.length - 1].timestamp[1]);

        if (lines.length >= 20) break; // Allow more lines with small chunks

        // Strict sanitization
        const cleanText = text
            .replace(/['":\\[\]{}()]/g, '')
            .replace(/\s+/g, ' ')
            .trim();

        if (cleanText && end > start) {
            lines.push({ text: cleanText, start, end });
        }
    }

    if (lines.length === 0) return '';

    const fontFile = 'caption.ttf';
    const filters = lines.map((line) => {
        const st = line.start.toFixed(2);
        const en = line.end.toFixed(2);

        // Premium Caption:
        // - box=1: semi-transparent background box
        // - uppercase: for a modern 'viral' look
        // - y=h*0.6: positioned in the safe area above facecam
        return `drawtext=fontfile=${fontFile}:text='${line.text}':fontsize=${s.fontsize}:fontcolor=${s.fontcolor}:borderw=${s.borderw}:bordercolor=black:box=1:boxcolor=black@0.4:boxborderw=10:x=(w-text_w)/2:y=h*0.60:enable='gte(t\\,${st})*lt(t\\,${en})'`;
    });

    return filters.join(',');
}

/**
 * Extract audio from a video file for transcription.
 */
export async function extractAudio(
    videoFile: File,
    onLog?: (msg: string) => void
): Promise<Blob> {
    const ffmpegInstance = await getFFmpeg();
    if (onLog) {
        ffmpegInstance.on("log", ({ message }) => onLog(message));
    }

    const inputName = "input.mp4";
    const outputName = "output.wav";

    await ffmpegInstance.writeFile(inputName, await fetchFile(videoFile));

    await ffmpegInstance.exec([
        "-i", inputName,
        "-vn",
        "-ar", "16000",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        outputName
    ]);

    const data = await ffmpegInstance.readFile(outputName) as Uint8Array;
    const blob = new Blob([data as any], { type: "audio/wav" });

    try { await ffmpegInstance.deleteFile(inputName); } catch (e) { }
    try { await ffmpegInstance.deleteFile(outputName); } catch (e) { }

    return blob;
}

/**
 * Extract a single frame at a specific timestamp.
 */
export async function extractSingleFrame(
    videoFile: File,
    timestampSec: number,
    onLog?: (msg: string) => void
): Promise<string | null> {
    const ffmpegInstance = await getFFmpeg();
    if (onLog) {
        ffmpegInstance.on("log", ({ message }) => onLog(message));
    }

    const inputName = "frame_input.mp4";
    const outputName = "frame_output.jpg";

    try {
        await ffmpegInstance.writeFile(inputName, await fetchFile(videoFile));

        await ffmpegInstance.exec([
            "-i", inputName,
            "-ss", Math.max(0, timestampSec).toFixed(2),
            "-frames:v", "1",
            "-q:v", "10",
            "-y",
            outputName
        ]);

        const data = await ffmpegInstance.readFile(outputName) as Uint8Array;

        try { await ffmpegInstance.deleteFile(inputName); } catch (e) { }
        try { await ffmpegInstance.deleteFile(outputName); } catch (e) { }

        if (data.length > 0) {
            const blob = new Blob([data as any], { type: "image/jpeg" });
            return URL.createObjectURL(blob);
        }
        return null;
    } catch (e) {
        console.warn(`Frame extraction failed at ${timestampSec}s:`, e);
        try { await ffmpegInstance.deleteFile(inputName); } catch (e) { }
        try { await ffmpegInstance.deleteFile(outputName); } catch (e) { }
        return null;
    }
}

/**
 * Extract frames one at a time with FFmpeg reset between each.
 * This uses more memory-safe approach for large files.
 */
export async function extractFrames(
    videoFile: File,
    timestamps: number[],
    onLog?: (msg: string) => void
): Promise<string[]> {
    const fileSizeMB = videoFile.size / (1024 * 1024);
    onLog?.(`Extracting ${timestamps.length} frames from ${fileSizeMB.toFixed(1)}MB file...`);

    const frames: string[] = [];
    const fileData = await fetchFile(videoFile);

    // Extract each frame in a separate FFmpeg session
    for (let i = 0; i < timestamps.length; i++) {
        const ts = timestamps[i];

        try {
            // Reset FFmpeg for each frame to avoid memory buildup
            await resetFFmpeg();
            const ffmpegInstance = await loadFFmpeg();

            if (onLog) {
                ffmpegInstance.on("log", ({ message }) => {
                    // Only log key messages
                    if (message.includes("frame=") || message.includes("error") || message.includes("Error")) {
                        onLog(message);
                    }
                });
            }

            const inputName = `input_${i}.mp4`;
            const outputName = `frame_${i}.jpg`;

            // Write video
            await ffmpegInstance.writeFile(inputName, fileData);

            // Extract frame with minimal decoding
            await ffmpegInstance.exec([
                "-ss", Math.max(0, ts).toFixed(2),  // Seek before input for faster decode
                "-i", inputName,
                "-frames:v", "1",
                "-q:v", "15",  // Lower quality = less memory
                "-update", "1",
                "-y",
                outputName
            ]);

            const data = await ffmpegInstance.readFile(outputName) as Uint8Array;

            if (data.length > 0) {
                const blob = new Blob([data as any], { type: "image/jpeg" });
                frames.push(URL.createObjectURL(blob));
                onLog?.(`✓ Frame ${i + 1}/${timestamps.length} extracted`);
            } else {
                onLog?.(`⚠ Frame ${i + 1} empty`);
            }

            // Cleanup
            try { await ffmpegInstance.deleteFile(inputName); } catch (e) { }
            try { await ffmpegInstance.deleteFile(outputName); } catch (e) { }

        } catch (e: any) {
            onLog?.(`✗ Frame ${i + 1} failed: ${e.message || e}`);
            // Continue with remaining frames
        }
    }

    // Final reset to free memory
    await resetFFmpeg();

    return frames;
}

/**
 * Render a clip with layout handling and caption burning.
 * Uses simple -vf chain for reliability in WASM environment.
 */
export async function renderClip(
    videoFile: File,
    startMs: number,
    durationMs: number,
    layout: "talking_head" | "screen_share",
    config: {
        faceBbox?: { x: number; y: number; w: number; h: number };
        captionStyle?: CaptionStyle;
        wordChunks?: WordChunk[];
    },
    onLog?: (msg: string) => void
): Promise<Blob> {
    const ffmpegInstance = await getFFmpeg();
    if (onLog) {
        ffmpegInstance.on("log", ({ message }) => onLog(message));
    }

    const inputName = "clip_input.mp4";
    const outputName = "clip_output.mp4";
    const fontName = "caption.ttf";

    onLog?.("Loading video for rendering...");
    await ffmpegInstance.writeFile(inputName, await fetchFile(videoFile));

    // Load font for captions
    const hasChunks = config.wordChunks && config.wordChunks.length > 0 && config.captionStyle !== 'none';
    let fontLoaded = false;
    if (hasChunks) {
        try {
            onLog?.("Loading caption font...");
            const fontResponse = await fetch('/fonts/caption.ttf');
            onLog?.(`Font fetch status: ${fontResponse.status} ${fontResponse.ok ? 'OK' : 'FAILED'}`);
            if (fontResponse.ok) {
                const fontData = await fontResponse.arrayBuffer();
                onLog?.(`Font data size: ${fontData.byteLength} bytes`);
                await ffmpegInstance.writeFile(fontName, new Uint8Array(fontData));
                fontLoaded = true;
                onLog?.("✓ Font loaded into WASM filesystem");
            } else {
                onLog?.(`⚠ Font fetch failed with status: ${fontResponse.status}`);
            }
        } catch (e: any) {
            onLog?.(`⚠ Font load error: ${e.message}`);
        }
    }

    onLog?.("Starting render...");

    const startSec = startMs / 1000;
    const durationSec = durationMs / 1000;

    // Build video filter based on layout
    let videoFilter: string;

    if (layout === "screen_share" && config.faceBbox) {
        // SPLIT-SCREEN LAYOUT: Main content (75%) on top, facecam (25%) on bottom
        // Vertical 9:16 output = 1080x1920
        //  - Top section: 1080x1440 (main screen content)
        //  - Bottom section: 1080x480 (facecam, scaled up)

        const fb = config.faceBbox;
        onLog?.(`Split-screen mode: facecam at (${fb.x},${fb.y}) ${fb.w}x${fb.h}`);

        // Determine where the main content is (opposite to facecam)
        // If facecam is on right, main content is on left
        // Crop main content from left 75% of the frame (excluding facecam area)
        const mainCropWidth = 1440;  // 16:9 aspect for main content area
        const mainCropHeight = 810;
        const mainCropX = 0;  // Start from left (facecam is usually in bottom-right)
        const mainCropY = 0;  // Start from top

        // Facecam crop - expand the bbox a bit for padding
        const fcPad = 20;
        const fcX = Math.max(0, fb.x - fcPad);
        const fcY = Math.max(0, fb.y - fcPad);
        const fcW = fb.w + fcPad * 2;
        const fcH = fb.h + fcPad * 2;

        // Build the filter:
        // 1. [0:v] -> split into two streams
        // 2. First stream: crop main content, scale to 1080x1440
        // 3. Second stream: crop facecam, scale to 1080x480
        // 4. Stack vertically
        videoFilter = `[0:v]split=2[main][fc];` +
            `[main]crop=${mainCropWidth}:${mainCropHeight}:${mainCropX}:${mainCropY},scale=1080:1440:flags=fast_bilinear[top];` +
            `[fc]crop=${fcW}:${fcH}:${fcX}:${fcY},scale=1080:480:flags=fast_bilinear[bottom];` +
            `[top][bottom]vstack=inputs=2`;

        onLog?.(`Using split-screen filter (top: main content, bottom: facecam)`);
    } else {
        // TALKING HEAD LAYOUT: Simple center crop to 9:16
        videoFilter = `crop=ih*9/16:ih:(iw-ih*9/16)/2:0,scale=1080:1920:flags=fast_bilinear`;
        onLog?.(`Using center-crop filter (talking head)`);
    }

    // Debug: log caption inputs
    onLog?.(`Caption config: style=${config.captionStyle}, chunks=${config.wordChunks?.length || 0}`);

    // Generate and add caption filter (timestamps already relative to clip start)
    const captionFilter = generateSimpleDrawtextFilter(
        config.wordChunks || [],
        config.captionStyle || 'none'
    );

    // Determine if we're using split-screen (requires filter_complex) or simple filter
    const isSplitScreen = layout === "screen_share" && config.faceBbox;

    try {
        let args: string[];

        if (isSplitScreen) {
            // SPLIT-SCREEN: Use filter_complex with vstack
            const fb = config.faceBbox!;
            onLog?.(`Split-screen mode: facecam at (${fb.x},${fb.y}) ${fb.w}x${fb.h}`);

            // Crop main content from left side (where screen share usually is)
            // For a 1920x1080 video, crop 1440x810 from top-left
            const mainW = 1440;
            const mainH = 810;
            const mainX = 0;
            const mainY = 0;

            // Expand facecam bbox with padding
            const pad = 40;
            const fcX = Math.max(0, fb.x - pad);
            const fcY = Math.max(0, fb.y - pad);
            const fcW = fb.w + pad * 2;
            const fcH = fb.h + pad * 2;

            // Build filter_complex: split, crop both, scale to 9:16 portions, stack
            // IMPORTANT: Facecam uses force_original_aspect_ratio to prevent stretching
            // Then pad with black bars to center it
            let filterComplex = `[0:v]split=2[main][fc];` +
                `[main]crop=${mainW}:${mainH}:${mainX}:${mainY},scale=1080:1440:flags=bilinear[top];` +
                `[fc]crop=${fcW}:${fcH}:${fcX}:${fcY},scale=1080:480:force_original_aspect_ratio=decrease,pad=1080:480:(ow-iw)/2:(oh-ih)/2:black[bottom];` +
                `[top][bottom]vstack=inputs=2`;

            // Add captions if available
            if (captionFilter) {
                filterComplex += `,${captionFilter}`;
                onLog?.(`✓ Caption filter added to split-screen (${captionFilter.length} chars)`);
            }

            filterComplex += `[out]`;

            args = [
                "-i", inputName,
                "-ss", startSec.toFixed(2),
                "-t", durationSec.toFixed(2),
                "-filter_complex", filterComplex,
                "-map", "[out]",
                "-map", "0:a?",
                "-c:v", "libx264",
                "-crf", "30",
                "-preset", "ultrafast",
                "-c:a", "aac",
                "-b:a", "96k",
                "-movflags", "+faststart",
                "-y",
                outputName
            ];

            onLog?.(`Using split-screen rendering (top: screen, bottom: facecam)`);
        } else {
            // TALKING HEAD: Simple -vf chain
            if (captionFilter) {
                videoFilter += `,${captionFilter}`;
                onLog?.(`✓ Caption filter generated (${captionFilter.length} chars)`);
            } else {
                onLog?.(`⚠ No caption filter generated`);
            }

            args = [
                "-i", inputName,
                "-ss", startSec.toFixed(2),
                "-t", durationSec.toFixed(2),
                "-vf", videoFilter,
                "-c:v", "libx264",
                "-crf", "30",
                "-preset", "ultrafast",
                "-c:a", "aac",
                "-b:a", "96k",
                "-movflags", "+faststart",
                "-y",
                outputName
            ];

            onLog?.(`Using center-crop rendering (talking head)`);
        }

        await ffmpegInstance.exec(args);

        const data = await ffmpegInstance.readFile(outputName) as Uint8Array;

        if (data.length === 0) {
            // Retry without captions if first attempt fails
            onLog?.("Render failed with captions, retrying without...");

            const simpleArgs = [
                "-i", inputName,
                "-ss", startSec.toFixed(2),
                "-t", durationSec.toFixed(2),
                "-vf", "crop=ih*9/16:ih:(iw-ih*9/16)/2:0,scale=1080:1920:flags=fast_bilinear",
                "-c:v", "libx264",
                "-crf", "30",
                "-preset", "ultrafast",
                "-c:a", "aac",
                "-b:a", "96k",
                "-y",
                outputName
            ];

            await ffmpegInstance.exec(simpleArgs);
            const retryData = await ffmpegInstance.readFile(outputName) as Uint8Array;

            if (retryData.length === 0) {
                throw new Error("Render produced empty output");
            }

            const blob = new Blob([retryData as any], { type: "video/mp4" });
            try { await ffmpegInstance.deleteFile(inputName); } catch (e) { }
            try { await ffmpegInstance.deleteFile(outputName); } catch (e) { }
            return blob;
        }

        const blob = new Blob([data as any], { type: "video/mp4" });

        try { await ffmpegInstance.deleteFile(inputName); } catch (e) { }
        try { await ffmpegInstance.deleteFile(outputName); } catch (e) { }

        return blob;
    } catch (error: any) {
        onLog?.(`FFmpeg error: ${error.message}`);
        try { await ffmpegInstance.deleteFile(inputName); } catch (e) { }
        try { await ffmpegInstance.deleteFile(outputName); } catch (e) { }
        throw new Error(`Render failed: ${error.message || error}`);
    }
}
