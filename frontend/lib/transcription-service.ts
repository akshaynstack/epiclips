/**
 * Word-level timestamp chunk from Whisper transcription
 */
export interface WordChunk {
    text: string;
    timestamp: [number, number]; // [start, end] in seconds
}

/**
 * Transcription result with text and word-level timestamps
 */
export interface TranscriptionResult {
    text: string;
    chunks: WordChunk[];
}

/**
 * Transcribes an audio blob using OpenAI Whisper (tiny.en) via a Web Worker.
 * Returns word-level timestamps for caption generation.
 */
export async function transcribeAudio(
    audioBlob: Blob,
    onProgress?: (status: string, progress?: number) => void
): Promise<TranscriptionResult> {
    return new Promise(async (resolve, reject) => {
        try {
            // 1. Convert Blob to AudioBuffer (16kHz mono) on the main thread
            onProgress?.("Decoding audio file...");
            const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
            const arrayBuffer = await audioBlob.arrayBuffer();
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
            const audioData = audioBuffer.getChannelData(0);
            await audioContext.close();
            onProgress?.("Audio decoded. Initializing worker...");

            // 2. Initialize Worker
            const worker = new Worker(new URL('./whisper.worker.ts', import.meta.url));

            worker.onmessage = (event) => {
                const { status, message, progress, result, error } = event.data;

                switch (status) {
                    case 'loading':
                    case 'ready':
                    case 'processing':
                        onProgress?.(message || status);
                        break;
                    case 'model-progress':
                        onProgress?.(`Downloading AI: ${Math.round(progress)}%`, progress);
                        break;
                    case 'complete':
                        worker.terminate();
                        resolve({
                            text: result.text,
                            chunks: result.chunks || []
                        });
                        break;
                    case 'error':
                        worker.terminate();
                        reject(new Error(error));
                        break;
                }
            };

            worker.onerror = (err) => {
                worker.terminate();
                reject(new Error("Worker error: " + err.message));
            };

            // 3. Start Transcription
            worker.postMessage({ audio: audioData });

        } catch (error) {
            reject(error);
        }
    });
}
