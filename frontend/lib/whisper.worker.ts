import { pipeline, env } from "@xenova/transformers";

// Configure environment for worker
env.allowLocalModels = false;
env.useBrowserCache = true;

let asr_pipeline: any = null;

interface WordChunk {
    text: string;
    timestamp: [number, number]; // [start, end] in seconds
}

self.onmessage = async (event: MessageEvent) => {
    const { audio, model } = event.data;

    try {
        if (!asr_pipeline) {
            self.postMessage({ status: 'loading', message: 'Initiating AI model...' });

            asr_pipeline = await pipeline('automatic-speech-recognition', model || 'Xenova/whisper-tiny.en', {
                quantized: true,
                progress_callback: (p: any) => {
                    if (p.status === 'progress') {
                        self.postMessage({
                            status: 'model-progress',
                            progress: p.progress,
                            file: p.file
                        });
                    }
                }
            });
            self.postMessage({ status: 'ready', message: 'AI model ready' });
        }

        if (audio) {
            const durationSec = audio.length / 16000;

            self.postMessage({
                status: 'processing',
                message: `Starting inference (${durationSec.toFixed(1)}s audio)...`
            });

            // For word-level timestamps, we process in manageable chunks
            const CHUNK_SIZE_S = 30;
            const allChunks: WordChunk[] = [];
            let fullText = '';

            if (durationSec > 60) {
                const numChunks = Math.ceil(durationSec / CHUNK_SIZE_S);
                addLog(`[AI] Processing in ${numChunks} segments for better feedback.`);

                for (let i = 0; i < durationSec; i += CHUNK_SIZE_S) {
                    const progress = Math.min(100, Math.round((i / durationSec) * 100));
                    self.postMessage({
                        status: 'processing',
                        message: `Transcribing audio...`,
                        progress: progress
                    });

                    const startSample = Math.floor(i * 16000);
                    const endSample = Math.floor(Math.min((i + CHUNK_SIZE_S) * 16000, audio.length));
                    const chunk = audio.slice(startSample, endSample);

                    const chunkResult = await asr_pipeline(chunk, {
                        task: 'transcribe',
                        return_timestamps: 'word'  // Word-level timestamps
                    });

                    // Offset timestamps by chunk start time
                    if (chunkResult.chunks) {
                        for (const c of chunkResult.chunks) {
                            allChunks.push({
                                text: c.text,
                                timestamp: [c.timestamp[0] + i, c.timestamp[1] + i]
                            });
                        }
                    }
                    fullText += chunkResult.text + ' ';
                }

                self.postMessage({
                    status: 'complete',
                    result: {
                        text: fullText.trim(),
                        chunks: allChunks
                    }
                });
            } else {
                // Short audio: process in one go
                const result = await asr_pipeline(audio, {
                    chunk_length_s: 30,
                    stride_length_s: 5,
                    task: 'transcribe',
                    return_timestamps: 'word'  // Word-level timestamps
                });

                // Extract chunks with word timestamps
                const chunks: WordChunk[] = result.chunks?.map((c: any) => ({
                    text: c.text,
                    timestamp: c.timestamp
                })) || [];

                self.postMessage({
                    status: 'complete',
                    result: {
                        text: result.text,
                        chunks: chunks
                    }
                });
            }
        }
    } catch (error: any) {
        console.error(`[Whisper Worker] Error:`, error);
        self.postMessage({ status: 'error', error: `AI Error: ${error.message || 'Unknown error'}` });
    }
};

function addLog(msg: string) {
    self.postMessage({ status: 'ready', message: msg });
}
