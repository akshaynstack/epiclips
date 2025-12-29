import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
    try {
        const formData = await req.formData();
        const audioFile = formData.get("file") as Blob;

        if (!audioFile) {
            return NextResponse.json({ error: "No audio file provided" }, { status: 400 });
        }

        // Convert blob to file for Groq API
        const file = new File([audioFile], "audio.mp3", { type: "audio/mp3" });

        // Create new FormData for Groq
        const groqFormData = new FormData();
        groqFormData.append("file", file);
        groqFormData.append("model", "whisper-large-v3-turbo");
        groqFormData.append("response_format", "verbose_json");

        const response = await fetch("https://api.groq.com/openai/v1/audio/transcriptions", {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${process.env.GROQ_API_KEY}`,
            },
            body: groqFormData,
        });

        if (!response.ok) {
            const error = await response.json();
            return NextResponse.json(error, { status: response.status });
        }

        const data = await response.json();
        return NextResponse.json(data);
    } catch (error) {
        console.error("Transcription error:", error);
        return NextResponse.json({ error: "Transcription failed" }, { status: 500 });
    }
}
