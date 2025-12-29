import { NextRequest, NextResponse } from "next/server";

export async function POST(req: NextRequest) {
    try {
        const { transcript, duration } = await req.json();

        if (!transcript) {
            return NextResponse.json({ error: "No transcript provided" }, { status: 400 });
        }

        const prompt = `
      You are a viral video expert. Analyze the following transcript and identify 3 potential viral clips.
      Each clip should be between 15-30 seconds long.
      
      CRITICAL: You MUST return a JSON object with this exact structure:
      {
        "clips": [
          {
            "start_time": number, // in seconds
            "end_time": number,   // in seconds
            "summary": string     // brief catchy title
          }
        ]
      }

      Transcript: ${transcript}
      Video Duration: ${duration}s
    `;

        const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "Authorization": `Bearer ${process.env.OPENROUTER_API_KEY}`,
                "HTTP-Referer": "https://epiclips.ai",
                "X-Title": "Epiclips",
            },
            body: JSON.stringify({
                model: "nvidia/nemotron-3-nano-30b-a3b:free",
                messages: [{ role: "user", content: prompt }],
                response_format: { type: "json_object" },
            }),
        });

        if (!response.ok) {
            const error = await response.json();
            return NextResponse.json(error, { status: response.status });
        }

        const data = await response.json();
        const content = JSON.parse(data.choices[0].message.content);

        return NextResponse.json(content);
    } catch (error) {
        console.error("AI Planning error:", error);
        return NextResponse.json({ error: "AI Planning failed" }, { status: 500 });
    }
}
