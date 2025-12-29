"use client";

import React from "react";
import { useEditorStore } from "@/lib/editorStore";
import { User, Check } from "lucide-react";

interface SpeakerManagerProps {
    className?: string;
}

export function SpeakerManager({ className }: SpeakerManagerProps) {
    const { speakers, faces, selectedFaceId, selectFace, setSpeakerLabel } = useEditorStore();

    if (speakers.length === 0) {
        return (
            <div className={`p-4 bg-zinc-800/50 rounded-lg ${className}`}>
                <p className="text-sm text-zinc-500">No speakers detected</p>
            </div>
        );
    }

    return (
        <div className={`flex flex-col gap-3 ${className}`}>
            <h3 className="text-sm font-semibold text-zinc-300 uppercase tracking-wide">
                Speakers
            </h3>

            {speakers.map((speaker) => {
                const face = faces.find(f => f.id === speaker.faceId);
                const isSelected = selectedFaceId === speaker.faceId;
                const isActive = face?.isActiveSpeaker;

                return (
                    <div
                        key={speaker.id}
                        className={`
              flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-all
              ${isSelected
                                ? "bg-purple-900/30 border-purple-500"
                                : "bg-zinc-800/50 border-zinc-700 hover:border-zinc-500"
                            }
            `}
                        onClick={() => selectFace(speaker.faceId)}
                    >
                        {/* Color indicator */}
                        <div
                            className="w-3 h-3 rounded-full flex-shrink-0"
                            style={{ backgroundColor: speaker.color }}
                        />

                        {/* Speaker icon */}
                        <User className={`w-5 h-5 ${isActive ? "text-green-400" : "text-zinc-400"}`} />

                        {/* Editable label */}
                        <input
                            type="text"
                            value={speaker.label}
                            onChange={(e) => setSpeakerLabel(speaker.id, e.target.value)}
                            onClick={(e) => e.stopPropagation()}
                            className="flex-1 bg-transparent border-none outline-none text-sm text-zinc-200 
                         focus:ring-1 focus:ring-purple-500 rounded px-1"
                            placeholder="Speaker name"
                        />

                        {/* Active speaker indicator */}
                        {isActive && (
                            <div className="flex items-center gap-1 text-green-400 text-xs">
                                <div className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
                                Speaking
                            </div>
                        )}
                    </div>
                );
            })}
        </div>
    );
}
