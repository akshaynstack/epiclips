"use client";

import React from "react";
import { useEditorStore } from "@/lib/editorStore";
import {
    Sparkles,
    Captions,
    Type,
    Upload,
    Layers,
    Wand2,
    Film,
    Music,
    RotateCcw
} from "lucide-react";

interface ToolsSidebarProps {
    className?: string;
}

const TOOLS = [
    { id: "ai-enhance", label: "AI enhance", icon: Sparkles, disabled: true },
    { id: "caption", label: "Caption", icon: Captions, disabled: false },
    { id: "text", label: "Text", icon: Type, disabled: true },
    { id: "upload", label: "Upload", icon: Upload, disabled: true },
    { id: "transitions", label: "Transitions", icon: Layers, disabled: true },
    { id: "ai-hook", label: "AI hook", icon: Wand2, disabled: true },
    { id: "b-roll", label: "B-Roll", icon: Film, disabled: true },
    { id: "music", label: "Music", icon: Music, disabled: true },
];

export function ToolsSidebar({ className }: ToolsSidebarProps) {
    const { isDirty, isLoading, applyChanges, resetChanges } = useEditorStore();

    return (
        <div className={`flex flex-col gap-2 p-2 bg-zinc-900 rounded-lg ${className}`}>
            {/* Tool buttons */}
            {TOOLS.map(({ id, label, icon: Icon, disabled }) => (
                <button
                    key={id}
                    disabled={disabled}
                    className={`
            flex flex-col items-center gap-1 p-3 rounded-lg transition-all
            ${disabled
                            ? "opacity-40 cursor-not-allowed"
                            : "hover:bg-zinc-800 cursor-pointer"
                        }
          `}
                >
                    <Icon className="w-5 h-5 text-zinc-300" />
                    <span className="text-[10px] text-zinc-400">{label}</span>
                </button>
            ))}

            {/* Spacer */}
            <div className="flex-1" />

            {/* Action buttons */}
            {isDirty && (
                <button
                    onClick={resetChanges}
                    className="flex flex-col items-center gap-1 p-3 rounded-lg hover:bg-zinc-800 transition-all"
                >
                    <RotateCcw className="w-5 h-5 text-zinc-300" />
                    <span className="text-[10px] text-zinc-400">Reset</span>
                </button>
            )}
        </div>
    );
}
