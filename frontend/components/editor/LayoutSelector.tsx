"use client";

import React from "react";
import { useEditorStore, LayoutPreset } from "@/lib/editorStore";
import { Square, RectangleVertical, RectangleHorizontal, Monitor, Maximize } from "lucide-react";

interface LayoutSelectorProps {
    className?: string;
}

const LAYOUT_OPTIONS: { preset: LayoutPreset; label: string; icon: React.ReactNode }[] = [
    { preset: "custom", label: "Custom", icon: <Maximize className="w-4 h-4" /> },
    { preset: "original", label: "Original", icon: <Monitor className="w-4 h-4" /> },
    { preset: "9:16", label: "9:16", icon: <RectangleVertical className="w-4 h-4" /> },
    { preset: "1:1", label: "1:1", icon: <Square className="w-4 h-4" /> },
    { preset: "16:9", label: "16:9", icon: <RectangleHorizontal className="w-4 h-4" /> },
    { preset: "4:3", label: "4:3", icon: <RectangleHorizontal className="w-4 h-4" /> },
    { preset: "9:8", label: "9:8", icon: <Square className="w-4 h-4" /> },
];

export function LayoutSelector({ className }: LayoutSelectorProps) {
    const { layoutPreset, setLayoutPreset } = useEditorStore();

    return (
        <div className={`flex flex-wrap items-center gap-2 ${className}`}>
            {LAYOUT_OPTIONS.map(({ preset, label, icon }) => (
                <button
                    key={preset}
                    onClick={() => setLayoutPreset(preset)}
                    className={`
            flex items-center gap-2 px-3 py-2 rounded-lg border transition-all
            ${layoutPreset === preset
                            ? "bg-purple-600 border-purple-500 text-white"
                            : "bg-zinc-800 border-zinc-700 text-zinc-300 hover:border-zinc-500"
                        }
          `}
                >
                    {icon}
                    <span className="text-sm font-medium">{label}</span>
                </button>
            ))}
        </div>
    );
}
