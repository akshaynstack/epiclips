"use client";

import { useEffect, useCallback, useState } from "react";
import { useEditorStore, Face } from "@/lib/editorStore";

interface FaceDetectionResult {
    faces: Face[];
    layout_type: string;
    confidence: number;
}

export function useFaceDetection(videoFile: File | null) {
    const { setFaces } = useEditorStore();
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const detectFaces = useCallback(async () => {
        if (!videoFile) return;

        setIsLoading(true);
        setError(null);

        try {
            const formData = new FormData();
            formData.append("video", videoFile);

            const response = await fetch("http://localhost:8000/ai-clipping/detect-layout-direct", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                throw new Error("Face detection failed");
            }

            const data = await response.json();

            // Convert backend response to Face format
            const faces: Face[] = [];

            // If there's a main facecam bbox, add it as face 1
            if (data.facecam_bbox) {
                faces.push({
                    id: "face_1",
                    label: "Speaker 1",
                    bbox: {
                        x: data.facecam_bbox[0],
                        y: data.facecam_bbox[1],
                        width: data.facecam_bbox[2] - data.facecam_bbox[0],
                        height: data.facecam_bbox[3] - data.facecam_bbox[1],
                    },
                    trackId: 1,
                    isActiveSpeaker: true,
                    confidence: data.confidence,
                });
            }

            // If there are podcast bboxes, add them
            if (data.podcast_face_bboxes && data.podcast_face_bboxes.length > 0) {
                data.podcast_face_bboxes.forEach((bbox: number[], i: number) => {
                    faces.push({
                        id: `face_${i + 1}`,
                        label: `Speaker ${i + 1}`,
                        bbox: {
                            x: bbox[0],
                            y: bbox[1],
                            width: bbox[2] - bbox[0],
                            height: bbox[3] - bbox[1],
                        },
                        trackId: i + 1,
                        isActiveSpeaker: i === 0,
                        confidence: data.confidence,
                    });
                });
            }

            // If no faces found, create a default center crop
            if (faces.length === 0) {
                faces.push({
                    id: "face_1",
                    label: "Speaker 1",
                    bbox: { x: 400, y: 200, width: 400, height: 400 },
                    trackId: 1,
                    isActiveSpeaker: true,
                    confidence: 0.5,
                });
            }

            setFaces(faces);
        } catch (err: any) {
            setError(err.message);
        } finally {
            setIsLoading(false);
        }
    }, [videoFile, setFaces]);

    return { detectFaces, isLoading, error };
}
