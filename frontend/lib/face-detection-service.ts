import {
    FaceDetector,
    FilesetResolver,
    Detection
} from "@mediapipe/tasks-vision";

let faceDetector: FaceDetector | null = null;

export async function getFaceDetector() {
    if (faceDetector) return faceDetector;

    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
    );

    faceDetector = await FaceDetector.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
            delegate: "GPU" // Forces WebGPU/WebGL
        },
        runningMode: "IMAGE"
    });

    return faceDetector;
}

export interface FaceResult {
    bbox: { x: number; y: number; w: number; h: number };
    score: number;
}

/**
 * Detect faces in a single frame
 */
export async function detectFacesInFrame(frameUrl: string): Promise<FaceResult[]> {
    const detector = await getFaceDetector();

    // Create an image element to process
    const img = new Image();
    img.src = frameUrl;
    await new Promise((resolve) => (img.onload = resolve));

    const result = detector.detect(img);

    return result.detections.map((d: Detection) => {
        const box = d.boundingBox!;
        return {
            bbox: {
                x: box.originX,
                y: box.originY,
                w: box.width,
                h: box.height
            },
            score: d.categories[0].score
        };
    });
}

/**
 * Analyze a sequence of frames to determine the best layout
 */
export async function analyzeLayout(frameUrls: string[], frameWidth: number, frameHeight: number) {
    const allDetections: FaceResult[][] = [];
    const framesWithCornerFacecam: number[] = [];
    const cornerFacecamBboxes: { x: number, y: number, w: number, h: number }[] = [];

    // Configuration for corner detection (25% of width/height from edges)
    const cornerXMax = frameWidth * 0.25;
    const cornerXMin = frameWidth * 0.75;
    const cornerYMax = frameHeight * 0.25;
    const cornerYMin = frameHeight * 0.75;

    // Area ratio bounds - INCREASED max to 10% to catch larger webcam overlays
    const minAreaRatio = 0.003;  // 0.3% minimum
    const maxAreaRatio = 0.10;   // 10% maximum (was 4%)
    const frameArea = frameWidth * frameHeight;

    console.log(`[Layout] Analyzing ${frameUrls.length} frames at ${frameWidth}x${frameHeight}`);
    console.log(`[Layout] Corner thresholds: X=[0-${cornerXMax}|${cornerXMin}-${frameWidth}], Y=[0-${cornerYMax}|${cornerYMin}-${frameHeight}]`);
    console.log(`[Layout] Area ratio range: ${(minAreaRatio * 100).toFixed(1)}% - ${(maxAreaRatio * 100).toFixed(1)}%`);

    for (let i = 0; i < frameUrls.length; i++) {
        const faces = await detectFacesInFrame(frameUrls[i]);
        allDetections.push(faces);

        console.log(`[Layout] Frame ${i + 1}: Detected ${faces.length} faces`);

        // Check for corner facecam
        let foundInCorner = false;
        for (const face of faces) {
            const { x, y, w, h } = face.bbox;
            const areaRatio = (w * h) / frameArea;

            const faceCenterX = x + w / 2;
            const faceCenterY = y + h / 2;

            const isInLeftCorner = faceCenterX < cornerXMax;
            const isInRightCorner = faceCenterX > cornerXMin;
            const isInTopCorner = faceCenterY < cornerYMax;
            const isInBottomCorner = faceCenterY > cornerYMin;

            const isInHorizontalCorner = isInLeftCorner || isInRightCorner;
            const isInVerticalCorner = isInTopCorner || isInBottomCorner;
            const hasValidSize = areaRatio >= minAreaRatio && areaRatio <= maxAreaRatio;

            console.log(`[Layout]   Face: pos=(${x.toFixed(0)},${y.toFixed(0)}) size=${w.toFixed(0)}x${h.toFixed(0)} area=${(areaRatio * 100).toFixed(2)}%`);
            console.log(`[Layout]   Center=(${faceCenterX.toFixed(0)},${faceCenterY.toFixed(0)}) inHCorner=${isInHorizontalCorner} inVCorner=${isInVerticalCorner} validSize=${hasValidSize}`);

            if (isInHorizontalCorner && isInVerticalCorner && hasValidSize) {
                framesWithCornerFacecam.push(i);
                cornerFacecamBboxes.push(face.bbox);
                foundInCorner = true;
                console.log(`[Layout]   ✓ CORNER FACECAM DETECTED!`);
                break;
            } else {
                console.log(`[Layout]   ✗ Not a corner facecam`);
            }
        }
    }

    // Determine dominant layout
    // If corner facecam is detected in > 40% of sampled frames, it's a screen_share
    const cornerRatio = framesWithCornerFacecam.length / frameUrls.length;
    const dominantLayout = cornerRatio > 0.4 ? "screen_share" : "talking_head";

    console.log(`[Layout] Result: ${framesWithCornerFacecam.length}/${frameUrls.length} frames with corner facecam (${(cornerRatio * 100).toFixed(0)}%)`);
    console.log(`[Layout] Dominant layout: ${dominantLayout}`);

    // Calculate average corner bbox if applicable
    let avgCornerBbox = null;
    if (cornerFacecamBboxes.length > 0) {
        avgCornerBbox = {
            x: Math.round(cornerFacecamBboxes.reduce((sum, b) => sum + b.x, 0) / cornerFacecamBboxes.length),
            y: Math.round(cornerFacecamBboxes.reduce((sum, b) => sum + b.y, 0) / cornerFacecamBboxes.length),
            w: Math.round(cornerFacecamBboxes.reduce((sum, b) => sum + b.w, 0) / cornerFacecamBboxes.length),
            h: Math.round(cornerFacecamBboxes.reduce((sum, b) => sum + b.h, 0) / cornerFacecamBboxes.length),
        };
    }

    return {
        dominantLayout,
        avgCornerBbox,
        cornerRatio,
        detections: allDetections,
    };
}
