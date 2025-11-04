import { Buffer } from "buffer";
import { CameraView } from "expo-camera";
import * as FileSystem from "expo-file-system";
import { Platform } from "react-native";

export async function runPreflightCheck(
  camera: CameraView,
  options?: {
    minBrightness?: number;
    maxBrightness?: number;
    minSharpness?: number;
    framesToSample?: number;
    sampleIntervalMs?: number;
  }
) {
  const {
    minBrightness = 40,
    maxBrightness = 200,
    minSharpness = 0.03,
    framesToSample = 5,
    sampleIntervalMs = 300,
  } = options || {};

  const brightnessValues: number[] = [];
  const sharpnessValues: number[] = [];

  for (let i = 0; i < framesToSample; i++) {
    try {
      const photo = await camera.takePictureAsync({
        quality: 0.1,
        base64: true, // Need base64 for native processing
        skipProcessing: true,
        exif: true, // Enable EXIF for accurate brightness on native
      });

      if (!photo || !photo.base64) continue;

      // Analyze the image (prefer EXIF brightness on native when available)
      const analysis = await analyzeImage(photo);
      
      if (analysis) {
        brightnessValues.push(analysis.brightness);
        sharpnessValues.push(analysis.sharpness);
      }

      // Clean up if we have a file URI
      if (photo.uri) {
        try {
          await FileSystem.deleteAsync(photo.uri, { idempotent: true });
        } catch (e) {
          // Ignore cleanup errors
        }
      }

      if (i < framesToSample - 1) {
        await new Promise((r) => setTimeout(r, sampleIntervalMs));
      }
    } catch (err) {
      console.warn("Error sampling frame:", err);
    }
  }

  const avgBrightness = average(brightnessValues);
  const avgSharpness = average(sharpnessValues);

  console.log("Sampled values:", { 
    brightnessValues, 
    sharpnessValues, 
    avgBrightness, 
    avgSharpness 
  });

  const pass = 
    brightnessValues.length > 0 &&
    avgBrightness > minBrightness && 
    avgBrightness < maxBrightness && 
    avgSharpness > minSharpness;

  return {
    pass,
    metrics: { avgBrightness, avgSharpness },
    thresholds: { minBrightness, maxBrightness, minSharpness },
  };
}

async function analyzeImage(photo: { base64?: string | null; exif?: any }): Promise<{ brightness: number; sharpness: number } | null> {
  if (Platform.OS === 'web') {
    // Web implementation using Canvas
    if (!photo.base64) return null;
    return analyzeImageWeb(photo.base64);
  } else {
    // Native implementation (iOS/Android) — prefer EXIF when available
    return analyzeImageNative(photo);
  }
}

// Web implementation using Canvas API
function analyzeImageWeb(base64: string): Promise<{ brightness: number; sharpness: number } | null> {
  return new Promise((resolve) => {
    const img = new Image();
    
    img.onload = () => {
      try {
        const canvas = document.createElement("canvas");
        const maxSize = 200;
        
        const scale = Math.min(maxSize / img.width, maxSize / img.height);
        canvas.width = img.width * scale;
        canvas.height = img.height * scale;
        
        const ctx = canvas.getContext("2d");
        if (!ctx) {
          resolve(null);
          return;
        }
        
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imageData.data;
        
        const brightness = getAverageBrightness(data);
        const sharpness = getSharpness(data, canvas.width);
        
        resolve({ brightness, sharpness });
      } catch (err) {
        console.error("Error analyzing image:", err);
        resolve(null);
      }
    };
    
    img.onerror = () => {
      console.error("Error loading image for analysis");
      resolve(null);
    };
    
    img.src = toDataUrl(base64);
  });
}

// Native implementation - uses EXIF BrightnessValue when available; falls back to JPEG byte sampling
async function analyzeImageNative(photo: { base64?: string | null; exif?: any }): Promise<{ brightness: number; sharpness: number } | null> {
  try {
    // Try to compute brightness from EXIF first (more accurate and faster)
    const brightnessFromExif = getBrightnessFromExif(photo.exif);

    // Prepare bytes for sharpness fallback/estimation
    let bytes: Uint8Array | null = null;
    if (photo.base64) {
      const cleanBase64 = photo.base64.replace(/^data:image\/\w+;base64,/, "");
      try {
        const buf = Buffer.from(cleanBase64, "base64");
        bytes = new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
      } catch (e) {
        bytes = null;
      }
    }
    
    // Estimate sharpness from JPEG bytes variance (heuristic)
    let sharpness = 0;
    if (bytes && bytes.length > 0) {
      const sampleSize = Math.min(10000, bytes.length);
      const step = Math.max(1, Math.floor(bytes.length / sampleSize));
      let sum = 0;
      let variance = 0;
      const samples: number[] = [];
      for (let i = 0; i < bytes.length; i += step) {
        const value = bytes[i];
        samples.push(value);
        sum += value;
      }
      const mean = samples.length > 0 ? sum / samples.length : 0;
      for (const value of samples) {
        variance += Math.pow(value - mean, 2);
      }
      variance = samples.length > 0 ? variance / samples.length : 0;
      sharpness = Math.sqrt(variance) / 1000; // Normalize variance
    }
    
    // Determine brightness: EXIF-derived preferred, else heuristic from bytes mean
    let brightness: number;
    if (brightnessFromExif != null) {
      brightness = brightnessFromExif;
    } else if (bytes && bytes.length > 0) {
      const sampleSize = Math.min(10000, bytes.length);
      const step = Math.max(1, Math.floor(bytes.length / sampleSize));
      let sum = 0;
      let count = 0;
      for (let i = 0; i < bytes.length; i += step) {
        sum += bytes[i];
        count++;
      }
      const mean = count > 0 ? sum / count : 0;
      brightness = mean * 0.8; // Scale to approximately match RGB brightness range
    } else {
      return null;
    }

    return { brightness, sharpness };
  } catch (err) {
    console.error("Error in native image analysis:", err);
    return null;
  }
}

// Map EXIF BrightnessValue (in EV) to 0..255 range for easier thresholding
function getBrightnessFromExif(exif: any): number | null {
  if (!exif) return null;
  // Common locations for EXIF brightness across platforms
  const evRaw =
    exif.BrightnessValue ??
    exif["BrightnessValue"] ??
    exif?.Exif?.BrightnessValue ??
    exif?.["{Exif}"]?.BrightnessValue;

  const evNumber = typeof evRaw === "number" ? evRaw : Number(evRaw);
  if (!isFinite(evNumber)) return null;
  // Typical EV ranges roughly from -6 (very dark) to +14 (very bright)
  const evClamped = Math.max(-6, Math.min(14, evNumber));
  const normalized = (evClamped + 6) / 20; // 0..1
  return normalized * 255; // Align with canvas brightness scale
}

function average(arr: number[]) {
  if (!arr || arr.length === 0) return 0;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function getAverageBrightness(data: Uint8ClampedArray): number {
  let sum = 0;
  
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    sum += 0.2126 * r + 0.7152 * g + 0.0722 * b;
  }
  
  const pixelCount = data.length / 4;
  return sum / pixelCount;
}

function getSharpness(data: Uint8ClampedArray, width: number): number {
  let diff = 0;
  const height = data.length / (width * 4);
  
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width - 1; x++) {
      const idx = (y * width + x) * 4;
      const nextIdx = idx + 4;
      
      const gray = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
      const nextGray = (data[nextIdx] + data[nextIdx + 1] + data[nextIdx + 2]) / 3;
      
      diff += Math.abs(gray - nextGray);
    }
  }
  
  return diff / (data.length / 4);
}

// Ensure base64 string is a proper data URL and strip any existing prefix
function toDataUrl(input: string): string {
  if (!input) return "";
  const match = input.match(/^data:(.*?);base64,(.*)$/);
  if (match) {
    // Already a data URL, normalize mime if missing (keep as-is otherwise)
    return `data:${match[1] || "image/jpeg"};base64,${match[2]}`;
  }
  // No prefix; default to JPEG which is what Camera outputs by default
  return `data:image/jpeg;base64,${input}`;
}
