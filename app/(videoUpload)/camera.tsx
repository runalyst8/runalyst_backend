import { Camera, CameraView } from "expo-camera";
import { useEffect, useRef, useState } from "react";
import { ActivityIndicator, StyleSheet, Text, TouchableOpacity, View } from "react-native";
import { runPreflightCheck } from "./preflightCheck";

export default function CameraScreen() {
  const cameraRef = useRef<CameraView | null>(null);
  const checkIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const initialCheckDoneRef = useRef(false);
  const lastPassRef = useRef<boolean | null>(null);

  const [hasPermission, setHasPermission] = useState<boolean | null>(null);
  const [status, setStatus] = useState<"loading" | "checking" | "ready" | "fail">("loading");
  const [metrics, setMetrics] = useState<any | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [mode, setMode] = useState<"day" | "night">("day");

  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === "granted");
      if (status === "granted") {
        startQualityChecks();
      } else {
        setStatus("fail");
      }
    })();

    return () => {
      if (checkIntervalRef.current) {
        clearInterval(checkIntervalRef.current);
        checkIntervalRef.current = null;
      }
    };
  }, []);

  const startQualityChecks = () => {
    const runCheck = async () => {
      if (!cameraRef.current || isRecording) return;
      
      try {
        if (!initialCheckDoneRef.current) setStatus("checking");

        const thresholds =
          mode === "day"
            ? { minBrightness: 60, maxBrightness: 220, minSharpness: 0.035 }
            : { minBrightness: 25, maxBrightness: 200, minSharpness: 0.02 };

        const result = await runPreflightCheck(cameraRef.current, thresholds);
        console.log("Preflight results:", result);
        setMetrics(result.metrics);

        if (!initialCheckDoneRef.current) {
          setStatus(result.pass ? "ready" : "fail");
        } else {
          const prev = lastPassRef.current;
          if (prev === null) {
            setStatus(result.pass ? "ready" : "fail");
          } else if (prev === true && result.pass === false) {
            setStatus("fail");
          } else if (prev === false && result.pass === true) {
            setStatus("ready");
          }
        }

        lastPassRef.current = result.pass;
        initialCheckDoneRef.current = true;
      } catch (err) {
        console.error("Error during preflight:", err);
        setStatus("fail");
      }
    };

    // Initial check after a short delay to let camera initialize
    setTimeout(runCheck, 500);

    // Poll every 1.5s
    if (!checkIntervalRef.current) {
      checkIntervalRef.current = setInterval(() => {
        if (!isRecording) runCheck();
      }, 1500);
    }
  };

  useEffect(() => {
    if (hasPermission && !isRecording) {
      // Restart checks when mode changes
      if (checkIntervalRef.current) {
        clearInterval(checkIntervalRef.current);
        checkIntervalRef.current = null;
      }
      initialCheckDoneRef.current = false;
      lastPassRef.current = null;
      startQualityChecks();
    }
  }, [mode]);

  function handleStartRecording() {
    if (status !== "ready") return;
    
    if (checkIntervalRef.current) {
      clearInterval(checkIntervalRef.current);
      checkIntervalRef.current = null;
    }
    
    setIsRecording(true);
    console.log("Recording started");
    
    // TODO: Implement actual recording with cameraRef.current.recordAsync()
  }

  function getAdvice(metrics: any | null) {
    if (!metrics) return "Try moving to a brighter area or clean your camera lens.";
    const { avgBrightness, avgSharpness } = metrics;
    const advices: string[] = [];

    if (typeof avgBrightness === "number") {
      if (avgBrightness < 50) {
        advices.push("The scene looks dark – move to a brighter area, add a front-facing light, or face a window.");
      } else if (avgBrightness < 90) {
        advices.push("A little dim – try adding more light in front of you or move closer to a light source.");
      } else if (avgBrightness > 220) {
        advices.push("The scene is very bright – avoid strong backlight (don't sit with a window behind you).");
      }
    }

    if (typeof avgSharpness === "number") {
      if (avgSharpness < 0.02) {
        advices.push("Image looks blurry – keep the camera steady, clean the lens, and avoid digital zoom.");
      } else if (avgSharpness < 0.04) {
        advices.push("Slight blur – try steadying your device or bringing it a bit closer to the subject.");
      }
    }

    if (advices.length === 0) return "Preview looks good – you're ready to record.";
    return advices.join(" ");
  }

  if (hasPermission === null) {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color="#2b8a3e" />
        <Text style={styles.loadingText}>Requesting camera permission...</Text>
      </View>
    );
  }

  if (hasPermission === false) {
    return (
      <View style={styles.container}>
        <Text style={styles.errorText}>No access to camera</Text>
        <Text style={styles.helpText}>Please grant camera permission in settings</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <CameraView
        ref={cameraRef}
        style={styles.camera}
        facing="back"
      />

      <TouchableOpacity
        onPress={handleStartRecording}
        disabled={status !== "ready"}
        style={[
          styles.recordButton,
          { backgroundColor: status === "ready" ? "#2b8a3e" : "#999" }
        ]}
      >
        <Text style={styles.recordButtonText}>
          {isRecording ? "Recording..." : "Start Recording"}
        </Text>
      </TouchableOpacity>

      <View style={styles.modeContainer}>
        <Text style={styles.modeLabel}>Mode:</Text>
        <TouchableOpacity
          onPress={() => setMode("day")}
          style={[
            styles.modeButton,
            { backgroundColor: mode === "day" ? "#2b8a3e" : "#ddd" }
          ]}
        >
          <Text style={[styles.modeButtonText, { color: mode === "day" ? "white" : "black" }]}>
            Day
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          onPress={() => setMode("night")}
          style={[
            styles.modeButton,
            { backgroundColor: mode === "night" ? "#5c5cff" : "#ddd" }
          ]}
        >
          <Text style={[styles.modeButtonText, { color: mode === "night" ? "white" : "black" }]}>
            Night
          </Text>
        </TouchableOpacity>
      </View>

      <View style={styles.statusContainer}>
        <Text style={styles.statusText}>
          {status === "loading" && "Loading camera..."}
          {status === "checking" && "Analyzing camera quality..."}
          {status === "ready" && "✅ Ready to record"}
          {status === "fail" && "❌ Camera quality too low"}
        </Text>
      </View>

      <View style={styles.adviceContainer}>
        <Text style={styles.adviceTitle}>How to improve:</Text>
        <Text style={styles.adviceText}>{getAdvice(metrics)}</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#fff",
    alignItems: "center",
    justifyContent: "center",
    padding: 20,
  },
  camera: {
    width: "90%",
    aspectRatio: 3 / 4,
    borderRadius: 16,
    overflow: "hidden",
    maxWidth: 400,
  },
  recordButton: {
    marginTop: 16,
    paddingVertical: 12,
    paddingHorizontal: 24,
    borderRadius: 8,
  },
  recordButtonText: {
    color: "white",
    fontSize: 16,
    fontWeight: "600",
  },
  modeContainer: {
    flexDirection: "row",
    alignItems: "center",
    marginTop: 16,
  },
  modeLabel: {
    marginRight: 8,
    fontSize: 16,
  },
  modeButton: {
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 6,
    marginHorizontal: 4,
  },
  modeButtonText: {
    fontSize: 14,
    fontWeight: "500",
  },
  statusContainer: {
    marginTop: 12,
  },
  statusText: {
    fontSize: 16,
    textAlign: "center",
  },
  adviceContainer: {
    maxWidth: 420,
    marginTop: 12,
    backgroundColor: "#fffbe6",
    padding: 12,
    borderRadius: 8,
  },
  adviceTitle: {
    fontWeight: "bold",
    fontSize: 15,
    marginBottom: 6,
  },
  adviceText: {
    fontSize: 14,
    lineHeight: 20,
  },
  loadingText: {
    marginTop: 12,
    fontSize: 16,
  },
  errorText: {
    fontSize: 18,
    fontWeight: "600",
    color: "#d32f2f",
  },
  helpText: {
    marginTop: 8,
    fontSize: 14,
    color: "#666",
  },
});