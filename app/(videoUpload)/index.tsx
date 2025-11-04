import Button from '@/components/VideoUploadButton';
import VideoPlayer from '@/components/VideoPlayer';
import * as ImagePicker from 'expo-image-picker';
import { useMemo, useState } from 'react';
import { Alert, StyleSheet, View } from 'react-native';


const UPLOAD_URL = 'http://localhost:3000/upload';


export default function Index() {
  const [selectedVideo, setSelectedVideo] = useState<string | undefined>(undefined);

  const pickVideoAsync = async () => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: 'videos',
      allowsEditing: false,
      quality: 1,
    });

    if (!result.canceled) {
      setSelectedVideo(result.assets?.[0]?.uri);
    } else {
      Alert.alert('No video selected');
    }
  };


  const fileInfo = useMemo(() => {
    if (!selectedVideo) return null;
    const uri = selectedVideo;
    const name = (uri.split('/').pop() || 'video.mp4').toLowerCase();
    const type =
      name.endsWith('.mov') ? 'video/quicktime' :
      name.endsWith('.webm') ? 'video/webm' :
      name.endsWith('.mkv') ? 'video/x-matroska' :
      'video/mp4';
    return { uri, name, type };
  }, [selectedVideo]);

  const uploadVideoAsync = async () => {
    if (!fileInfo) {
      Alert.alert('Pick a video first');
      return;
    }

    try {
      const form = new FormData();
      // @ts-ignore React Native file shape
      form.append('video', {
        uri: fileInfo.uri,
        name: fileInfo.name,
        type: fileInfo.type,
      });

      const res = await fetch(UPLOAD_URL, {
        method: 'POST',
        // headers: { Authorization: 'Bearer <token>' }, // add if your API needs it
        body: form, // let RN set the multipart boundary; don't set Content-Type
      });

      if (!res.ok) {
        const text = await res.text().catch(() => '');
        throw new Error(`Upload failed (${res.status}): ${text}`);
      }

      Alert.alert('Success', 'Video uploaded successfully!');
    } catch (err: any) {
      Alert.alert('Upload error', err?.message ?? 'Unknown error');
    }
  };

  return (
    <View style={styles.container}>
      <View style={styles.viewerContainer}>
        <VideoPlayer uri={selectedVideo} />
      </View>

      <View style={styles.footerContainer}>
        <Button theme="video-select" label="Choose a video" onPress={pickVideoAsync} />
        <Button theme="video-upload" label="Upload the video" onPress={uploadVideoAsync} />
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#25292e', alignItems: 'center' },
  viewerContainer: { flex: 1, justifyContent: 'center' },
  footerContainer: { flex: 1 / 3, alignItems: 'center' },
});
