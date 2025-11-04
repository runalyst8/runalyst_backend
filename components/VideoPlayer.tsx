import { ResizeMode, Video } from 'expo-av';
import { StyleSheet } from 'react-native';

type Props = {
  uri?: string;
};

export default function VideoPlayer({ uri }: Props) {
  if (!uri) return null;

  return (
    <Video
      source={{ uri }}
      style={styles.video}
      useNativeControls
      resizeMode={ResizeMode.CONTAIN}
      isLooping
    />
  );
}

const styles = StyleSheet.create({
  video: {
    width: 320,
    height: 440,
    borderRadius: 18,
  },
});
