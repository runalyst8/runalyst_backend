import { Stack } from "expo-router";

export default function RootLayout() {
  return <Stack>
    <Stack.Screen name="(videoUpload)" options={{headerShown: false}}/>

  </Stack>
}
