import { Tabs } from 'expo-router';

import Ionicons from '@expo/vector-icons/Ionicons';


export default function TabLayout() {
  return (
    <Tabs
      screenOptions={{
        tabBarActiveTintColor: '#ffd33d',
        headerStyle: {
        backgroundColor: '#25292e',
        },
        headerShadowVisible: false,
        headerTintColor: '#fff',
        tabBarStyle: {
        backgroundColor: '#25292e',
        },
    }}
    >
      <Tabs.Screen
        name="index"
        options={{
          title: 'Upload Video',
          tabBarIcon: ({ color}) => (
            <Ionicons name='videocam-outline' color={color} size={24} />
          ),
        }}
      />
      <Tabs.Screen
        name="camera"
        options={{
          title: 'Capture Video',
          tabBarIcon: ({ color}) => (
            <Ionicons name='cloud-upload-outline' color={color} size={24} />
          ),
        }}
        />
    </Tabs>
  );
}
