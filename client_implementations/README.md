# Resilient WebSocket Client Implementations

This directory contains complete client-side implementations for resilient WebSocket connections that work seamlessly with the aia-v4 server's reattachment grace period system.

## Overview

These clients provide automatic reconnection, audio buffering, and network change detection to ensure uninterrupted audio recording even during network outages, WiFi transitions, and temporary connectivity issues.

## Key Features

### 🔄 **Automatic Reconnection**
- Exponential backoff strategy (1s, 2s, 4s, 8s, 16s, 30s max)
- Immediate retry on network restoration
- Configurable maximum reconnection attempts
- Session reattachment within server's 120-second grace period

### 📡 **Network Change Detection**
- **Web**: Online/offline events, connection type monitoring
- **React Native**: NetInfo integration for cellular/WiFi transitions
- **Flutter**: Connectivity+ plugin for comprehensive network monitoring

### 🎵 **Audio Continuity**
- Automatic audio buffering during disconnections
- Timestamp preservation for perfect server-side ordering
- Seamless resume after reconnection
- Format optimization (WebM for web, MP4 for mobile)

### ❤️ **Connection Health**
- Heartbeat/ping-pong mechanism
- Connection timeout handling
- Health checks on network type changes
- Server status message processing

## Implementations

| Platform | File | Dependencies |
|----------|------|--------------|
| **Web/JavaScript** | `web_resilient_websocket.js` | Native WebSocket, MediaRecorder |
| **React Native** | `react_native_resilient_websocket.js` | `@react-native-netinfo/netinfo` |
| **Flutter** | `flutter_resilient_websocket.dart` | `connectivity_plus`, `record`, `web_socket_channel` |

## Quick Start

### Web
```javascript
const ws = new ResilientWebSocket('https://your-server.com', 'auth-token');
ws.on('open', () => console.log('Connected'));
await ws.connect('agent-name', 'user-id');
await ws.startAudioCapture();
```

### React Native
```javascript
import ResilientWebSocketRN from './react_native_resilient_websocket';
const ws = new ResilientWebSocketRN('https://your-server.com', 'auth-token');
await ws.connect('agent-name', 'user-id');
```

### Flutter
```dart
final ws = ResilientWebSocketFlutter('https://your-server.com', 'auth-token');
await ws.connect('agent-name', 'user-id');
```

## Configuration

All clients support these options:

```javascript
{
    maxReconnectAttempts: 10,        // Max reconnection attempts
    initialReconnectDelay: 1000,     // Initial delay (ms)
    maxReconnectDelay: 30000,        // Max delay (ms)
    graceWindowMs: 120000,           // Server grace period (ms)
    heartbeatIntervalMs: 30000,      // Heartbeat interval (ms)
    connectionTimeoutMs: 10000,      // Connection timeout (ms)
    audioFormat: 'audio/webm'        // Audio format
}
```

## Network Scenarios Handled

### ✅ **WiFi to Cellular Transition**
```
WiFi disconnects → Client detects change → Immediate reconnection → Session resumes
```

### ✅ **Complete Network Loss**
```
Network lost → Audio buffered locally → Network restored → Reconnection → Buffered audio sent
```

### ✅ **Intermittent Connectivity**
```
Connection drops → Exponential backoff → Multiple retry attempts → Connection restored
```

### ✅ **Airplane Mode**
```
Airplane mode ON → Network lost detected → Airplane mode OFF → Immediate reconnection
```

## Server Compatibility

These clients are designed for the aia-v4 server's resilient recording system:

- ✅ **120-second reattachment grace period**
- ✅ **Session ID preservation and reuse**
- ✅ **Ring buffer audio delivery**
- ✅ **Perfect ordering system compatibility**
- ✅ **Hallucination detection integration**
- ✅ **Provider fallback (Deepgram/Whisper) support**

## Event System

All clients provide consistent events:

```javascript
// Connection events
ws.on('open', () => {});           // Initial connection
ws.on('reconnected', () => {});    // Successful reconnection
ws.on('reconnecting', (attempt) => {}); // Reconnection attempt
ws.on('close', () => {});          // Connection closed
ws.on('error', (error) => {});     // Connection error

// Status events
ws.on('statusChanged', (newState, oldState) => {}); // State changes
ws.on('connectionFailed', () => {}); // Permanent failure

// Data events
ws.on('message', (event) => {});   // Server messages
```

## Best Practices

### 1. **User Feedback**
Show connection status and reconnection attempts in your UI:
```javascript
ws.on('reconnecting', (attempt) => {
    showStatus(`Reconnecting... (${attempt}/10)`);
});
```

### 2. **Handle Failures Gracefully**
```javascript
ws.on('connectionFailed', () => {
    showRetryButton('Connection failed. Tap to retry.');
});
```

### 3. **Monitor Audio Buffering**
```javascript
const status = ws.getStatus();
if (status.bufferedAudioChunks > 0) {
    showStatus(`Sending ${status.bufferedAudioChunks} buffered chunks...`);
}
```

### 4. **Clean Up Resources**
```javascript
// Web/React Native
ws.close();

// React Native (additional)
ws.cleanup();

// Flutter
ws.dispose();
```

## Testing Network Scenarios

### Mobile Testing Checklist
- [ ] WiFi to cellular transition
- [ ] Cellular to WiFi transition
- [ ] Airplane mode on/off
- [ ] Moving between WiFi networks
- [ ] Weak signal areas
- [ ] Network timeout scenarios
- [ ] Server restarts during recording

### Web Testing Checklist
- [ ] Browser network tab disable/enable
- [ ] Router disconnect/reconnect
- [ ] VPN on/off transitions
- [ ] Browser tab focus/blur
- [ ] Page refresh during recording
- [ ] Network type changes (if available)

## Files Structure

```
client_implementations/
├── README.md                           # This file
├── usage_examples.md                   # Detailed usage examples
├── web_resilient_websocket.js          # Web/JavaScript implementation
├── react_native_resilient_websocket.js # React Native implementation
└── flutter_resilient_websocket.dart    # Flutter/Dart implementation
```

## Dependencies

### Web
- Native browser APIs (WebSocket, MediaRecorder, Navigator)
- No external dependencies

### React Native
```bash
npm install @react-native-netinfo/netinfo
# For audio recording (example):
npm install react-native-audio-record
```

### Flutter
```yaml
dependencies:
  connectivity_plus: ^4.0.0
  record: ^4.4.4
  web_socket_channel: ^2.4.0
  http: ^1.1.0
```

## Contributing

When contributing to these implementations:

1. Maintain API consistency across platforms
2. Test all network transition scenarios
3. Ensure server compatibility
4. Update usage examples
5. Add platform-specific optimizations where appropriate

## Support

For issues with:
- Server integration → Check aia-v4 server logs
- Network detection → Verify platform-specific permissions
- Audio capture → Check microphone permissions
- WebSocket errors → Verify server URL and authentication

The clients will automatically handle most network-related issues, but proper error handling in your application is essential for the best user experience.