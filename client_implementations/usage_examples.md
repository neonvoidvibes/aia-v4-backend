# Resilient WebSocket Client Usage Examples

This document provides usage examples for the resilient WebSocket clients implemented for different platforms.

## Web/JavaScript Usage

```javascript
// Initialize the resilient WebSocket
const resilientWS = new ResilientWebSocket(
    'https://your-server.com', // baseUrl
    'your-auth-token',         // authToken
    {
        maxReconnectAttempts: 15,
        initialReconnectDelay: 500,
        maxReconnectDelay: 30000,
        heartbeatIntervalMs: 25000
    }
);

// Set up event handlers
resilientWS.on('open', () => {
    console.log('WebSocket connected');
});

resilientWS.on('reconnected', () => {
    console.log('WebSocket reconnected successfully');
});

resilientWS.on('reconnecting', (attemptNumber) => {
    console.log(`Reconnecting attempt ${attemptNumber}`);
});

resilientWS.on('message', (event) => {
    console.log('Received message:', event.data);
});

resilientWS.on('statusChanged', (newState, oldState) => {
    console.log(`Connection state changed: ${oldState} → ${newState}`);
    updateUIConnectionStatus(newState);
});

resilientWS.on('connectionFailed', () => {
    console.error('Connection failed permanently');
    showReconnectionFailedUI();
});

// Connect and start audio capture
async function startRecording() {
    try {
        await resilientWS.connect('agent-name', 'user-id');
        await resilientWS.startAudioCapture();
        console.log('Recording started successfully');
    } catch (error) {
        console.error('Failed to start recording:', error);
    }
}

// Stop recording
function stopRecording() {
    resilientWS.stopAudioCapture();
    resilientWS.close();
}

// Send custom message
function sendCustomMessage(data) {
    resilientWS.send({
        type: 'custom',
        payload: data,
        timestamp: Date.now()
    });
}

// Get connection status for UI updates
function updateConnectionInfo() {
    const status = resilientWS.getStatus();
    document.getElementById('connection-status').textContent = status.state;
    document.getElementById('session-id').textContent = status.sessionId || 'None';
    document.getElementById('buffered-chunks').textContent = status.bufferedAudioChunks;
}

// Start the recording when page loads
startRecording();
```

## React Native Usage

```javascript
import ResilientWebSocketRN from './react_native_resilient_websocket';

class AudioRecordingService {
    constructor() {
        this.resilientWS = null;
        this.isRecording = false;
    }

    async initialize(serverUrl, authToken) {
        this.resilientWS = new ResilientWebSocketRN(serverUrl, authToken, {
            maxReconnectAttempts: 12,
            audioFormat: 'audio/mp4', // Mobile format
            heartbeatIntervalMs: 20000
        });

        // Set up event handlers
        this.resilientWS.on('open', this.handleConnected.bind(this));
        this.resilientWS.on('reconnected', this.handleReconnected.bind(this));
        this.resilientWS.on('reconnecting', this.handleReconnecting.bind(this));
        this.resilientWS.on('message', this.handleMessage.bind(this));
        this.resilientWS.on('networkChanged', this.handleNetworkChanged.bind(this));
        this.resilientWS.on('statusChanged', this.handleStatusChanged.bind(this));
        this.resilientWS.on('connectionFailed', this.handleConnectionFailed.bind(this));

        return this.resilientWS;
    }

    async startRecording(agentName, userId) {
        if (!this.resilientWS) {
            throw new Error('WebSocket not initialized');
        }

        try {
            await this.resilientWS.connect(agentName, userId);
            await this.resilientWS.startAudioCapture();
            this.isRecording = true;
            console.log('Recording started successfully');
        } catch (error) {
            console.error('Failed to start recording:', error);
            throw error;
        }
    }

    async stopRecording() {
        if (this.resilientWS) {
            await this.resilientWS.stopAudioCapture();
            this.resilientWS.close();
            this.isRecording = false;
        }
    }

    handleConnected() {
        console.log('WebSocket connected');
        this.updateConnectionStatus('connected');
    }

    handleReconnected() {
        console.log('WebSocket reconnected after network interruption');
        this.updateConnectionStatus('reconnected');
    }

    handleReconnecting(attemptNumber) {
        console.log(`Reconnecting attempt ${attemptNumber}`);
        this.updateConnectionStatus('reconnecting', attemptNumber);
    }

    handleMessage(event) {
        // Handle transcription results or other messages
        console.log('Received message:', event.data);
    }

    handleNetworkChanged(networkState) {
        console.log('Network changed:', networkState);
        if (!networkState.isConnected) {
            this.updateConnectionStatus('network_lost');
        } else {
            this.updateConnectionStatus('network_restored', networkState.type);
        }
    }

    handleStatusChanged(newState, oldState) {
        console.log(`Connection state: ${oldState} → ${newState}`);
        this.updateConnectionStatus(newState);
    }

    handleConnectionFailed() {
        console.error('Connection failed permanently');
        this.updateConnectionStatus('failed');
        // Show user notification about connection failure
    }

    updateConnectionStatus(status, extra = null) {
        // Update your React Native UI here
        // This could emit events to your components or update a state store
        console.log(`Connection status: ${status}`, extra);
    }

    getStatus() {
        return this.resilientWS ? this.resilientWS.getStatus() : null;
    }

    cleanup() {
        if (this.resilientWS) {
            this.resilientWS.cleanup();
        }
    }
}

// Usage in a React Native component
export default function AudioRecorder() {
    const [recordingService] = useState(new AudioRecordingService());
    const [connectionStatus, setConnectionStatus] = useState('disconnected');
    const [isRecording, setIsRecording] = useState(false);

    useEffect(() => {
        // Initialize service
        recordingService.initialize('https://your-server.com', 'your-auth-token');

        // Cleanup on unmount
        return () => {
            recordingService.cleanup();
        };
    }, []);

    const startRecording = async () => {
        try {
            await recordingService.startRecording('agent-name', 'user-id');
            setIsRecording(true);
        } catch (error) {
            Alert.alert('Error', 'Failed to start recording');
        }
    };

    const stopRecording = async () => {
        await recordingService.stopRecording();
        setIsRecording(false);
    };

    return (
        <View style={styles.container}>
            <Text>Connection: {connectionStatus}</Text>
            <TouchableOpacity
                onPress={isRecording ? stopRecording : startRecording}
                style={[styles.button, isRecording ? styles.stopButton : styles.startButton]}
            >
                <Text>{isRecording ? 'Stop Recording' : 'Start Recording'}</Text>
            </TouchableOpacity>
        </View>
    );
}
```

## Flutter Usage

```dart
import 'flutter_resilient_websocket.dart';

class AudioRecordingService {
  late ResilientWebSocketFlutter _resilientWS;
  bool _isInitialized = false;
  bool _isRecording = false;

  // Stream controllers for UI updates
  final StreamController<String> _connectionStatusController = StreamController.broadcast();
  final StreamController<String> _messageController = StreamController.broadcast();

  Stream<String> get connectionStatusStream => _connectionStatusController.stream;
  Stream<String> get messageStream => _messageController.stream;

  Future<void> initialize(String serverUrl, String authToken) async {
    _resilientWS = ResilientWebSocketFlutter(serverUrl, authToken, {
      'maxReconnectAttempts': 12,
      'audioFormat': 'audio/mp4',
      'heartbeatIntervalMs': 20000,
    });

    // Set up event listeners
    _resilientWS.onOpen.listen((_) => _handleConnected());
    _resilientWS.onReconnected.listen((_) => _handleReconnected());
    _resilientWS.onReconnecting.listen((attemptNumber) => _handleReconnecting(attemptNumber));
    _resilientWS.onMessage.listen(_handleMessage);
    _resilientWS.onStatusChanged.listen(_handleStatusChanged);
    _resilientWS.onConnectionFailed.listen((_) => _handleConnectionFailed());

    _isInitialized = true;
  }

  Future<void> startRecording(String agentName, String userId) async {
    if (!_isInitialized) {
      throw Exception('Service not initialized');
    }

    try {
      await _resilientWS.connect(agentName, userId);
      await _resilientWS.startAudioCapture();
      _isRecording = true;
      print('Recording started successfully');
    } catch (error) {
      print('Failed to start recording: $error');
      rethrow;
    }
  }

  Future<void> stopRecording() async {
    if (_isInitialized) {
      await _resilientWS.stopAudioCapture();
      _resilientWS.close();
      _isRecording = false;
    }
  }

  void _handleConnected() {
    print('WebSocket connected');
    _connectionStatusController.add('connected');
  }

  void _handleReconnected() {
    print('WebSocket reconnected after network interruption');
    _connectionStatusController.add('reconnected');
  }

  void _handleReconnecting(int attemptNumber) {
    print('Reconnecting attempt $attemptNumber');
    _connectionStatusController.add('reconnecting_$attemptNumber');
  }

  void _handleMessage(dynamic message) {
    print('Received message: $message');
    _messageController.add(message.toString());
  }

  void _handleStatusChanged(ConnectionState newState) {
    print('Connection state changed to: $newState');
    _connectionStatusController.add(newState.toString());
  }

  void _handleConnectionFailed() {
    print('Connection failed permanently');
    _connectionStatusController.add('failed');
  }

  Map<String, dynamic>? getStatus() {
    return _isInitialized ? _resilientWS.getStatus() : null;
  }

  void dispose() {
    if (_isInitialized) {
      _resilientWS.dispose();
    }
    _connectionStatusController.close();
    _messageController.close();
  }
}

// Usage in a Flutter widget
class AudioRecorderWidget extends StatefulWidget {
  @override
  _AudioRecorderWidgetState createState() => _AudioRecorderWidgetState();
}

class _AudioRecorderWidgetState extends State<AudioRecorderWidget> {
  final AudioRecordingService _recordingService = AudioRecordingService();
  String _connectionStatus = 'disconnected';
  bool _isRecording = false;
  List<String> _messages = [];

  @override
  void initState() {
    super.initState();
    _initializeService();
  }

  Future<void> _initializeService() async {
    try {
      await _recordingService.initialize('https://your-server.com', 'your-auth-token');

      // Listen to connection status changes
      _recordingService.connectionStatusStream.listen((status) {
        setState(() {
          _connectionStatus = status;
        });
      });

      // Listen to messages
      _recordingService.messageStream.listen((message) {
        setState(() {
          _messages.add(message);
          if (_messages.length > 10) {
            _messages.removeAt(0); // Keep only last 10 messages
          }
        });
      });

    } catch (error) {
      print('Failed to initialize recording service: $error');
    }
  }

  Future<void> _toggleRecording() async {
    try {
      if (_isRecording) {
        await _recordingService.stopRecording();
      } else {
        await _recordingService.startRecording('agent-name', 'user-id');
      }
      setState(() {
        _isRecording = !_isRecording;
      });
    } catch (error) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Recording error: $error')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Audio Recorder')),
      body: Padding(
        padding: EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Connection Status: $_connectionStatus',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
            SizedBox(height: 20),
            ElevatedButton(
              onPressed: _toggleRecording,
              child: Text(_isRecording ? 'Stop Recording' : 'Start Recording'),
              style: ElevatedButton.styleFrom(
                backgroundColor: _isRecording ? Colors.red : Colors.green,
                minimumSize: Size(double.infinity, 50),
              ),
            ),
            SizedBox(height: 20),
            Text('Recent Messages:',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
            Expanded(
              child: ListView.builder(
                itemCount: _messages.length,
                itemBuilder: (context, index) {
                  return ListTile(
                    title: Text(_messages[index]),
                    dense: true,
                  );
                },
              ),
            ),
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    _recordingService.dispose();
    super.dispose();
  }
}
```

## Configuration Options

All clients support the following configuration options:

```javascript
{
    maxReconnectAttempts: 10,        // Maximum number of reconnection attempts
    initialReconnectDelay: 1000,     // Initial delay before first reconnection (ms)
    maxReconnectDelay: 30000,        // Maximum delay between reconnections (ms)
    graceWindowMs: 120000,           // Server's reattachment grace period (ms)
    heartbeatIntervalMs: 30000,      // Heartbeat interval to keep connection alive (ms)
    connectionTimeoutMs: 10000,      // WebSocket connection timeout (ms)
    audioFormat: 'audio/webm'        // Audio format ('audio/webm' for web, 'audio/mp4' for mobile)
}
```

## Error Handling Best Practices

1. **Listen to all events**: Set up handlers for `open`, `close`, `error`, `reconnecting`, `reconnected`, and `connectionFailed` events.

2. **Provide user feedback**: Show connection status in your UI and inform users about reconnection attempts.

3. **Handle permanent failures**: When `connectionFailed` is emitted, offer users options like retry or manual reconnection.

4. **Buffer audio during disconnections**: The clients automatically buffer audio, but you may want to inform users that recording continues during brief network outages.

5. **Test network scenarios**: Test your application with various network conditions:
   - WiFi to cellular transitions
   - Complete network loss and restoration
   - Intermittent connectivity
   - Airplane mode on/off

6. **Clean up resources**: Always call `close()` or `dispose()` methods when done to properly clean up timers and connections.

## Integration with Server

These clients are designed to work with the aia-v4 server's resilient recording system:

- **Session reattachment**: Clients automatically reuse session IDs within the 120-second grace period
- **Audio buffering**: Clients buffer audio during disconnections and send it when reconnected
- **Perfect ordering**: Audio chunks maintain timestamps for proper server-side sequencing
- **Status messages**: Clients handle server status messages like `RESUMED` confirmations
- **Heartbeat/ping-pong**: Clients implement keepalive mechanisms compatible with the server

The server handles session continuity, transcript ordering, and provider fallbacks, while these clients ensure reliable connectivity and audio delivery.