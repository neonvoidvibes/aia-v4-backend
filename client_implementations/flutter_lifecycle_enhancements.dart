/**
 * Flutter App Lifecycle Enhancements
 * Handles app backgrounding, incoming calls, and system interruptions
 */

import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'flutter_resilient_websocket.dart';

enum AppLifecycleState {
  resumed,
  inactive,
  paused,
  detached,
}

class ResilientWebSocketFlutterWithLifecycle extends ResilientWebSocketFlutter
    with WidgetsBindingObserver {

  AppLifecycleState _appState = AppLifecycleState.resumed;
  bool _wasRecordingBeforeBackground = false;
  DateTime? _backgroundTime;
  Timer? _backgroundHeartbeat;

  // Call detection stream controller
  final StreamController<bool> _callStateController = StreamController.broadcast();
  Stream<bool> get callStateStream => _callStateController.stream;

  // App lifecycle stream controller
  final StreamController<AppLifecycleState> _lifecycleController = StreamController.broadcast();
  Stream<AppLifecycleState> get lifecycleStream => _lifecycleController.stream;

  ResilientWebSocketFlutterWithLifecycle(String baseUrl, String authToken, [Map<String, dynamic>? options])
      : super(baseUrl, authToken, options) {
    _setupAppLifecycleMonitoring();
  }

  void _setupAppLifecycleMonitoring() {
    // Register as observer for app lifecycle changes
    WidgetsBinding.instance.addObserver(this);

    // Setup call detection (requires additional plugins)
    _setupCallDetection();
  }

  void _setupCallDetection() {
    // Example using phone_state plugin
    // Add to pubspec.yaml: phone_state: ^1.0.3
    try {
      // PhoneState.stream.listen((status) {
      //   switch (status) {
      //     case PhoneStateStatus.CALL_INCOMING:
      //       _handleIncomingCall();
      //       break;
      //     case PhoneStateStatus.CALL_STARTED:
      //       _handleCallStarted();
      //       break;
      //     case PhoneStateStatus.CALL_ENDED:
      //       _handleCallEnded();
      //       break;
      //   }
      // });
    } catch (error) {
      print('Call detection not available: $error');
    }
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    print('App lifecycle state changed: ${_appState.toString()} → ${state.toString()}');

    switch (state) {
      case AppLifecycleState.resumed:
        if (_appState == AppLifecycleState.paused) {
          _handleAppForeground();
        }
        break;
      case AppLifecycleState.inactive:
        // App is transitioning (incoming call, notification, etc.)
        _handleAppInactive();
        break;
      case AppLifecycleState.paused:
        _handleAppBackground();
        break;
      case AppLifecycleState.detached:
        _handleAppDetached();
        break;
    }

    _appState = state;
    _lifecycleController.add(state);
  }

  void _handleAppBackground() {
    print('Flutter app went to background');
    _backgroundTime = DateTime.now();
    _wasRecordingBeforeBackground = _isAudioCaptureActive;

    // Stop audio recording (system requirement)
    if (_isAudioCaptureActive) {
      stopAudioCapture();
    }

    // Adjust heartbeat for background
    _adjustHeartbeatForBackground();

    // Notify server about background state
    send({
      'type': 'app_state_changed',
      'state': 'background',
      'timestamp': DateTime.now().millisecondsSinceEpoch,
    });
  }

  void _handleAppForeground() {
    print('Flutter app returned to foreground');
    final backgroundDuration = _backgroundTime != null
        ? DateTime.now().difference(_backgroundTime!).inMilliseconds
        : 0;

    _backgroundTime = null;

    // Test connection health after returning from background
    Future.delayed(Duration(milliseconds: 500), () {
      _testConnectionHealth();

      // Resume audio capture if it was active before backgrounding
      if (_wasRecordingBeforeBackground) {
        startAudioCapture().catchError((error) {
          print('Failed to resume audio after foreground: $error');
        });
        _wasRecordingBeforeBackground = false;
      }
    });

    // Restore normal heartbeat
    _adjustHeartbeatForForeground();

    // Notify server about foreground state
    send({
      'type': 'app_state_changed',
      'state': 'foreground',
      'background_duration_ms': backgroundDuration,
      'timestamp': DateTime.now().millisecondsSinceEpoch,
    });
  }

  void _handleAppInactive() {
    print('Flutter app became inactive');

    // App is transitioning - could be incoming call, notification panel, etc.
    // Pause recording temporarily but don't fully stop
    if (_isAudioCaptureActive && _audioRecorder != null) {
      // Note: Actual pause implementation depends on the audio library
      // Some libraries don't support pause, so you might need to stop/start
      print('App inactive - pausing audio capture');
    }
  }

  void _handleAppDetached() {
    print('Flutter app detached from engine');
    // Clean shutdown
    if (_isAudioCaptureActive) {
      stopAudioCapture();
    }
    close(1001, 'App detached');
  }

  void _handleIncomingCall() {
    print('Incoming call detected');
    _wasRecordingBeforeBackground = _isAudioCaptureActive;

    if (_isAudioCaptureActive) {
      stopAudioCapture();
    }

    // Notify server about call interruption
    send({
      'type': 'audio_interrupted',
      'reason': 'incoming_call',
      'timestamp': DateTime.now().millisecondsSinceEpoch,
    });

    _callStateController.add(true); // Call started
  }

  void _handleCallStarted() {
    print('Call started');
    // Call is now active, ensure recording is stopped
    if (_isAudioCaptureActive) {
      stopAudioCapture();
    }
  }

  void _handleCallEnded() {
    print('Call ended');

    // Wait before resuming to ensure call is fully ended
    Future.delayed(Duration(seconds: 2), () {
      if (_wasRecordingBeforeBackground && _appState == AppLifecycleState.resumed) {
        startAudioCapture().catchError((error) {
          print('Failed to resume audio after call: $error');
        });
        _wasRecordingBeforeBackground = false;
      }
    });

    send({
      'type': 'audio_resumed',
      'reason': 'call_ended',
      'timestamp': DateTime.now().millisecondsSinceEpoch,
    });

    _callStateController.add(false); // Call ended
  }

  void _adjustHeartbeatForBackground() {
    _stopHeartbeat();

    // Use longer intervals in background to comply with mobile OS restrictions
    _backgroundHeartbeat = Timer.periodic(
      Duration(minutes: 2), // 2 minute intervals
      (timer) {
        if (_connectionState == ConnectionState.connected) {
          send({
            'type': 'ping',
            'timestamp': DateTime.now().millisecondsSinceEpoch,
            'app_state': 'background'
          });
        }
      },
    );
  }

  void _adjustHeartbeatForForeground() {
    _backgroundHeartbeat?.cancel();
    _backgroundHeartbeat = null;
    _startHeartbeat(); // Resume normal heartbeat
  }

  // Override connection handling for app lifecycle
  @override
  void _handleClose() {
    super._handleClose();

    // If app is in background and connection closed, be more patient
    if (_appState == AppLifecycleState.paused && !_isIntentionalDisconnect) {
      print('Connection closed while in background, will reconnect when app returns to foreground');
      // Don't aggressively reconnect in background
      return;
    }
  }

  // Enhanced status with app state
  @override
  Map<String, dynamic> getStatus() {
    final baseStatus = super.getStatus();
    return {
      ...baseStatus,
      'appState': _appState.toString(),
      'wasRecordingBeforeBackground': _wasRecordingBeforeBackground,
      'backgroundTime': _backgroundTime?.toIso8601String(),
      'backgroundDuration': _backgroundTime != null
          ? DateTime.now().difference(_backgroundTime!).inMilliseconds
          : null,
    };
  }

  @override
  void dispose() {
    // Remove lifecycle observer
    WidgetsBinding.instance.removeObserver(this);

    // Cancel background heartbeat
    _backgroundHeartbeat?.cancel();

    // Close stream controllers
    _callStateController.close();
    _lifecycleController.close();

    super.dispose();
  }
}

// Usage Example Widget
class AudioRecorderWithLifecycle extends StatefulWidget {
  @override
  _AudioRecorderWithLifecycleState createState() => _AudioRecorderWithLifecycleState();
}

class _AudioRecorderWithLifecycleState extends State<AudioRecorderWithLifecycle> {
  late ResilientWebSocketFlutterWithLifecycle _webSocket;
  String _connectionStatus = 'disconnected';
  String _appState = 'resumed';
  bool _isInCall = false;
  bool _isRecording = false;

  @override
  void initState() {
    super.initState();
    _initializeWebSocket();
  }

  Future<void> _initializeWebSocket() async {
    _webSocket = ResilientWebSocketFlutterWithLifecycle(
      'https://your-server.com',
      'your-auth-token'
    );

    // Listen to connection status
    _webSocket.onStatusChanged.listen((status) {
      setState(() {
        _connectionStatus = status.toString();
      });
    });

    // Listen to app lifecycle changes
    _webSocket.lifecycleStream.listen((state) {
      setState(() {
        _appState = state.toString();
      });

      switch (state) {
        case AppLifecycleState.paused:
          _showSnackBar('App backgrounded - recording paused');
          break;
        case AppLifecycleState.resumed:
          _showSnackBar('App resumed - recording restored');
          break;
        case AppLifecycleState.inactive:
          _showSnackBar('App inactive - possible interruption');
          break;
        default:
          break;
      }
    });

    // Listen to call state changes
    _webSocket.callStateStream.listen((inCall) {
      setState(() {
        _isInCall = inCall;
      });

      if (inCall) {
        _showSnackBar('Incoming call - recording paused');
      } else {
        _showSnackBar('Call ended - recording resumed');
      }
    });
  }

  void _showSnackBar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message), duration: Duration(seconds: 2)),
    );
  }

  Future<void> _toggleRecording() async {
    try {
      if (_isRecording) {
        await _webSocket.stopAudioCapture();
        _webSocket.close();
      } else {
        await _webSocket.connect('agent-name', 'user-id');
        await _webSocket.startAudioCapture();
      }
      setState(() {
        _isRecording = !_isRecording;
      });
    } catch (error) {
      _showSnackBar('Recording error: $error');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Lifecycle-Aware Audio Recorder')),
      body: Padding(
        padding: EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildStatusCard('Connection', _connectionStatus),
            _buildStatusCard('App State', _appState),
            _buildStatusCard('Call State', _isInCall ? 'In Call' : 'No Call'),
            SizedBox(height: 20),

            ElevatedButton(
              onPressed: _isInCall ? null : _toggleRecording,
              child: Text(_isRecording ? 'Stop Recording' : 'Start Recording'),
              style: ElevatedButton.styleFrom(
                backgroundColor: _isRecording ? Colors.red : Colors.green,
                minimumSize: Size(double.infinity, 50),
              ),
            ),

            if (_isInCall)
              Padding(
                padding: EdgeInsets.only(top: 16),
                child: Text(
                  'Recording paused due to active call',
                  style: TextStyle(color: Colors.orange, fontWeight: FontWeight.bold),
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _buildStatusCard(String title, String value) {
    return Card(
      child: ListTile(
        title: Text(title),
        subtitle: Text(value),
        dense: true,
      ),
    );
  }

  @override
  void dispose() {
    _webSocket.dispose();
    super.dispose();
  }
}