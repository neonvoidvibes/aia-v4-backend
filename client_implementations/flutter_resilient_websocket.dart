/**
 * Resilient WebSocket client for Flutter
 * Provides automatic reconnection, audio buffering, and network change detection
 * Compatible with the aia-v4 server's reattachment grace period system
 */

import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:connectivity_plus/connectivity_plus.dart';
import 'package:record/record.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:http/http.dart' as http;

class AudioChunk {
  final Uint8List data;
  final DateTime timestamp;

  AudioChunk(this.data, this.timestamp);
}

enum ConnectionState {
  disconnected,
  connecting,
  connected,
  reconnecting,
}

class ResilientWebSocketFlutter {
  final String baseUrl;
  final String authToken;
  final Map<String, dynamic> options;

  String? sessionId;
  WebSocketChannel? _ws;
  ConnectionState _connectionState = ConnectionState.disconnected;

  // Reconnection state
  int _reconnectAttempts = 0;
  Timer? _reconnectTimer;
  Timer? _heartbeatTimer;
  bool _isIntentionalDisconnect = false;
  DateTime? _lastDisconnectTime;
  bool _wasConnected = false;

  // Audio buffering
  final List<AudioChunk> _audioBuffer = [];
  bool _isAudioCaptureActive = false;
  Record? _audioRecorder;

  // Network monitoring
  late StreamSubscription<ConnectivityResult> _connectivitySubscription;
  ConnectivityResult _currentConnectivity = ConnectivityResult.none;
  bool _wasOffline = false;

  // Event streams
  final StreamController<void> _openController = StreamController.broadcast();
  final StreamController<void> _closeController = StreamController.broadcast();
  final StreamController<dynamic> _messageController = StreamController.broadcast();
  final StreamController<dynamic> _errorController = StreamController.broadcast();
  final StreamController<int> _reconnectingController = StreamController.broadcast();
  final StreamController<void> _reconnectedController = StreamController.broadcast();
  final StreamController<void> _connectionFailedController = StreamController.broadcast();
  final StreamController<ConnectionState> _statusChangedController = StreamController.broadcast();

  // Default configuration
  static const Map<String, dynamic> _defaultOptions = {
    'maxReconnectAttempts': 10,
    'initialReconnectDelay': 1000, // milliseconds
    'maxReconnectDelay': 30000,    // milliseconds
    'graceWindowMs': 120000,       // Server's 120-second grace window
    'heartbeatIntervalMs': 30000,  // Send heartbeat every 30 seconds
    'connectionTimeoutMs': 10000,  // Connection timeout
    'audioFormat': 'audio/mp4',    // Mobile audio format
  };

  ResilientWebSocketFlutter(this.baseUrl, this.authToken, [Map<String, dynamic>? options])
      : options = {..._defaultOptions, ...?options} {
    _setupNetworkMonitoring();
  }

  // Event streams getters
  Stream<void> get onOpen => _openController.stream;
  Stream<void> get onClose => _closeController.stream;
  Stream<dynamic> get onMessage => _messageController.stream;
  Stream<dynamic> get onError => _errorController.stream;
  Stream<int> get onReconnecting => _reconnectingController.stream;
  Stream<void> get onReconnected => _reconnectedController.stream;
  Stream<void> get onConnectionFailed => _connectionFailedController.stream;
  Stream<ConnectionState> get onStatusChanged => _statusChangedController.stream;

  ConnectionState get connectionState => _connectionState;
  int get bufferedAudioChunks => _audioBuffer.length;
  bool get isAudioCaptureActive => _isAudioCaptureActive;

  /**
   * Set up network connectivity monitoring
   */
  void _setupNetworkMonitoring() {
    _connectivitySubscription = Connectivity().onConnectivityChanged.listen(_handleNetworkStateChange);

    // Get initial connectivity state
    Connectivity().checkConnectivity().then((result) {
      _currentConnectivity = result;
    });
  }

  /**
   * Handle network state changes
   */
  void _handleNetworkStateChange(ConnectivityResult result) {
    print('Network state changed: $result');

    final wasConnected = _currentConnectivity != ConnectivityResult.none;
    final isConnected = result != ConnectivityResult.none;
    _currentConnectivity = result;

    // Handle reconnection logic
    if (!wasConnected && isConnected) {
      // Network was restored
      print('Network restored, attempting reconnection...');
      _wasOffline = false;

      if (_connectionState == ConnectionState.disconnected && !_isIntentionalDisconnect) {
        // Reset attempts for network restore
        _reconnectAttempts = 0;
        _reconnect();
      }
    } else if (wasConnected && !isConnected) {
      // Network was lost
      print('Network lost');
      _wasOffline = true;
      _setConnectionState(ConnectionState.disconnected);
    } else if (isConnected && wasConnected && result != _currentConnectivity) {
      // Network type changed
      print('Network type changed to $result');
      _testConnectionHealth();
    }
  }

  /**
   * Test connection health by sending a ping
   */
  void _testConnectionHealth() {
    if (_ws != null) {
      try {
        send({'type': 'ping', 'timestamp': DateTime.now().millisecondsSinceEpoch});
      } catch (error) {
        print('Health check ping failed, may need reconnection: $error');
      }
    }
  }

  /**
   * Set connection state and emit status change
   */
  void _setConnectionState(ConnectionState newState) {
    if (_connectionState != newState) {
      _connectionState = newState;
      _statusChangedController.add(newState);
    }
  }

  /**
   * Connect to WebSocket with session initialization
   */
  Future<void> connect(String agentName, String userId) async {
    if (_connectionState == ConnectionState.connecting) {
      print('Connection already in progress');
      return;
    }

    // Check network connectivity first
    if (_currentConnectivity == ConnectivityResult.none) {
      print('No network connectivity, waiting for network...');
      throw Exception('No network connectivity');
    }

    _setConnectionState(ConnectionState.connecting);
    _isIntentionalDisconnect = false;

    try {
      // If we don't have a session ID or it's been too long since disconnect, start new session
      if (sessionId == null || _isGracePeriodExpired()) {
        await _startNewSession(agentName, userId);
      }

      // Connect WebSocket with session ID
      await _connectWebSocket();

    } catch (error) {
      print('Connection failed: $error');
      _setConnectionState(ConnectionState.disconnected);
      _scheduleReconnect();
    }
  }

  /**
   * Check if grace period has expired
   */
  bool _isGracePeriodExpired() {
    if (_lastDisconnectTime == null) return false;
    return DateTime.now().difference(_lastDisconnectTime!).inMilliseconds > options['graceWindowMs'];
  }

  /**
   * Start a new recording session
   */
  Future<void> _startNewSession(String agentName, String userId) async {
    final response = await http.post(
      Uri.parse('$baseUrl/api/audio/start-recording'),
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer $authToken',
      },
      body: json.encode({
        'agentName': agentName,
        'userId': userId,
        'contentType': options['audioFormat'],
        'language': 'en-US', // Can be made configurable
      }),
    );

    if (response.statusCode != 200) {
      throw Exception('Failed to start session: ${response.reasonPhrase}');
    }

    final data = json.decode(response.body);
    sessionId = data['session_id'];
    print('Started new session: $sessionId');
  }

  /**
   * Connect WebSocket with current session
   */
  Future<void> _connectWebSocket() async {
    final wsUrl = baseUrl.replaceFirst('http', 'ws') + '/api/audio/websocket?session_id=$sessionId&resume=true';

    final completer = Completer<void>();
    Timer? timeout;

    try {
      _ws = WebSocketChannel.connect(Uri.parse(wsUrl));

      // Connection timeout
      timeout = Timer(Duration(milliseconds: options['connectionTimeoutMs']), () {
        if (!completer.isCompleted) {
          _ws?.sink.close();
          completer.completeError(Exception('Connection timeout'));
        }
      });

      // Listen to WebSocket stream
      _ws!.stream.listen(
        _handleMessage,
        onError: (error) {
          _handleError(error);
          if (!completer.isCompleted) {
            completer.completeError(error);
          }
        },
        onDone: () {
          _handleClose();
        },
      );

      // Wait a bit to ensure connection is established
      await Future.delayed(Duration(milliseconds: 100));

      timeout?.cancel();
      _handleOpen();
      completer.complete();

    } catch (error) {
      timeout?.cancel();
      if (!completer.isCompleted) {
        completer.completeError(error);
      }
    }

    return completer.future;
  }

  /**
   * Handle WebSocket open
   */
  void _handleOpen() {
    print('WebSocket connected for session $sessionId');

    // Reset reconnection state
    _reconnectAttempts = 0;
    _clearReconnectTimer();

    // Update state
    final wasReconnecting = _connectionState == ConnectionState.reconnecting;
    _setConnectionState(ConnectionState.connected);

    // Start heartbeat
    _startHeartbeat();

    // Send buffered audio if any
    _sendBufferedAudio();

    // Emit appropriate event
    if (wasReconnecting && _wasConnected) {
      _reconnectedController.add(null);
    } else {
      _wasConnected = true;
      _openController.add(null);
    }

    // Resume audio capture if it was active
    if (_isAudioCaptureActive) {
      _resumeAudioCapture();
    }
  }

  /**
   * Handle WebSocket close
   */
  void _handleClose() {
    print('WebSocket closed');

    _stopHeartbeat();
    _lastDisconnectTime = DateTime.now();

    if (_isIntentionalDisconnect) {
      _setConnectionState(ConnectionState.disconnected);
      _closeController.add(null);
    } else {
      // Unintentional disconnect - attempt reconnection if we have network
      _setConnectionState(ConnectionState.disconnected);

      if (_currentConnectivity != ConnectivityResult.none) {
        _scheduleReconnect();
      } else {
        print('WebSocket closed and no network - will reconnect when network returns');
      }
    }
  }

  /**
   * Handle WebSocket error
   */
  void _handleError(dynamic error) {
    print('WebSocket error: $error');
    _errorController.add(error);
  }

  /**
   * Handle WebSocket message
   */
  void _handleMessage(dynamic message) {
    try {
      if (message is String) {
        final data = json.decode(message);

        // Handle server status messages
        if (data['type'] == 'status') {
          if (data['state'] == 'RESUMED') {
            print('Session successfully resumed');
          }
        }

        // Handle pong responses
        if (data['type'] == 'pong') {
          // Connection is healthy
          return;
        }
      }

      _messageController.add(message);

    } catch (error) {
      // Not JSON, pass through as-is
      _messageController.add(message);
    }
  }

  /**
   * Send message through WebSocket
   */
  bool send(dynamic data) {
    if (_ws != null) {
      try {
        if (data is Map || data is List) {
          _ws!.sink.add(json.encode(data));
        } else {
          _ws!.sink.add(data);
        }
        return true;
      } catch (error) {
        print('Error sending message: $error');
      }
    }
    return false;
  }

  /**
   * Send audio data (with buffering support)
   */
  void sendAudio(Uint8List audioData, [DateTime? timestamp]) {
    final audioChunk = AudioChunk(audioData, timestamp ?? DateTime.now());

    if (_connectionState == ConnectionState.connected) {
      // Send immediately
      _ws?.sink.add(audioData);
    } else {
      // Buffer for later
      _audioBuffer.add(audioChunk);
      print('Buffered audio chunk (${_audioBuffer.length} chunks buffered)');
    }
  }

  /**
   * Send all buffered audio chunks
   */
  void _sendBufferedAudio() {
    if (_audioBuffer.isEmpty) return;

    print('Sending ${_audioBuffer.length} buffered audio chunks');

    // Sort by timestamp to maintain order
    _audioBuffer.sort((a, b) => a.timestamp.compareTo(b.timestamp));

    // Send each chunk
    for (final chunk in _audioBuffer) {
      if (_ws != null) {
        _ws!.sink.add(chunk.data);
      }
    }

    // Clear buffer
    _audioBuffer.clear();
  }

  /**
   * Start heartbeat to keep connection alive
   */
  void _startHeartbeat() {
    _stopHeartbeat(); // Clear any existing heartbeat

    _heartbeatTimer = Timer.periodic(
      Duration(milliseconds: options['heartbeatIntervalMs']),
      (timer) {
        if (_connectionState == ConnectionState.connected) {
          send({'type': 'ping', 'timestamp': DateTime.now().millisecondsSinceEpoch});
        }
      },
    );
  }

  /**
   * Stop heartbeat
   */
  void _stopHeartbeat() {
    _heartbeatTimer?.cancel();
    _heartbeatTimer = null;
  }

  /**
   * Schedule reconnection attempt
   */
  void _scheduleReconnect() {
    if (_isIntentionalDisconnect) return;

    // Don't reconnect if no network
    if (_currentConnectivity == ConnectivityResult.none) {
      print('No network connectivity, will retry when network returns');
      return;
    }

    if (_reconnectAttempts >= options['maxReconnectAttempts']) {
      print('Max reconnection attempts reached');
      _connectionFailedController.add(null);
      return;
    }

    _setConnectionState(ConnectionState.reconnecting);
    _reconnectingController.add(_reconnectAttempts + 1);

    // Calculate delay with exponential backoff
    final delay = (options['initialReconnectDelay'] *
                   (1 << _reconnectAttempts)).clamp(0, options['maxReconnectDelay']);

    print('Scheduling reconnection attempt ${_reconnectAttempts + 1} in ${delay}ms');

    _reconnectTimer = Timer(Duration(milliseconds: delay), () {
      _reconnectAttempts++;
      _reconnect();
    });
  }

  /**
   * Clear reconnection timer
   */
  void _clearReconnectTimer() {
    _reconnectTimer?.cancel();
    _reconnectTimer = null;
  }

  /**
   * Attempt reconnection
   */
  Future<void> _reconnect() async {
    if (_isIntentionalDisconnect) return;

    // Check network state before attempting
    if (_currentConnectivity == ConnectivityResult.none) {
      print('No network for reconnection, waiting...');
      return;
    }

    print('Reconnection attempt $_reconnectAttempts');

    try {
      await _connectWebSocket();
    } catch (error) {
      print('Reconnection failed: $error');
      _scheduleReconnect();
    }
  }

  /**
   * Start audio capture
   */
  Future<void> startAudioCapture() async {
    try {
      _audioRecorder = Record();

      if (await _audioRecorder!.hasPermission()) {
        await _audioRecorder!.start(
          path: null, // Use default path or configure as needed
          encoder: AudioEncoder.aacLc, // For MP4 format
          bitRate: 16000,
          samplingRate: 16000,
        );

        // Set up periodic audio data collection
        Timer.periodic(Duration(milliseconds: 100), (timer) {
          if (!_isAudioCaptureActive) {
            timer.cancel();
            return;
          }

          // In a real implementation, you'd collect audio data here
          // This is a placeholder - actual implementation would depend on
          // the audio recording library's API
          // sendAudio(audioData);
        });

        _isAudioCaptureActive = true;
        print('Audio capture started');
      } else {
        throw Exception('Audio recording permission not granted');
      }

    } catch (error) {
      print('Failed to start audio capture: $error');
      throw error;
    }
  }

  /**
   * Stop audio capture
   */
  Future<void> stopAudioCapture() async {
    try {
      if (_audioRecorder != null) {
        await _audioRecorder!.stop();
        _audioRecorder = null;
      }
      _isAudioCaptureActive = false;
      print('Audio capture stopped');
    } catch (error) {
      print('Error stopping audio capture: $error');
    }
  }

  /**
   * Resume audio capture after reconnection
   */
  Future<void> _resumeAudioCapture() async {
    if (_isAudioCaptureActive) {
      print('Resuming audio capture after reconnection');
      try {
        await startAudioCapture();
      } catch (error) {
        print('Failed to resume audio capture: $error');
      }
    }
  }

  /**
   * Close connection intentionally
   */
  void close([int? code, String? reason]) {
    print('Closing WebSocket connection intentionally');

    _isIntentionalDisconnect = true;
    _clearReconnectTimer();
    _stopHeartbeat();
    stopAudioCapture();

    _ws?.sink.close(code ?? 1000, reason ?? 'Client closing');
    _setConnectionState(ConnectionState.disconnected);
  }

  /**
   * Clean up all resources
   */
  void dispose() {
    close();
    _connectivitySubscription.cancel();

    // Close all stream controllers
    _openController.close();
    _closeController.close();
    _messageController.close();
    _errorController.close();
    _reconnectingController.close();
    _reconnectedController.close();
    _connectionFailedController.close();
    _statusChangedController.close();
  }

  /**
   * Get current connection status
   */
  Map<String, dynamic> getStatus() {
    return {
      'state': _connectionState.toString(),
      'sessionId': sessionId,
      'reconnectAttempts': _reconnectAttempts,
      'bufferedAudioChunks': _audioBuffer.length,
      'isAudioCaptureActive': _isAudioCaptureActive,
      'lastDisconnectTime': _lastDisconnectTime?.toIso8601String(),
      'gracePeriodExpired': _isGracePeriodExpired(),
      'networkConnectivity': _currentConnectivity.toString(),
    };
  }
}