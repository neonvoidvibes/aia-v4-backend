/**
 * Resilient WebSocket client for React Native
 * Provides automatic reconnection, audio buffering, and network change detection
 * Compatible with the aia-v4 server's reattachment grace period system
 */

import NetInfo from '@react-native-netinfo/netinfo';
import { AudioRecorder } from '@react-native-async-storage/async-storage';
import { Platform } from 'react-native';

class ResilientWebSocketRN {
    constructor(baseUrl, authToken, options = {}) {
        this.baseUrl = baseUrl;
        this.authToken = authToken;
        this.sessionId = null;
        this.ws = null;

        // Configuration
        this.options = {
            maxReconnectAttempts: 10,
            initialReconnectDelay: 1000, // 1 second
            maxReconnectDelay: 30000,    // 30 seconds max
            graceWindowMs: 120000,       // Server's 120-second grace window
            heartbeatIntervalMs: 30000,  // Send heartbeat every 30 seconds
            connectionTimeoutMs: 10000,  // Connection timeout
            audioFormat: 'audio/mp4',    // Mobile audio format
            ...options
        };

        // State
        this.reconnectAttempts = 0;
        this.reconnectTimer = null;
        this.heartbeatTimer = null;
        this.isIntentionalDisconnect = false;
        this.connectionState = 'disconnected'; // disconnected, connecting, connected, reconnecting
        this.lastDisconnectTime = null;
        this.wasConnected = false;

        // Audio buffering
        this.audioBuffer = [];
        this.isAudioCaptureActive = false;
        this.audioRecorder = null;

        // Network state tracking
        this.networkState = { isConnected: true, type: 'unknown' };
        this.wasOffline = false;

        // Event handlers
        this.eventHandlers = {
            open: [],
            close: [],
            message: [],
            error: [],
            reconnecting: [],
            reconnected: [],
            connectionFailed: [],
            statusChanged: [],
            networkChanged: []
        };

        // Set up network monitoring
        this.setupNetworkMonitoring();

        // Bind methods
        this.connect = this.connect.bind(this);
        this.reconnect = this.reconnect.bind(this);
        this.handleOpen = this.handleOpen.bind(this);
        this.handleClose = this.handleClose.bind(this);
        this.handleError = this.handleError.bind(this);
        this.handleMessage = this.handleMessage.bind(this);
        this.handleNetworkStateChange = this.handleNetworkStateChange.bind(this);
    }

    /**
     * Set up network change monitoring for React Native
     */
    setupNetworkMonitoring() {
        this.netInfoUnsubscribe = NetInfo.addEventListener(this.handleNetworkStateChange);
    }

    /**
     * Handle network state changes
     */
    handleNetworkStateChange(state) {
        console.log('Network state changed:', state);

        const wasConnected = this.networkState.isConnected;
        this.networkState = {
            isConnected: state.isConnected,
            type: state.type,
            details: state.details
        };

        this.emit('networkChanged', this.networkState);

        // Handle reconnection logic
        if (!wasConnected && state.isConnected) {
            // Network was restored
            console.log('Network restored, attempting reconnection...');
            this.wasOffline = false;

            if (this.connectionState === 'disconnected' && !this.isIntentionalDisconnect) {
                // Reset attempts for network restore
                this.reconnectAttempts = 0;
                this.reconnect();
            }
        } else if (wasConnected && !state.isConnected) {
            // Network was lost
            console.log('Network lost');
            this.wasOffline = true;
            this.setConnectionState('disconnected');
        } else if (state.isConnected && wasConnected && state.type !== this.networkState.type) {
            // Network type changed (e.g., WiFi to cellular)
            console.log(`Network type changed from ${this.networkState.type} to ${state.type}`);
            this.testConnectionHealth();
        }
    }

    /**
     * Test connection health by sending a ping
     */
    testConnectionHealth() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            try {
                this.send({ type: 'ping', timestamp: Date.now() });
            } catch (error) {
                console.warn('Health check ping failed, may need reconnection:', error);
            }
        }
    }

    /**
     * Add event listener
     */
    on(event, handler) {
        if (this.eventHandlers[event]) {
            this.eventHandlers[event].push(handler);
        }
    }

    /**
     * Remove event listener
     */
    off(event, handler) {
        if (this.eventHandlers[event]) {
            const index = this.eventHandlers[event].indexOf(handler);
            if (index > -1) {
                this.eventHandlers[event].splice(index, 1);
            }
        }
    }

    /**
     * Emit event to all listeners
     */
    emit(event, ...args) {
        if (this.eventHandlers[event]) {
            this.eventHandlers[event].forEach(handler => {
                try {
                    handler(...args);
                } catch (error) {
                    console.error(`Error in ${event} handler:`, error);
                }
            });
        }
    }

    /**
     * Set connection state and emit status change
     */
    setConnectionState(newState) {
        if (this.connectionState !== newState) {
            const oldState = this.connectionState;
            this.connectionState = newState;
            this.emit('statusChanged', newState, oldState);
        }
    }

    /**
     * Connect to WebSocket with session initialization
     */
    async connect(agentName, userId) {
        if (this.connectionState === 'connecting') {
            console.log('Connection already in progress');
            return;
        }

        // Check network connectivity first
        if (!this.networkState.isConnected) {
            console.log('No network connectivity, waiting for network...');
            throw new Error('No network connectivity');
        }

        this.setConnectionState('connecting');
        this.isIntentionalDisconnect = false;

        try {
            // If we don't have a session ID or it's been too long since disconnect, start new session
            if (!this.sessionId || this.isGracePeriodExpired()) {
                await this.startNewSession(agentName, userId);
            }

            // Connect WebSocket with session ID
            await this.connectWebSocket();

        } catch (error) {
            console.error('Connection failed:', error);
            this.setConnectionState('disconnected');
            this.scheduleReconnect();
        }
    }

    /**
     * Check if grace period has expired
     */
    isGracePeriodExpired() {
        if (!this.lastDisconnectTime) return false;
        return (Date.now() - this.lastDisconnectTime) > this.options.graceWindowMs;
    }

    /**
     * Start a new recording session
     */
    async startNewSession(agentName, userId) {
        const response = await fetch(`${this.baseUrl}/api/audio/start-recording`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${this.authToken}`
            },
            body: JSON.stringify({
                agentName,
                userId,
                contentType: this.options.audioFormat,
                language: Platform.OS === 'ios' ? 'en-US' : 'en-US' // Can be made configurable
            })
        });

        if (!response.ok) {
            throw new Error(`Failed to start session: ${response.statusText}`);
        }

        const data = await response.json();
        this.sessionId = data.session_id;
        console.log(`Started new session: ${this.sessionId}`);
    }

    /**
     * Connect WebSocket with current session
     */
    async connectWebSocket() {
        return new Promise((resolve, reject) => {
            const wsUrl = `${this.baseUrl.replace('http', 'ws')}/api/audio/websocket?session_id=${this.sessionId}&resume=true`;

            this.ws = new WebSocket(wsUrl);
            this.ws.binaryType = 'arraybuffer';

            // Connection timeout
            const timeout = setTimeout(() => {
                if (this.ws.readyState === WebSocket.CONNECTING) {
                    this.ws.close();
                    reject(new Error('Connection timeout'));
                }
            }, this.options.connectionTimeoutMs);

            this.ws.onopen = (event) => {
                clearTimeout(timeout);
                this.handleOpen(event);
                resolve();
            };

            this.ws.onclose = this.handleClose;
            this.ws.onerror = this.handleError;
            this.ws.onmessage = this.handleMessage;
        });
    }

    /**
     * Handle WebSocket open
     */
    handleOpen(event) {
        console.log(`WebSocket connected for session ${this.sessionId}`);

        // Reset reconnection state
        this.reconnectAttempts = 0;
        this.clearReconnectTimer();

        // Update state
        const wasReconnecting = this.connectionState === 'reconnecting';
        this.setConnectionState('connected');

        // Start heartbeat
        this.startHeartbeat();

        // Send buffered audio if any
        this.sendBufferedAudio();

        // Emit appropriate event
        if (wasReconnecting && this.wasConnected) {
            this.emit('reconnected', event);
        } else {
            this.wasConnected = true;
            this.emit('open', event);
        }

        // Resume audio capture if it was active
        if (this.isAudioCaptureActive) {
            this.resumeAudioCapture();
        }
    }

    /**
     * Handle WebSocket close
     */
    handleClose(event) {
        console.log(`WebSocket closed: ${event.code} - ${event.reason}`);

        this.stopHeartbeat();
        this.lastDisconnectTime = Date.now();

        if (this.isIntentionalDisconnect) {
            this.setConnectionState('disconnected');
            this.emit('close', event);
        } else {
            // Unintentional disconnect - attempt reconnection if we have network
            this.setConnectionState('disconnected');

            if (this.networkState.isConnected) {
                this.scheduleReconnect();
            } else {
                console.log('WebSocket closed and no network - will reconnect when network returns');
            }
        }
    }

    /**
     * Handle WebSocket error
     */
    handleError(event) {
        console.error('WebSocket error:', event);
        this.emit('error', event);
    }

    /**
     * Handle WebSocket message
     */
    handleMessage(event) {
        try {
            const data = JSON.parse(event.data);

            // Handle server status messages
            if (data.type === 'status') {
                if (data.state === 'RESUMED') {
                    console.log('Session successfully resumed');
                }
            }

            // Handle pong responses
            if (data.type === 'pong') {
                // Connection is healthy
                return;
            }

            this.emit('message', event);

        } catch (error) {
            // Not JSON, pass through as-is
            this.emit('message', event);
        }
    }

    /**
     * Send message through WebSocket
     */
    send(data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            if (typeof data === 'object') {
                this.ws.send(JSON.stringify(data));
            } else {
                this.ws.send(data);
            }
            return true;
        }
        return false;
    }

    /**
     * Send audio data (with buffering support)
     */
    sendAudio(audioData, timestamp = null) {
        const audioChunk = {
            data: audioData,
            timestamp: timestamp || Date.now()
        };

        if (this.connectionState === 'connected') {
            // Send immediately
            this.ws.send(audioData);
        } else {
            // Buffer for later
            this.audioBuffer.push(audioChunk);
            console.log(`Buffered audio chunk (${this.audioBuffer.length} chunks buffered)`);
        }
    }

    /**
     * Send all buffered audio chunks
     */
    sendBufferedAudio() {
        if (this.audioBuffer.length === 0) return;

        console.log(`Sending ${this.audioBuffer.length} buffered audio chunks`);

        // Sort by timestamp to maintain order
        this.audioBuffer.sort((a, b) => a.timestamp - b.timestamp);

        // Send each chunk
        for (const chunk of this.audioBuffer) {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(chunk.data);
            }
        }

        // Clear buffer
        this.audioBuffer = [];
    }

    /**
     * Start heartbeat to keep connection alive
     */
    startHeartbeat() {
        this.stopHeartbeat(); // Clear any existing heartbeat

        this.heartbeatTimer = setInterval(() => {
            if (this.connectionState === 'connected') {
                this.send({ type: 'ping', timestamp: Date.now() });
            }
        }, this.options.heartbeatIntervalMs);
    }

    /**
     * Stop heartbeat
     */
    stopHeartbeat() {
        if (this.heartbeatTimer) {
            clearInterval(this.heartbeatTimer);
            this.heartbeatTimer = null;
        }
    }

    /**
     * Schedule reconnection attempt
     */
    scheduleReconnect() {
        if (this.isIntentionalDisconnect) return;

        // Don't reconnect if no network
        if (!this.networkState.isConnected) {
            console.log('No network connectivity, will retry when network returns');
            return;
        }

        if (this.reconnectAttempts >= this.options.maxReconnectAttempts) {
            console.error('Max reconnection attempts reached');
            this.emit('connectionFailed');
            return;
        }

        this.setConnectionState('reconnecting');
        this.emit('reconnecting', this.reconnectAttempts + 1);

        // Calculate delay with exponential backoff
        const delay = Math.min(
            this.options.initialReconnectDelay * Math.pow(2, this.reconnectAttempts),
            this.options.maxReconnectDelay
        );

        console.log(`Scheduling reconnection attempt ${this.reconnectAttempts + 1} in ${delay}ms`);

        this.reconnectTimer = setTimeout(() => {
            this.reconnectAttempts++;
            this.reconnect();
        }, delay);
    }

    /**
     * Clear reconnection timer
     */
    clearReconnectTimer() {
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
    }

    /**
     * Attempt reconnection
     */
    async reconnect() {
        if (this.isIntentionalDisconnect) return;

        // Check network state before attempting
        if (!this.networkState.isConnected) {
            console.log('No network for reconnection, waiting...');
            return;
        }

        console.log(`Reconnection attempt ${this.reconnectAttempts}`);

        try {
            await this.connectWebSocket();
        } catch (error) {
            console.error('Reconnection failed:', error);
            this.scheduleReconnect();
        }
    }

    /**
     * Start audio capture (React Native implementation)
     */
    async startAudioCapture() {
        try {
            // Note: This is a simplified example. Real implementation would use
            // react-native-audio-record or similar library
            console.log('Starting audio capture (implement with react-native-audio-record)');

            // Pseudo-code for audio capture:
            /*
            import AudioRecord from 'react-native-audio-record';

            const options = {
                sampleRate: 16000,
                channels: 1,
                bitsPerSample: 16,
                audioEncoding: 'mp4',
                includeBase64: false,
                audioFormat: 'mp4',
            };

            AudioRecord.init(options);
            AudioRecord.start();
            AudioRecord.on('data', data => {
                this.sendAudio(data);
            });
            */

            this.isAudioCaptureActive = true;
            console.log('Audio capture started');

        } catch (error) {
            console.error('Failed to start audio capture:', error);
            throw error;
        }
    }

    /**
     * Stop audio capture
     */
    async stopAudioCapture() {
        try {
            // AudioRecord.stop();
            this.isAudioCaptureActive = false;
            console.log('Audio capture stopped');
        } catch (error) {
            console.error('Error stopping audio capture:', error);
        }
    }

    /**
     * Resume audio capture after reconnection
     */
    async resumeAudioCapture() {
        if (this.isAudioCaptureActive) {
            console.log('Resuming audio capture after reconnection');
            try {
                await this.startAudioCapture();
            } catch (error) {
                console.error('Failed to resume audio capture:', error);
            }
        }
    }

    /**
     * Close connection intentionally
     */
    close(code = 1000, reason = 'Client closing') {
        console.log('Closing WebSocket connection intentionally');

        this.isIntentionalDisconnect = true;
        this.clearReconnectTimer();
        this.stopHeartbeat();
        this.stopAudioCapture();

        if (this.ws) {
            this.ws.close(code, reason);
        }

        this.setConnectionState('disconnected');
    }

    /**
     * Clean up resources
     */
    cleanup() {
        this.close();

        if (this.netInfoUnsubscribe) {
            this.netInfoUnsubscribe();
        }
    }

    /**
     * Get current connection status
     */
    getStatus() {
        return {
            state: this.connectionState,
            sessionId: this.sessionId,
            reconnectAttempts: this.reconnectAttempts,
            bufferedAudioChunks: this.audioBuffer.length,
            isAudioCaptureActive: this.isAudioCaptureActive,
            lastDisconnectTime: this.lastDisconnectTime,
            gracePeriodExpired: this.isGracePeriodExpired(),
            networkState: this.networkState
        };
    }
}

export default ResilientWebSocketRN;