/**
 * App Lifecycle Enhancements for Resilient WebSocket
 * Handles app backgrounding, incoming calls, and foreground restoration
 */

// Web Browser Enhancement
class ResilientWebSocketWithLifecycle extends ResilientWebSocket {
    constructor(baseUrl, authToken, options = {}) {
        super(baseUrl, authToken, options);

        // App lifecycle state
        this.appState = 'active';
        this.wasRecordingBeforeBackground = false;
        this.backgroundTime = null;

        // Setup lifecycle monitoring
        this.setupAppLifecycleMonitoring();
    }

    setupAppLifecycleMonitoring() {
        if (typeof document !== 'undefined') {
            // Page visibility changes (tab switching, minimize)
            document.addEventListener('visibilitychange', this.handleVisibilityChange.bind(this));

            // Window focus/blur events
            window.addEventListener('focus', this.handleAppForeground.bind(this));
            window.addEventListener('blur', this.handleAppBackground.bind(this));

            // Before page unload
            window.addEventListener('beforeunload', this.handleAppTermination.bind(this));

            // Page freeze/resume (modern browsers)
            if ('onfreeze' in document) {
                document.addEventListener('freeze', this.handleAppFreeze.bind(this));
                document.addEventListener('resume', this.handleAppResume.bind(this));
            }
        }
    }

    handleVisibilityChange() {
        if (document.visibilityState === 'hidden') {
            this.handleAppBackground();
        } else if (document.visibilityState === 'visible') {
            this.handleAppForeground();
        }
    }

    handleAppBackground() {
        console.log('App went to background');
        this.appState = 'background';
        this.backgroundTime = Date.now();

        // Remember if we were recording
        this.wasRecordingBeforeBackground = this.isAudioCaptureActive;

        // Pause audio capture but keep connection alive
        if (this.isAudioCaptureActive) {
            this.pauseAudioCapture();
        }

        // Reduce heartbeat frequency to preserve battery
        this.adjustHeartbeatForBackground();

        this.emit('appBackground');
    }

    handleAppForeground() {
        console.log('App returned to foreground');
        const backgroundDuration = this.backgroundTime ? Date.now() - this.backgroundTime : 0;

        this.appState = 'active';
        this.backgroundTime = null;

        // Check connection health after returning from background
        this.testConnectionHealth();

        // Resume audio capture if it was active before backgrounding
        if (this.wasRecordingBeforeBackground) {
            setTimeout(() => {
                this.resumeAudioCapture();
                this.wasRecordingBeforeBackground = false;
            }, 500); // Small delay to ensure app is fully active
        }

        // Restore normal heartbeat
        this.adjustHeartbeatForForeground();

        this.emit('appForeground', { backgroundDuration });
    }

    handleAppFreeze() {
        console.log('App frozen by system');
        // Similar to background but more aggressive
        this.handleAppBackground();
    }

    handleAppResume() {
        console.log('App resumed from freeze');
        // Check if connection is still alive after system freeze
        setTimeout(() => {
            if (this.connectionState === 'connected') {
                this.testConnectionHealth();
            } else {
                this.reconnect();
            }
        }, 1000);

        this.handleAppForeground();
    }

    handleAppTermination() {
        console.log('App terminating');
        // Clean shutdown
        if (this.isAudioCaptureActive) {
            this.stopAudioCapture();
        }
        this.close(1001, 'App terminating');
    }

    pauseAudioCapture() {
        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            this.mediaRecorder.pause();
            console.log('Audio capture paused for background');
        }
    }

    resumeAudioCapture() {
        if (this.mediaRecorder && this.mediaRecorder.state === 'paused') {
            this.mediaRecorder.resume();
            console.log('Audio capture resumed from background');
        } else if (this.wasRecordingBeforeBackground) {
            // Restart audio capture if it was stopped
            this.startAudioCapture().catch(error => {
                console.error('Failed to resume audio capture:', error);
            });
        }
    }

    adjustHeartbeatForBackground() {
        // Increase heartbeat interval to save battery
        if (this.heartbeatTimer) {
            this.stopHeartbeat();
            this.heartbeatTimer = setInterval(() => {
                if (this.connectionState === 'connected') {
                    this.send({ type: 'ping', timestamp: Date.now(), app_state: 'background' });
                }
            }, 60000); // 1 minute intervals in background
        }
    }

    adjustHeartbeatForForeground() {
        // Restore normal heartbeat
        this.startHeartbeat();
    }

    // Override connection handling for app lifecycle
    handleClose(event) {
        super.handleClose(event);

        // If app is in background and connection closed, be more patient
        if (this.appState === 'background' && !this.isIntentionalDisconnect) {
            console.log('Connection closed while in background, will reconnect when app returns to foreground');
            // Don't aggressively reconnect in background
            return;
        }
    }
}

// React Native Enhancement
class ResilientWebSocketRNWithLifecycle extends ResilientWebSocketRN {
    constructor(baseUrl, authToken, options = {}) {
        super(baseUrl, authToken, options);

        this.appState = 'active';
        this.wasRecordingBeforeBackground = false;
        this.backgroundTime = null;

        this.setupAppLifecycleMonitoring();
    }

    setupAppLifecycleMonitoring() {
        // React Native app state monitoring
        if (typeof require !== 'undefined') {
            try {
                const { AppState } = require('react-native');

                AppState.addEventListener('change', this.handleAppStateChange.bind(this));
                this.appState = AppState.currentState;

                // Call detection (requires additional permissions and libraries)
                this.setupCallDetection();
            } catch (error) {
                console.warn('React Native AppState not available:', error);
            }
        }
    }

    setupCallDetection() {
        // Example using react-native-call-detection
        // npm install react-native-call-detection
        try {
            const CallDetectorManager = require('react-native-call-detection').default;

            this.callDetector = new CallDetectorManager((data) => {
                if (data.isIncoming) {
                    this.handleIncomingCall();
                } else {
                    this.handleCallEnded();
                }
            }, true, () => {}, () => {});

        } catch (error) {
            console.warn('Call detection not available:', error);
        }
    }

    handleAppStateChange(nextAppState) {
        console.log(`App state changed: ${this.appState} → ${nextAppState}`);

        if (this.appState.match(/inactive|background/) && nextAppState === 'active') {
            // App came to foreground
            this.handleAppForeground();
        } else if (this.appState === 'active' && nextAppState.match(/inactive|background/)) {
            // App went to background
            this.handleAppBackground();
        }

        this.appState = nextAppState;
    }

    handleAppBackground() {
        console.log('React Native app went to background');
        this.backgroundTime = Date.now();
        this.wasRecordingBeforeBackground = this.isAudioCaptureActive;

        // Stop audio recording (system requirement on mobile)
        if (this.isAudioCaptureActive) {
            this.stopAudioCapture();
        }

        // Reduce heartbeat frequency
        this.adjustHeartbeatForBackground();

        this.emit('appBackground');
    }

    handleAppForeground() {
        console.log('React Native app returned to foreground');
        const backgroundDuration = this.backgroundTime ? Date.now() - this.backgroundTime : 0;

        // Check connection and resume recording
        setTimeout(() => {
            this.testConnectionHealth();

            if (this.wasRecordingBeforeBackground) {
                this.startAudioCapture().catch(error => {
                    console.error('Failed to resume audio after foreground:', error);
                });
                this.wasRecordingBeforeBackground = false;
            }
        }, 1000);

        this.adjustHeartbeatForForeground();
        this.emit('appForeground', { backgroundDuration });
    }

    handleIncomingCall() {
        console.log('Incoming call detected');
        this.wasRecordingBeforeBackground = this.isAudioCaptureActive;

        if (this.isAudioCaptureActive) {
            this.stopAudioCapture();
        }

        // Send notification to server about call interruption
        this.send({
            type: 'audio_interrupted',
            reason: 'incoming_call',
            timestamp: Date.now()
        });

        this.emit('callStarted');
    }

    handleCallEnded() {
        console.log('Call ended');

        // Wait a moment before resuming to ensure call is fully ended
        setTimeout(() => {
            if (this.wasRecordingBeforeBackground && this.appState === 'active') {
                this.startAudioCapture().catch(error => {
                    console.error('Failed to resume audio after call:', error);
                });
                this.wasRecordingBeforeBackground = false;
            }
        }, 2000);

        this.send({
            type: 'audio_resumed',
            reason: 'call_ended',
            timestamp: Date.now()
        });

        this.emit('callEnded');
    }

    adjustHeartbeatForBackground() {
        if (this.heartbeatTimer) {
            this.stopHeartbeat();
            // Longer intervals in background to comply with mobile OS restrictions
            this.heartbeatTimer = setInterval(() => {
                if (this.connectionState === 'connected') {
                    this.send({ type: 'ping', timestamp: Date.now(), app_state: 'background' });
                }
            }, 120000); // 2 minute intervals
        }
    }

    adjustHeartbeatForForeground() {
        this.startHeartbeat();
    }

    cleanup() {
        super.cleanup();

        if (this.callDetector) {
            this.callDetector.dispose();
        }
    }
}

// Usage Example
const enhancedWS = new ResilientWebSocketWithLifecycle('https://server.com', 'token');

// Listen to app lifecycle events
enhancedWS.on('appBackground', () => {
    console.log('App backgrounded - recording paused, connection maintained');
    showStatus('Recording paused (app in background)');
});

enhancedWS.on('appForeground', ({ backgroundDuration }) => {
    console.log(`App foregrounded after ${backgroundDuration}ms - resuming recording`);
    showStatus('Recording resumed');
});

enhancedWS.on('callStarted', () => {
    showStatus('Call detected - recording paused');
});

enhancedWS.on('callEnded', () => {
    showStatus('Call ended - recording resumed');
});

export { ResilientWebSocketWithLifecycle, ResilientWebSocketRNWithLifecycle };