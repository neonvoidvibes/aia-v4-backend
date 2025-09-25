# Optional server enhancement for client-side reconnection
# Add this endpoint to api_server.py to provide session reattachability status

@app.route('/api/session/<session_id>/status', methods=['GET'])
@supabase_auth_required(agent_required=False)
def get_session_status(user: SupabaseUser, session_id: str):
    """
    Get session reattachment status for client-side reconnection logic.

    This endpoint allows clients to check if a session is still reattachable
    before attempting reconnection, providing better user experience.
    """
    logger.info(f"Session status request for {session_id} from user {user.id}")

    # Check if session exists and is accessible by this user
    if session_id not in active_sessions:
        return jsonify({
            "reattachable": False,
            "reason": "session_not_found",
            "message": "Session not found or has been cleaned up"
        }), 404

    session_data = active_sessions[session_id]
    session_user_id = session_data.get("user_id")

    # Verify user owns this session
    if session_user_id != user.id:
        return jsonify({
            "reattachable": False,
            "reason": "access_denied",
            "message": "Session belongs to a different user"
        }), 403

    # Check session state
    session_state = session_data.get("session_state")
    if not session_state:
        return jsonify({
            "reattachable": False,
            "reason": "no_session_state",
            "message": "Session state not available"
        }), 500

    # Calculate reattachment info
    current_time = datetime.now(timezone.utc)
    reattach_deadline = session_state.reattach_deadline

    if not reattach_deadline:
        # No deadline set - session is active or never disconnected
        has_websocket = session_data.get("websocket_connection") is not None

        return jsonify({
            "reattachable": True,
            "reason": "active_session" if has_websocket else "no_websocket",
            "message": "Session is active" if has_websocket else "Session exists but no active WebSocket",
            "session_id": session_id,
            "is_active": session_data.get("is_active", False),
            "is_finalizing": session_data.get("is_finalizing", False),
            "has_websocket": has_websocket,
            "grace_period_seconds": config.reattach_grace_seconds,
            "seconds_remaining": None
        })

    # Check if grace period has expired
    if session_state.is_reattach_expired():
        return jsonify({
            "reattachable": False,
            "reason": "grace_period_expired",
            "message": "Reattachment grace period has expired",
            "session_id": session_id,
            "expired_at": reattach_deadline.isoformat(),
            "expired_seconds_ago": int((current_time - reattach_deadline).total_seconds())
        })

    # Session is reattachable within grace period
    seconds_remaining = int((reattach_deadline - current_time).total_seconds())

    return jsonify({
        "reattachable": True,
        "reason": "within_grace_period",
        "message": "Session can be reattached within grace period",
        "session_id": session_id,
        "is_active": session_data.get("is_active", False),
        "is_finalizing": session_data.get("is_finalizing", False),
        "has_websocket": session_data.get("websocket_connection") is not None,
        "grace_period_seconds": config.reattach_grace_seconds,
        "seconds_remaining": seconds_remaining,
        "deadline": reattach_deadline.isoformat()
    })


# Enhanced client-side usage example for the status endpoint
"""
JavaScript client enhancement:

class ResilientWebSocket {
    // ... existing code ...

    async checkSessionStatus() {
        if (!this.sessionId) return { reattachable: false };

        try {
            const response = await fetch(`${this.baseUrl}/api/session/${this.sessionId}/status`, {
                headers: {
                    'Authorization': `Bearer ${this.authToken}`
                }
            });

            if (response.ok) {
                return await response.json();
            }

            return { reattachable: false, reason: 'status_check_failed' };
        } catch (error) {
            console.warn('Failed to check session status:', error);
            return { reattachable: false, reason: 'network_error' };
        }
    }

    async connect(agentName, userId) {
        if (this.connectionState === 'connecting') {
            console.log('Connection already in progress');
            return;
        }

        this.setConnectionState('connecting');
        this.isIntentionalDisconnect = false;

        try {
            // Check if we can reattach to existing session
            if (this.sessionId) {
                const status = await this.checkSessionStatus();
                if (status.reattachable) {
                    console.log(`Session ${this.sessionId} is reattachable (${status.seconds_remaining}s remaining)`);
                } else {
                    console.log(`Session ${this.sessionId} not reattachable: ${status.reason}`);
                    this.sessionId = null; // Force new session creation
                }
            }

            // If we don't have a session ID, start new session
            if (!this.sessionId) {
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

    // Enhanced reconnection with status checking
    async reconnect() {
        if (this.isIntentionalDisconnect) return;

        // Check session status before attempting reconnection
        const status = await this.checkSessionStatus();

        if (!status.reattachable) {
            console.log(`Session not reattachable: ${status.reason}, starting new session`);
            this.sessionId = null;
            // Will create new session in connect()
        } else {
            console.log(`Session reattachable with ${status.seconds_remaining}s remaining`);
        }

        console.log(`Reconnection attempt ${this.reconnectAttempts}`);

        try {
            await this.connectWebSocket();
        } catch (error) {
            console.error('Reconnection failed:', error);
            this.scheduleReconnect();
        }
    }
}
"""