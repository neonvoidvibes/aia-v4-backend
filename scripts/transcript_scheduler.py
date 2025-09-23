"""Scheduler for transcript management (no rolling files)."""
import os
import time
import logging
import schedule
import threading
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_session_id() -> str:
    """Create a session ID using current timestamp."""
    return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

def create_event_id() -> str:
    """Create an event ID using current date."""
    return datetime.now().strftime('%Y%m%d')

def start_scheduler(agent_name: str, session_id: str, event_id: str) -> threading.Thread:
    """Start the transcript scheduler in a background thread.
    Note: In the new architecture, no rolling files are written. This scheduler
    now only handles canonical transcript monitoring if needed.

    Args:
        agent_name: Name of the agent
        session_id: Current session ID
        event_id: Current event ID

    Returns:
        The scheduler thread
    """
    def update_all():
        """Monitor canonical transcripts (no rolling files created)."""
        try:
            logger.info("Scheduled transcript monitoring (no rolling file writes)")
            # In the new architecture, windowing happens at read-time.
            # This scheduler could be used for other transcript processing if needed.
            logger.info("Canonical transcripts are processed at read-time via window mode")
        except Exception as e:
            logger.error(f"Error in scheduled update: {e}")

    def run_scheduler():
        """Run the scheduler in a loop."""
        # Schedule monitoring every minute (can be adjusted or removed)
        schedule.every(1).minutes.do(update_all)

        # Run initial update
        update_all()

        logger.info(f"Started transcript scheduler for agent {agent_name}")
        logger.info(f"Session ID: {session_id}")
        logger.info(f"Event ID: {event_id}")
        logger.info("Note: No rolling files are created - windowing happens at read-time")

        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Less frequent polling since no file processing
        except Exception as e:
            logger.error(f"Scheduler error: {e}")

    # Start scheduler in background thread
    scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
    scheduler_thread.start()
    return scheduler_thread

def main():
    """Main function when running scheduler standalone."""
    session_id = create_session_id()
    event_id = create_event_id()
    agent_name = os.getenv('AGENT_NAME', 'river')
    
    # Start scheduler and wait for interrupt
    thread = start_scheduler(agent_name, session_id, event_id)
    try:
        thread.join()
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")

if __name__ == "__main__":
    main()
