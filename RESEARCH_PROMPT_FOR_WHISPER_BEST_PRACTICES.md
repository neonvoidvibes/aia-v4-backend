# Research Prompt: Whisper Live Transcription Best Practices

## Task
Document industry best practices for real-time Whisper transcription to solve persistent hallucination issues.

## System Architecture
- **Backend**: Python with WebRTC VAD
- **Frontend**: React/Next.js with WebSocket audio streaming
- **Storage**: AWS S3 for transcript chunks
- **Processing**: Audio segments transcribed every 8-15 seconds

## Key Research Areas

### 1. Whisper API Configuration
- Optimal parameters for live transcription (temperature, thresholds, etc.)
- **Context/Prompt usage**: Should we use prompts or not? (Critical - we're having contamination issues)
- Language detection vs. explicit language setting
- Segment duration recommendations

### 2. VAD Pipeline
- WebRTC VAD aggressiveness levels for real-time use
- Audio preprocessing before Whisper
- Silence detection and filtering
- Optimal chunk sizes and overlap

### 3. Hallucination Prevention
- **Most Important**: How to prevent context contamination between segments
- Industry approaches: zero-context vs. context management
- Post-processing filters and validation
- Quality thresholds and confidence scoring

### 4. Production Implementation
- Error handling and retry strategies
- Performance optimization for real-time processing
- Memory management for long sessions
- S3 streaming append patterns

## Specific Questions to Answer
1. Do production systems use Whisper prompts for live transcription?
2. How do major platforms (Google, Azure, AWS) handle segment isolation?
3. What are the proven strategies to prevent transcription contamination?
4. What VAD settings work best with Whisper's expectations?

## Focus
Emphasize **hallucination reduction** and **segment isolation** - these are our primary challenges.
