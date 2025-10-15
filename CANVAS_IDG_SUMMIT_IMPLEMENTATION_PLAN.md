# Canvas View Enhancement - IDG Summit Demo Implementation Plan

**Goal:** Demo the canvas agent at the IDG Summit track about "leading in the age of AI"

**Date:** 2025-10-15

---

## Background & Requirements

### Current Limitations

1. **Single source limitation**: Canvas currently only reads from latest/historical transcripts
2. **No additional document support**: Cannot upload other files for canvas analysis
3. **Canvas agent lacks raw context**: Only reads pre-analyzed MLP docs, missing raw transcript access
4. **No RAG retrieval**: Canvas and MLP agents don't use Pinecone for contextual memory
5. **Flat storage structure**: Analysis docs stored directly in `_canvas/` without organization
6. **Limited version history**: Only "current" and "previous" versions tracked

### IDG Summit Requirements

1. **Multiple sources**: Upload files beyond transcripts (reports, notes, documents)
2. **Canvas agent reads raw + analysis**: Access to both unprocessed data and MLP insights
3. **MLP agents read additional docs**: Analyze all available sources, not just transcripts
4. **Source attribution**: Clear differentiation of insights by source
5. **RAG-enabled agents**: Contextual awareness from Pinecone memory
6. **Better version tracking**: Organized history of analyses with dates

---

## Current State Review

### Folder Structure (Current)
```
organizations/{org}/agents/{agent_name}/
  _canvas/
    0000_mirror.md              # Current mirror analysis
    0000_mirror_previous.md     # Previous mirror analysis
    0000_lens.md                # Current lens analysis
    0000_lens_previous.md       # Previous lens analysis
    0000_portal.md              # Current portal analysis
    0000_portal_previous.md     # Previous portal analysis
  events/{event_id}/
    transcripts/
      {timestamp}.txt
```

### Data Flow (Current)

**MLP Analysis Generation:**
```
Settings Toggles → get_transcript_content_for_analysis() → Raw Transcripts
                                                          ↓
                                          Groq LLM (gpt-oss-120b)
                                                          ↓
                                            Analysis Markdown Docs
                                                          ↓
                                            S3 (_canvas/*.md)
```

**Canvas Agent Response:**
```
User Query + History → Canvas Agent (Claude Sonnet 4.5)
                              ↓
                      Reads: ALL 3 MLP docs (mirror/lens/portal)
                            + Objective Function
                            + Agent Prompt
                            + Conversation History
                              ↓
                      Responds (no RAG, no raw transcripts)
```

### Settings Toggles (Current)

**Respected by MLP agents:**
- `transcript_listen_mode`: 'none' | 'latest' | 'some' | 'all'
- `groups_read_mode`: 'none' | 'latest' | 'all' | 'breakout'
- `individual_raw_transcript_toggle_states`: Dict[s3_key, bool]

**NOT respected by Canvas agent:**
- Canvas doesn't read transcripts directly
- Canvas only reads pre-analyzed MLP docs

### RAG Integration (Current)

**Main Chat Agent:**
- ✅ Uses `RetrievalHandler` with Pinecone
- ✅ Tiered retrieval (t0, t_personal, t1, t2, t3)
- ✅ Query transformation + time-decay scoring

**Canvas Agent:**
- ❌ No RAG retrieval

**MLP Agents:**
- ❌ No RAG retrieval

---

## Proposed Changes

### 1. Restructure `_canvas/` Folder

**New Structure:**
```
organizations/{org}/agents/{agent_name}/
  _canvas/
    mlp/
      mlp-latest/
        0000_mirror.md          # Current analysis
        0000_lens.md
        0000_portal.md
      mlp-previous/
        0000_mirror_20251014.md # Dated previous version
        0000_lens_20251014.md
        0000_portal_20251014.md
      mlp-history/
        0000_mirror_20251010.md # Older versions
        0000_mirror_20251012.md
        0000_lens_20251010.md
        ...
    docs/
      uploaded_report.pdf       # Additional source documents
      meeting_notes.txt
      strategy_doc.md
      ...
```

**Implementation Files:**
- `utils/canvas_analysis_agents.py`: Update S3 key paths
- New functions: `archive_analysis_to_history()`, `list_analysis_history()`

---

### 2. Canvas Agent Reads Raw Transcripts

**Current:** Canvas agent only reads MLP analysis docs

**Proposed:** Canvas agent reads:
1. MLP analysis docs (all 3 modes, current + previous)
2. Raw transcript content (respecting Settings toggles)
3. Additional docs from `_canvas/docs/`
4. RAG-retrieved context from Pinecone

**Benefits:**
- Canvas can verify analysis against raw data
- Canvas can answer questions about specific quotes
- Canvas has full context for nuanced responses

**Implementation:**
- Add `get_transcript_content_for_analysis()` call in `canvas_routes.py`
- Add transcript section to canvas system prompt
- Respect `transcript_listen_mode` and `groups_read_mode`

---

### 3. MLP Agents Read Additional Docs

**Current:** MLP agents only read transcripts

**Proposed:** MLP agents read:
1. Transcripts (as before, respecting toggles)
2. Additional docs from `_canvas/docs/`
3. RAG-retrieved context from Pinecone

**Implementation:**
- Add `list_and_read_canvas_docs()` function
- Inject docs into MLP agent prompts
- Update analysis instructions to handle diverse sources

---

### 4. Enhanced Source Attribution

**Current Implementation:**
- Transcripts labeled: `"--- START Transcript Source: {name} ---"`
- Groups labeled: `"--- Group Event: {gid} ---"`
- Instructions tell agents to differentiate sources

**Proposed Enhancements:**

**A. Standardized Source Headers:**
```markdown
=== SOURCE: Transcript - 2025-10-15 14:30 ===
[content]
=== END SOURCE ===

=== SOURCE: Group Event - breakout_1 ===
[content]
=== END SOURCE ===

=== SOURCE: Document - strategy_2025.pdf ===
[content]
=== END SOURCE ===
```

**B. Metadata Table in Analysis:**
```markdown
# Analysis Metadata
- Sources Analyzed: 3 transcripts, 2 uploaded docs, 1 group event
- Transcript Listen Mode: latest
- Groups Read Mode: breakout (2 events)
- Generated: 2025-10-15 14:45:23 UTC

---
[analysis content]
```

**C. Inline Attribution:**
Strengthen MLP agent instructions to include source attribution naturally:
```
"In breakout_1, participants focused on X..."
"The uploaded strategy document emphasizes Y..."
"Across all three transcript sources, the pattern shows Z..."
```

---

### 5. RAG Integration for Canvas & MLP Agents

**Add Pinecone retrieval to both agent types:**

**MLP Agents (Analysis Generation):**
```python
# In get_or_generate_analysis_doc()
retriever = RetrievalHandler(
    index_name="river",
    agent_name=agent_name,
    event_id=event_id,
    anthropic_api_key=get_api_key(agent_name, 'anthropic'),
    openai_api_key=get_api_key(agent_name, 'openai'),
    event_type=event_type,
    personal_event_id=personal_event_id,
    include_t3=True if event_id == "0000" else False,
)

# Query based on transcript content
query = f"Relevant context for analyzing: {transcript_summary[:200]}"
rag_docs = retriever.get_relevant_context_tiered(query, tier_caps=[4,8,6,6,4])

# Add to analysis prompt
rag_context = format_rag_context(rag_docs)
system_prompt += f"\n\n=== RETRIEVED CONTEXT ===\n{rag_context}\n=== END CONTEXT ==="
```

**Canvas Agent (Response Generation):**
```python
# In handle_canvas_stream()
retriever = RetrievalHandler(
    index_name="river",
    agent_name=agent_name,
    event_id=event_id,
    anthropic_api_key=agent_anthropic_key,
    openai_api_key=get_api_key(agent_name, 'openai'),
    event_type='shared',
    include_t3=True,
)

# Query based on user message
rag_docs = retriever.get_relevant_context_tiered(
    query=transcript_text,
    tier_caps=[4,8,6,6,4]
)

# Add to system prompt (before analyses)
rag_context = format_rag_context(rag_docs)
system_prompt += f"\n\n=== RETRIEVED MEMORY ===\n{rag_context}\n=== END MEMORY ==="

# Reinforce memories after response
retriever.reinforce_memories(rag_docs)
```

**Benefits:**
- Agents can reference past conversations
- Agents aware of foundational documents
- Agents have cross-event context when appropriate
- Memory reinforcement improves relevance over time

---

## Implementation Steps

### Phase 1: Folder Restructure (2-3 hours)

**Files to modify:**
- `utils/canvas_analysis_agents.py`

**Tasks:**
1. Update `get_s3_analysis_doc_key()`:
   ```python
   def get_s3_analysis_doc_key(agent_name: str, event_id: str, mode: str, version: str = 'latest') -> str:
       """
       Get S3 key for analysis document.

       Args:
           version: 'latest' | 'previous' | 'YYYYMMDD' (for history)
       """
       if version == 'latest':
           return f"organizations/{CANVAS_ANALYSIS_ORG}/agents/{agent_name}/_canvas/mlp/mlp-latest/{event_id}_{mode}.md"
       elif version == 'previous':
           date_str = datetime.now(timezone.utc).strftime('%Y%m%d')
           return f"organizations/{CANVAS_ANALYSIS_ORG}/agents/{agent_name}/_canvas/mlp/mlp-previous/{event_id}_{mode}_{date_str}.md"
       else:  # version is a date string YYYYMMDD
           return f"organizations/{CANVAS_ANALYSIS_ORG}/agents/{agent_name}/_canvas/mlp/mlp-history/{event_id}_{mode}_{version}.md"
   ```

2. Update `save_analysis_doc_to_s3()`:
   ```python
   def save_analysis_doc_to_s3(agent_name: str, event_id: str, mode: str, content: str) -> bool:
       """
       Save analysis document to S3.
       Flow: latest → previous (with date) → history (if previous exists)
       """
       try:
           s3_client = get_s3_client()
           date_str = datetime.now(timezone.utc).strftime('%Y%m%d')

           latest_key = get_s3_analysis_doc_key(agent_name, event_id, mode, version='latest')
           previous_key = get_s3_analysis_doc_key(agent_name, event_id, mode, version='previous')

           # Step 1: If previous exists, move it to history
           try:
               s3_client.head_object(Bucket=CANVAS_ANALYSIS_BUCKET, Key=previous_key)
               # Previous exists - extract its date and move to history
               previous_obj = s3_client.get_object(Bucket=CANVAS_ANALYSIS_BUCKET, Key=previous_key)
               previous_date = previous_obj['Metadata'].get('date', date_str)
               history_key = get_s3_analysis_doc_key(agent_name, event_id, mode, version=previous_date)

               s3_client.copy_object(
                   Bucket=CANVAS_ANALYSIS_BUCKET,
                   CopySource={'Bucket': CANVAS_ANALYSIS_BUCKET, 'Key': previous_key},
                   Key=history_key
               )
               logger.info(f"Archived previous {mode} to history: {history_key}")
           except s3_client.exceptions.NoSuchKey:
               logger.info(f"No previous {mode} to archive")

           # Step 2: Move current latest to previous (with today's date)
           try:
               s3_client.copy_object(
                   Bucket=CANVAS_ANALYSIS_BUCKET,
                   CopySource={'Bucket': CANVAS_ANALYSIS_BUCKET, 'Key': latest_key},
                   Key=previous_key,
                   Metadata={'date': date_str}
               )
               logger.info(f"Moved latest {mode} to previous: {previous_key}")
           except s3_client.exceptions.NoSuchKey:
               logger.info(f"No existing latest {mode}")

           # Step 3: Save new content as latest
           s3_client.put_object(
               Bucket=CANVAS_ANALYSIS_BUCKET,
               Key=latest_key,
               Body=content.encode('utf-8'),
               ContentType='text/markdown',
               Metadata={
                   'timestamp': datetime.now(timezone.utc).isoformat(),
                   'date': date_str,
                   'mode': mode,
                   'agent': agent_name,
                   'event': event_id
               }
           )

           logger.info(f"Saved {mode} analysis to latest: {latest_key}")
           return True
       except Exception as e:
           logger.error(f"Error saving analysis to S3: {e}", exc_info=True)
           return False
   ```

3. Add new functions:
   ```python
   def list_canvas_docs(agent_name: str) -> List[Dict[str, Any]]:
       """List all documents in _canvas/docs/ folder."""
       s3_client = get_s3_client()
       prefix = f"organizations/{CANVAS_ANALYSIS_ORG}/agents/{agent_name}/_canvas/docs/"

       docs = []
       try:
           response = s3_client.list_objects_v2(Bucket=CANVAS_ANALYSIS_BUCKET, Prefix=prefix)
           if 'Contents' in response:
               for obj in response['Contents']:
                   if obj['Key'] != prefix:  # Skip the folder itself
                       docs.append({
                           'key': obj['Key'],
                           'filename': os.path.basename(obj['Key']),
                           'size': obj['Size'],
                           'last_modified': obj['LastModified']
                       })
       except Exception as e:
           logger.error(f"Error listing canvas docs: {e}")

       return docs

   def read_canvas_doc(s3_key: str) -> Optional[str]:
       """Read a document from _canvas/docs/."""
       try:
           s3_client = get_s3_client()
           response = s3_client.get_object(Bucket=CANVAS_ANALYSIS_BUCKET, Key=s3_key)

           # Handle different file types
           content_type = response['ContentType']
           if 'text' in content_type or 'markdown' in content_type:
               return response['Body'].read().decode('utf-8')
           elif 'pdf' in content_type:
               # TODO: Add PDF extraction (could use PyPDF2 or similar)
               logger.warning("PDF extraction not yet implemented")
               return f"[PDF Document: {os.path.basename(s3_key)}]"
           else:
               logger.warning(f"Unsupported content type: {content_type}")
               return f"[Binary Document: {os.path.basename(s3_key)}]"
       except Exception as e:
           logger.error(f"Error reading canvas doc {s3_key}: {e}")
           return None

   def get_all_canvas_source_content(
       agent_name: str,
       event_id: str,
       transcript_listen_mode: str = 'latest',
       groups_read_mode: str = 'none',
       individual_raw_transcript_toggle_states: Optional[Dict[str, bool]] = None
   ) -> str:
       """
       Get all source content for canvas/MLP agents:
       1. Transcripts (based on settings)
       2. Additional docs from _canvas/docs/
       """
       parts = []

       # 1. Get transcripts (existing function)
       transcript_content = get_transcript_content_for_analysis(
           agent_name=agent_name,
           event_id=event_id,
           transcript_listen_mode=transcript_listen_mode,
           groups_read_mode=groups_read_mode,
           individual_raw_transcript_toggle_states=individual_raw_transcript_toggle_states
       )
       if transcript_content:
           parts.append(transcript_content)

       # 2. Get additional docs
       canvas_docs = list_canvas_docs(agent_name)
       if canvas_docs:
           docs_parts = []
           for doc in canvas_docs:
               content = read_canvas_doc(doc['key'])
               if content:
                   docs_parts.append(
                       f"=== SOURCE: Document - {doc['filename']} ===\n"
                       f"{content}\n"
                       f"=== END SOURCE: {doc['filename']} ==="
                   )

           if docs_parts:
               combined_docs = "\n\n".join(docs_parts)
               parts.append(f"=== ADDITIONAL DOCUMENTS ===\n{combined_docs}\n=== END ADDITIONAL DOCUMENTS ===")

       return "\n\n".join(parts) if parts else ""
   ```

4. Update `get_transcript_content_for_analysis()` to use standardized headers:
   - Change `"--- START Transcript Source: {name} ---"` to `"=== SOURCE: Transcript - {name} ==="`
   - Change `"--- Group Event: {gid} ---"` to `"=== SOURCE: Group Event - {gid} ==="`
   - Add `"=== END SOURCE ===" ` markers

5. Migration: Create script to move existing analysis docs to new structure
   ```python
   # scripts/migrate_canvas_structure.py
   def migrate_canvas_structure():
       """Move existing _canvas/*.md to _canvas/mlp/mlp-latest/"""
       # Implementation details...
   ```

**Testing:**
- Verify new folder structure in S3
- Confirm analysis docs save to correct locations
- Test version history (latest → previous → history flow)
- Verify migration script moves old docs correctly

---

### Phase 2: Canvas Agent Reads Raw Transcripts (2-3 hours)

**Files to modify:**
- `routes/canvas_routes.py`

**Changes:**

1. In `handle_canvas_stream()`, add transcript reading:
   ```python
   # After loading analyses (around line 260)

   # Get raw transcript content based on Settings toggles
   raw_transcript_content = None
   try:
       raw_transcript_content = get_all_canvas_source_content(
           agent_name=agent_name,
           event_id=event_id,
           transcript_listen_mode=transcript_listen_mode,
           groups_read_mode=groups_read_mode,
           individual_raw_transcript_toggle_states=individual_raw_transcript_toggle_states
       )
       if raw_transcript_content:
           logger.info(f"Canvas: Loaded raw source content ({len(raw_transcript_content)} chars)")
       else:
           logger.info("Canvas: No raw source content available")
   except Exception as source_err:
       logger.error(f"Error loading source content for canvas: {source_err}", exc_info=True)
   ```

2. Update system prompt assembly (around line 290):
   ```python
   # Current order:
   # 1. Objective → 2. Agent → 3. Canvas Base + Depth → 4. Previous Analyses → 5. Current Analyses → 6. Mode Emphasis → 7. Time

   # New order (add raw sources before analyses):
   # 1. Objective → 2. Agent → 3. Canvas Base + Depth → 4. Raw Sources → 5. Previous Analyses → 6. Current Analyses → 7. Mode Emphasis → 8. Time

   # After canvas_base_and_depth, add:
   if raw_transcript_content:
       system_prompt += f"\n\n=== RAW SOURCES ===\n"
       system_prompt += f"You have access to the raw source content that was used to generate the analyses below.\n"
       system_prompt += f"Use this to verify analysis insights, find specific quotes, or surface details not captured in the analysis.\n\n"
       system_prompt += raw_transcript_content
       system_prompt += f"\n=== END RAW SOURCES ==="
   ```

3. Update prompt logging to include raw sources:
   ```python
   # Around line 388
   prompt_components = []
   if objective_function:
       prompt_components.append("objective")
   if agent_specific:
       prompt_components.append("agent")
   prompt_components.extend(["base", "depth"])

   # Add raw sources indicator
   if raw_transcript_content:
       prompt_components.append(f"raw_sources({len(raw_transcript_content)} chars)")

   # ... rest of components
   ```

4. Update Canvas Base prompt to mention raw sources:
   ```python
   # In get_canvas_base_and_depth_prompt() around line 85
   canvas_base = """=== CANVAS BASE ===
   ...

   KNOWLEDGE SOURCES:
   - You draw on OBJECTIVE FUNCTION, AGENT CONTEXT, RAW SOURCES, and ANALYSIS DOCUMENTS
   - RAW SOURCES contain unprocessed transcripts and documents
   - ANALYSIS DOCUMENTS provide interpreted insights (mirror/lens/portal)
   - Synthesize raw + analyzed naturally - never mention them explicitly
   - If no analysis exists, respond based on raw sources using your general understanding

   SOURCE DIFFERENTIATION:
   - Raw sources and analyses may contain insights from MULTIPLE SOURCES
   - When responding, preserve source differentiation if it matters to the question
   ...
   ```

**Testing:**
- Canvas agent receives raw transcript content
- Canvas can answer questions about specific quotes
- Canvas can verify/expand on analysis insights
- Settings toggles properly filter what canvas sees

---

### Phase 3: MLP Agents Read Additional Docs (1-2 hours)

**Files to modify:**
- `utils/canvas_analysis_agents.py`

**Changes:**

1. Update `run_analysis_agent()` to use new source function:
   ```python
   # Around line 773 in get_or_generate_analysis_doc()

   # OLD:
   transcript_content = get_transcript_content_for_analysis(...)

   # NEW:
   all_source_content = get_all_canvas_source_content(
       agent_name=agent_name,
       event_id=event_id,
       transcript_listen_mode=transcript_listen_mode,
       groups_read_mode=groups_read_mode,
       individual_raw_transcript_toggle_states=individual_raw_transcript_toggle_states
   )
   ```

2. Update MLP agent prompts to mention additional docs:
   ```python
   # In get_analysis_agent_prompt() around line 72

   transcript_section = f"""
   === SOURCE DATA ===
   {all_source_content}
   === END SOURCE DATA ===

   CRITICAL: The source data above may contain:
   1. TRANSCRIPT DATA from multiple events/groups/times
   2. ADDITIONAL DOCUMENTS (reports, notes, uploaded files)

   Each source is clearly marked with headers like "=== SOURCE: Type - Name ===" and "=== END SOURCE ===".

   SOURCE DIFFERENTIATION REQUIREMENT:
   When analyzing data from multiple sources:
   - Identify the TYPE of each source (transcript vs document)
   - Identify patterns WITHIN each source
   - Compare/contrast patterns BETWEEN sources when relevant
   - Always specify which source(s) your observations come from
   - Use source labels naturally: "In the uploaded strategy document..." or "In breakout_1 transcript..."
   - If all sources show the same pattern: "Across all sources..."
   - Preserve the differentiation signal - don't blend sources into homogeneous "the data"

   If only ONE source is present, you may refer to "the conversation" or "the document" naturally.
   ```

3. Add source metadata to analysis output:
   ```python
   # In analysis_tasks prompts (mirror/lens/portal), add at the beginning:

   """
   === ANALYSIS METADATA ===
   FIRST, document what sources you're analyzing:
   - List each source with its type and identifier
   - Note which Settings were used (Listen mode, Groups mode)
   - Record generation timestamp

   Format:
   ```
   Sources Analyzed:
   - Transcript: 2025-10-15_14-30.txt (event 0000)
   - Document: strategy_2025.pdf
   - Group Transcripts: 2 breakout events (breakout_1, breakout_2)

   Settings:
   - Transcript Listen: latest
   - Groups Read: breakout

   Generated: 2025-10-15 14:45:23 UTC
   ```
   === END METADATA ===

   [Then proceed with your analysis...]
   """
   ```

**Testing:**
- Upload test document to `_canvas/docs/`
- Trigger MLP analysis refresh
- Verify analysis includes insights from uploaded doc
- Check metadata section lists all sources
- Confirm source attribution in analysis body

---

### Phase 4: RAG Integration (3-4 hours)

**Files to modify:**
- `utils/canvas_analysis_agents.py`
- `routes/canvas_routes.py`

**Part A: MLP Agents RAG (canvas_analysis_agents.py)**

1. Import RetrievalHandler:
   ```python
   # At top of file
   from .retrieval_handler import RetrievalHandler
   from .api_key_manager import get_api_key
   ```

2. Update `run_analysis_agent()` signature and implementation:
   ```python
   def run_analysis_agent(
       agent_name: str,
       event_id: str,
       mode: str,
       all_source_content: str,
       event_type: str = 'shared',
       personal_layer: Optional[str] = None,
       personal_event_id: Optional[str] = None,
       groq_api_key: Optional[str] = None,
       enable_rag: bool = True  # NEW parameter
   ) -> Optional[str]:
       """Run analysis agent with optional RAG retrieval."""

       logger.info(f"Running {mode} analysis agent for {agent_name}/{event_id} (RAG: {enable_rag})")

       # NEW: RAG retrieval
       rag_context = ""
       if enable_rag:
           try:
               # Create summary of source content for RAG query
               summary = all_source_content[:500] if len(all_source_content) > 500 else all_source_content
               rag_query = f"Context for analyzing: {summary}"

               retriever = RetrievalHandler(
                   index_name="river",
                   agent_name=agent_name,
                   event_id=event_id,
                   anthropic_api_key=get_api_key(agent_name, 'anthropic'),
                   openai_api_key=get_api_key(agent_name, 'openai'),
                   event_type=event_type,
                   personal_event_id=personal_event_id,
                   include_t3=(event_id == "0000"),
               )

               rag_docs = retriever.get_relevant_context_tiered(
                   query=rag_query,
                   tier_caps=[4, 8, 6, 6, 4],  # Modest caps for analysis
                   include_t3=(event_id == "0000")
               )

               if rag_docs:
                   rag_parts = []
                   for doc in rag_docs:
                       source_label = doc.metadata.get('source_label', '')
                       filename = doc.metadata.get('filename', 'unknown')
                       tier = doc.metadata.get('retrieval_tier', 'unknown')
                       score = doc.metadata.get('score', 0)

                       rag_parts.append(
                           f"--- Retrieved Context (tier: {tier}, score: {score:.3f}, source: {filename}) ---\n"
                           f"{doc.page_content}\n"
                           f"--- End Retrieved Context ---"
                       )

                   rag_context = "\n\n".join(rag_parts)
                   logger.info(f"Retrieved {len(rag_docs)} context docs for {mode} analysis")
               else:
                   logger.info(f"No RAG context retrieved for {mode} analysis")

           except Exception as rag_err:
               logger.error(f"RAG retrieval failed for {mode} analysis: {rag_err}", exc_info=True)

       # Build analysis prompt (existing logic)
       system_prompt = get_analysis_agent_prompt(
           agent_name=agent_name,
           event_id=event_id,
           mode=mode,
           event_type=event_type,
           personal_layer=personal_layer,
           personal_event_id=personal_event_id,
           source_content=all_source_content,
           rag_context=rag_context  # NEW: pass RAG context
       )

       # ... rest of function
   ```

3. Update `get_analysis_agent_prompt()` to accept and inject RAG context:
   ```python
   def get_analysis_agent_prompt(
       agent_name: str,
       event_id: str,
       mode: str,
       event_type: str,
       personal_layer: Optional[str],
       personal_event_id: Optional[str],
       source_content: str,
       rag_context: str = ""  # NEW parameter
   ) -> str:
       """Build full taxonomy prompt + RAG context + mode-specific instructions."""

       # 1. Build full main chat agent taxonomy (existing)
       base_prompt = prompt_builder(...)

       # 2. Add RAG context if available (NEW)
       rag_section = ""
       if rag_context:
           rag_section = f"""
   === RETRIEVED MEMORY CONTEXT ===
   The following context was retrieved from the agent's long-term memory based on the source content.
   Use this to inform your analysis with relevant historical knowledge, foundational documents, or related conversations.

   {rag_context}
   === END RETRIEVED MEMORY ===

   """

       # 3. Add source content section (existing, with updated headers)
       source_section = f"""
   === SOURCE DATA ===
   {source_content}
   === END SOURCE DATA ===
   ...
   """

       # 4. Add mode-specific task (existing)
       analysis_task = analysis_tasks[mode]

       # Combine: base → RAG → sources → task
       return base_prompt + "\n\n" + rag_section + source_section + "\n\n" + analysis_task
   ```

**Part B: Canvas Agent RAG (canvas_routes.py)**

1. In `handle_canvas_stream()`, add RAG retrieval:
   ```python
   # Around line 220, after loading analyses and before building system prompt

   # RAG retrieval for canvas agent
   rag_context = None
   try:
       retriever = RetrievalHandler(
           index_name="river",
           agent_name=agent_name,
           event_id=event_id,
           anthropic_api_key=agent_anthropic_key,
           openai_api_key=get_api_key(agent_name, 'openai'),
           event_type='shared',
           include_t3=True,
       )

       # Query based on user message
       rag_docs = retriever.get_relevant_context_tiered(
           query=transcript_text,
           tier_caps=[4, 8, 6, 6, 4],
           include_t3=True
       )

       if rag_docs:
           rag_parts = []
           for doc in rag_docs:
               source_label = doc.metadata.get('source_label', '')
               filename = doc.metadata.get('filename', 'unknown')
               tier = doc.metadata.get('retrieval_tier', 'unknown')
               score = doc.metadata.get('score', 0)
               age_display = doc.metadata.get('age_display', 'unknown age')

               rag_parts.append(
                   f"[{tier}] {filename} (score: {score:.3f}, age: {age_display})\n{doc.page_content}"
               )

           rag_context = "\n\n---\n\n".join(rag_parts)
           logger.info(f"Canvas: Retrieved {len(rag_docs)} context docs")
       else:
           logger.info("Canvas: No RAG context retrieved")

   except Exception as rag_err:
       logger.error(f"Canvas RAG retrieval error: {rag_err}", exc_info=True)
   ```

2. Add RAG context to system prompt:
   ```python
   # After objective function and agent context, before canvas base

   if rag_context:
       system_prompt += f"\n\n=== RETRIEVED MEMORY ===\n"
       system_prompt += "The following context was retrieved from long-term memory based on your current query.\n"
       system_prompt += "Draw on this when relevant to provide informed, contextual responses.\n\n"
       system_prompt += rag_context
       system_prompt += f"\n=== END RETRIEVED MEMORY ==="
   ```

3. Reinforce memories after streaming completes:
   ```python
   # After streaming ends (around line 416)

   # Send completion signal
   yield f"data: {json.dumps({'done': True})}\n\n"

   # Reinforce retrieved memories (NEW)
   if rag_docs:
       try:
           retriever.reinforce_memories(rag_docs)
           logger.info(f"Canvas: Reinforced {len(rag_docs)} memories")
       except Exception as reinforce_err:
           logger.error(f"Canvas memory reinforcement error: {reinforce_err}")

   logger.info(f"Canvas stream completed successfully for agent {agent_name}")
   ```

4. Update prompt logging:
   ```python
   # Around line 388, add to prompt_components
   if rag_context:
       prompt_components.append(f"rag({len(rag_docs)}docs)")
   ```

**Testing:**
- Verify MLP agents retrieve relevant context from Pinecone
- Verify Canvas agent retrieves context based on user query
- Check memory reinforcement after canvas responses
- Test with queries that should match foundational docs
- Test cross-event retrieval in shared space (event 0000)

---

### Phase 5: Enhanced Source Attribution (1-2 hours)

**Files to modify:**
- `utils/canvas_analysis_agents.py`

**Changes:**

1. Update source headers in `get_all_canvas_source_content()`:
   ```python
   # Standardize all source headers

   # For transcripts:
   f"=== SOURCE: Transcript - {filename} (Event: {event_id}, {timestamp}) ===\n"
   f"{content}\n"
   f"=== END SOURCE: Transcript - {filename} ==="

   # For documents:
   f"=== SOURCE: Document - {filename} (Type: {file_type}, Size: {size}KB) ===\n"
   f"{content}\n"
   f"=== END SOURCE: Document - {filename} ==="

   # For group/breakout events:
   f"=== SOURCE: Group Event - {event_id} (Type: {event_type}) ===\n"
   f"{content}\n"
   f"=== END SOURCE: Group Event - {event_id} ==="
   ```

2. Add metadata section to MLP prompts:
   ```python
   # In analysis_tasks dict for each mode, add instruction:

   """
   YOUR ANALYSIS MUST BEGIN WITH A METADATA SECTION:

   # Analysis Metadata

   **Sources Analyzed:**
   [List each source you analyzed with type and identifier]

   **Settings:**
   - Transcript Listen: [mode]
   - Groups Read: [mode]

   **Generated:** [ISO timestamp]

   ---

   [Then proceed with your mode-specific analysis...]
   """
   ```

3. Strengthen inline attribution instructions:
   ```python
   # Update SOURCE DIFFERENTIATION section in transcript_section:

   """
   SOURCE DIFFERENTIATION REQUIREMENT:

   1. IDENTIFY source types:
      - Transcripts: Real-time conversations
      - Documents: Uploaded files (reports, notes, etc.)
      - Group Events: Multi-participant recordings

   2. ATTRIBUTE naturally in your narrative:
      ✓ Good: "In breakout_1, participants focused on X, while the strategy document emphasizes Y"
      ✓ Good: "The uploaded Q3 report shows Z, confirmed by comments in the main transcript"
      ✗ Bad: "The group focused on X" (which group? which source?)

   3. PATTERNS across sources:
      - Within-source: "Within breakout_1, three themes emerged..."
      - Cross-source: "Across all transcripts and documents, the consistent pattern is..."
      - Contrasts: "While breakout_1 emphasized X, breakout_2 and the strategy doc both highlight Y"

   4. PRESERVE differentiation throughout:
      - Don't blend multiple sources into vague "the group" or "the data"
      - Keep source labels visible in your analysis prose
      - Only generalize when pattern is truly universal

   If only ONE source exists, natural language ("the conversation", "the document") is fine.
   """
   ```

**Testing:**
- Generate analyses with multiple sources
- Verify metadata section appears at top
- Check inline attribution mentions specific sources
- Confirm patterns are attributed to specific sources
- Test with single source (should not over-attribute)

---

## Testing & Validation

### Test Scenarios

1. **Single Transcript Source:**
   - Toggle: Listen = latest, Groups = none
   - Expected: Simple analysis of one transcript

2. **Multiple Transcript Sources:**
   - Toggle: Listen = latest, Groups = breakout (2 groups)
   - Expected: Analysis differentiates between main event + 2 breakouts

3. **Transcripts + Uploaded Document:**
   - Upload PDF report to `_canvas/docs/`
   - Toggle: Listen = latest, Groups = none
   - Expected: Analysis synthesizes transcript + document

4. **Full Mixed Sources:**
   - Upload 2 documents
   - Toggle: Listen = all, Groups = breakout (3 groups)
   - Expected: Analysis of 5+ transcripts + 2 docs with clear attribution

5. **RAG Context:**
   - Query about topic covered in old chat or foundational doc
   - Expected: Canvas/MLP cite retrieved context

6. **Version History:**
   - Generate analysis 3 times
   - Expected: latest, previous (dated), history (older dates)

### Validation Checklist

- [ ] New folder structure exists in S3
- [ ] Old analyses migrated to new structure
- [ ] Canvas agent reads raw transcripts
- [ ] Canvas agent uses RAG retrieval
- [ ] Canvas agent reinforces memories
- [ ] MLP agents read additional docs from `_canvas/docs/`
- [ ] MLP agents use RAG retrieval
- [ ] MLP analyses include metadata section
- [ ] MLP analyses have clear source attribution
- [ ] Analysis versioning works (latest → previous → history)
- [ ] Settings toggles respected by canvas agent
- [ ] Breakout mode reads ALL transcripts per group
- [ ] Source headers standardized across all content types
- [ ] UI shows raw sources in canvas (optional enhancement)

---

## Rollout Plan

### Pre-Deployment

1. **Backup existing analyses:**
   ```bash
   aws s3 sync s3://bucket/organizations/river/agents/ ./backup-analyses/
   ```

2. **Run migration script:**
   ```bash
   python scripts/migrate_canvas_structure.py --agent river --dry-run
   python scripts/migrate_canvas_structure.py --agent river
   ```

3. **Test on staging:**
   - Deploy to staging environment
   - Run test scenarios
   - Verify S3 structure
   - Check logs for errors

### Deployment

1. **Deploy backend changes:**
   - Update `utils/canvas_analysis_agents.py`
   - Update `routes/canvas_routes.py`
   - Restart API server

2. **Verify deployment:**
   - Check health endpoint
   - Trigger canvas analysis refresh
   - Verify new folder structure
   - Test canvas responses

3. **Monitor:**
   - Watch CloudWatch logs
   - Check Sentry for errors
   - Monitor S3 write patterns
   - Track API latency

### Post-Deployment

1. **IDG Summit prep:**
   - Upload demo documents to `_canvas/docs/`
   - Generate fresh analyses
   - Test canvas responses
   - Prepare demo scenarios

2. **Documentation:**
   - Update README with new folder structure
   - Document file upload process
   - Add RAG integration notes
   - Update API documentation

---

## Performance Considerations

### Latency Impact

**MLP Analysis Generation:**
- Current: ~10-15 seconds per mode
- With RAG: +2-3 seconds for retrieval
- With additional docs: +1-2 seconds for reading
- **Total: ~13-20 seconds per mode** (acceptable for background job)

**Canvas Agent Response:**
- Current: ~2-4 seconds TTFB (time to first byte)
- With RAG: +1-2 seconds
- With raw transcripts: +0.5-1 second (cached)
- **Total: ~3.5-7 seconds TTFB** (acceptable for streaming)

### Optimization Strategies

1. **Parallel RAG queries:**
   - Run RAG retrieval while loading analyses
   - Use asyncio for concurrent operations

2. **Caching:**
   - Cache parsed documents from `_canvas/docs/`
   - Cache RAG results for identical queries (TTL: 5 min)

3. **Lazy loading:**
   - Only load raw transcripts if canvas agent requests them
   - Only read documents if Settings indicate they should be used

4. **Rate limiting:**
   - Limit document upload size (max 10MB per file)
   - Limit total docs per agent (max 20 files)

---

## Risk Mitigation

### Risks & Mitigations

1. **Breaking existing analyses:**
   - **Risk:** Old S3 paths break after restructure
   - **Mitigation:** Migration script + backward compatibility code
   - **Rollback:** Keep old paths for 30 days, redirect to new paths

2. **Context window overflow:**
   - **Risk:** Too much content (transcripts + docs + RAG + analyses)
   - **Mitigation:** Truncate raw sources if > 100k chars, prioritize recent
   - **Monitoring:** Log context sizes, alert if > 150k chars

3. **RAG retrieval degrades quality:**
   - **Risk:** Irrelevant context confuses agents
   - **Mitigation:** Lower tier caps, increase score threshold
   - **Testing:** A/B test with/without RAG, compare quality

4. **Upload abuse:**
   - **Risk:** Users upload massive/inappropriate files
   - **Mitigation:** File size limits, content type validation, rate limiting
   - **Monitoring:** Track upload patterns, alert on anomalies

5. **Performance degradation:**
   - **Risk:** Additional processing slows down system
   - **Mitigation:** Async operations, caching, parallel processing
   - **Monitoring:** Track p95 latency, alert if > 10s

---

## Future Enhancements (Post-IDG)

1. **Document Intelligence:**
   - Extract structured data from PDFs/docs
   - Table parsing and chart extraction
   - Multi-modal analysis (images, diagrams)

2. **Source Management UI:**
   - Upload interface for `_canvas/docs/`
   - List and preview uploaded documents
   - Delete/replace documents
   - Version control for documents

3. **Analysis Comparison:**
   - UI to view historical analyses side-by-side
   - Diff view showing what changed
   - Trend analysis across time

4. **Advanced Attribution:**
   - Clickable source references in canvas responses
   - Highlight quotes from specific sources
   - Filter canvas by source ("only use Document X")

5. **Multi-Agent Synthesis:**
   - Multiple MLP agents analyze same content
   - Canvas synthesizes different perspectives
   - Consensus/dissent highlighting

---

## Code Diff Summary

### New Files

- `scripts/migrate_canvas_structure.py` - Migration script for folder restructure
- `CANVAS_IDG_SUMMIT_IMPLEMENTATION_PLAN.md` - This document

### Modified Files

**`utils/canvas_analysis_agents.py`:**
- Update S3 key paths (latest/previous/history structure)
- Add `list_canvas_docs()`, `read_canvas_doc()`, `get_all_canvas_source_content()`
- Update `save_analysis_doc_to_s3()` for versioning
- Update source headers to standardized format
- Add RAG retrieval to `run_analysis_agent()`
- Update `get_analysis_agent_prompt()` to inject RAG context
- Add metadata section instructions to MLP prompts
- Enhance source differentiation instructions

**`routes/canvas_routes.py`:**
- Add raw transcript reading with `get_all_canvas_source_content()`
- Add RAG retrieval with `RetrievalHandler`
- Update system prompt assembly (add raw sources + RAG sections)
- Add memory reinforcement after streaming
- Update prompt logging for new components

**`utils/transcript_utils.py`:**
- Update source headers in `get_transcript_content_for_analysis()`
- Standardize header format across functions

### Lines Changed (Estimated)

- `canvas_analysis_agents.py`: ~300 lines added/modified
- `canvas_routes.py`: ~150 lines added/modified
- `transcript_utils.py`: ~50 lines modified
- New migration script: ~200 lines
- Total: **~700 lines of code**

---

## Timeline Estimate

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Folder restructure | 2-3 hours | None |
| Phase 2: Canvas reads raw | 2-3 hours | Phase 1 complete |
| Phase 3: MLP reads docs | 1-2 hours | Phase 1 complete |
| Phase 4: RAG integration | 3-4 hours | Phases 2-3 complete |
| Phase 5: Source attribution | 1-2 hours | All previous phases |
| Testing & validation | 2-3 hours | All phases complete |
| Documentation | 1 hour | All phases complete |
| **Total** | **12-18 hours** | Sequential + parallel work |

**Recommended approach:** Implement phases 1-3 in parallel (different developers), then phase 4-5 sequentially.

---

## Success Metrics

### Functional Metrics

- ✅ Canvas agent responds with raw source awareness
- ✅ MLP analyses include all uploaded documents
- ✅ Clear source attribution in analyses
- ✅ RAG context enhances responses
- ✅ Version history tracked correctly

### Quality Metrics

- MLP analysis quality score ≥ 4/5 (manual evaluation)
- Source attribution present in ≥ 90% of multi-source analyses
- RAG context relevance ≥ 0.7 average score
- Canvas responses cite sources in ≥ 80% of queries

### Performance Metrics

- MLP analysis generation time ≤ 20 seconds
- Canvas TTFB ≤ 7 seconds (p95)
- RAG retrieval time ≤ 2 seconds
- Zero S3 migration errors

### Demo Metrics (IDG Summit)

- Successfully demo multi-source analysis
- Successfully demo raw + analyzed synthesis
- Successfully demo source attribution
- Zero errors during live demo
- Positive audience feedback

---

## Appendix: Example Output

### Example MLP Analysis with Enhanced Attribution

```markdown
# Analysis Metadata

**Sources Analyzed:**
- Transcript: 2025-10-15_14-30.txt (Event: 0000)
- Transcript: breakout_1_session.txt (Event: breakout_1)
- Transcript: breakout_2_session.txt (Event: breakout_2)
- Document: idg_strategy_2025.pdf (Type: PDF, 245KB)

**Settings:**
- Transcript Listen: latest
- Groups Read: breakout (2 events)

**Generated:** 2025-10-15 14:45:23 UTC

---

# Mirror Analysis: Explicit Information

## Surface Level: Center of Distribution

In the main event transcript, participants repeatedly emphasize the importance of psychological safety as a foundation for leadership development. Sarah explicitly states: "We can't have authentic leadership without creating space for vulnerability." This theme surfaces again when Marcus describes his team's experience: "Once we established ground rules for respectful disagreement, the quality of our strategic discussions doubled."

Meanwhile, breakout_1 focuses primarily on practical implementation challenges. The group identifies three concrete obstacles: limited time for reflection, competing organizational priorities, and lack of executive sponsorship. Jennifer notes: "We know what good leadership looks like, but we're constantly firefighting."

In contrast, breakout_2 and the uploaded IDG strategy document both converge on developmental pathways. Breakout_2 explores peer coaching models, with multiple participants sharing positive experiences. The strategy document outlines a five-stage maturity model that aligns closely with what breakout_2 discovered organically through their discussion.

Across all sources, the pattern of "inner development before outer impact" appears consistently, though expressed differently: the main transcript frames it philosophically, breakout_1 struggles with it practically, and breakout_2 plus the strategy document offer structured approaches.

## Deep Level: Edge Cases

One participant in breakout_2 raises a dissenting view that receives little pickup: "What if some people are already internally developed but just need skills training?" This challenges the prevailing assumption but goes unexplored.

The strategy document includes a brief footnote about cultural adaptation that none of the transcripts address: "Western frameworks of individual development may require modification for collectivist cultural contexts."

In the main event, there's a single comment from Thomas about generational differences in leadership expectations that doesn't connect to the broader conversation: "Gen Z leaders want immediate impact, not slow developmental journeys."

## Most Pertinent Observation

The most significant explicit pattern across all four sources is the tension between aspirational leadership models and organizational readiness. The main transcript articulates the vision, the strategy document provides the framework, breakout_2 finds peer support mechanisms, but breakout_1 reveals the implementation gap: even when leaders understand inner development principles and have access to structured models, systemic barriers prevent consistent application. This suggests the missing piece isn't better frameworks or more awareness, but organizational architecture that enables developmental practices within existing workflows.
```

### Example Canvas Agent Response (with RAG + Raw Sources)

**User Query:** "What did breakout 1 struggle with?"

**Canvas Response:**
"Breakout 1 grappled with three core implementation challenges: finding time for reflection amid constant firefighting, navigating competing organizational priorities that deprioritize development, and securing executive sponsorship for leadership programs. Jennifer captured it precisely when she said they know what good leadership looks like but remain stuck in reactive mode."

**Behind the scenes (system prompt):**
```
=== RETRIEVED MEMORY ===
[t0_foundation] idg_framework_overview.md (score: 0.842, age: 15 days)
The IDG Framework emphasizes five key dimensions of inner development...

[t1] previous_chat_about_challenges.txt (score: 0.791, age: 3 hours)
User previously asked about implementation barriers, agent discussed organizational culture...
=== END RETRIEVED MEMORY ===

=== RAW SOURCES ===
=== SOURCE: Transcript - breakout_1_session.txt (Event: breakout_1, 2025-10-15 14:30) ===
Jennifer: We know what good leadership looks like, but we're constantly firefighting...
Marcus: The CEO talks about development but rewards quarterly results only...
=== END SOURCE ===
=== END RAW SOURCES ===

=== CURRENT ANALYSES ===
--- MIRROR Analysis ---
In contrast, breakout_1 focuses primarily on practical implementation challenges...
--- LENS Analysis ---
Breakout_1 reveals a latent need for structural permission to prioritize development...
--- PORTAL Analysis ---
What if breakout_1's firefighting mode is actually a symptom of deeper misalignment...
=== END CURRENT ANALYSES ===
```

---

**End of Implementation Plan**
