# Face Replacement Improvements - Planning Document

**Date:** 2024  
**Status:** Planning Phase

## Overview

This document outlines identified limitations in FaceFusion's face replacement capabilities and proposes technical solutions to address them. The improvements focus on three main areas: multi-face replacement, robust face tracking, and adaptive settings.

---

## Architectural Discovery: Face Detection is On-Demand, Not Preprocessed

### Key Finding
**FaceFusion does NOT perform a preprocessing step that scans all video frames for faces.** Face detection happens on-demand during frame processing.

### Current Architecture

**Location:** `facefusion/workflows/image_to_video.py` and `facefusion/face_analyser.py`

1. **Frame Extraction Phase:**
   - `extract_frames()` extracts all frames to temp directory
   - **No face detection occurs at this stage**

2. **On-Demand Detection:**
   - `process_temp_frame()` processes frames in parallel threads
   - Face detection happens when `get_many_faces()` is called (line 96 in `face_analyser.py`)
   - Detection is triggered per-frame as needed by processors

3. **Caching Mechanism:**
   - `face_store.py` caches detected faces using frame hash as key
   - Prevents re-detection of identical frames
   - **Not a global video analysis** - just per-frame caching

4. **Processing Flow:**
   ```
   extract_frames() → process_video() → process_temp_frame() → 
   get_many_faces() → [check cache] → [if miss: detect on-the-fly] → [cache result]
   ```

### Implications for All Issues

This architectural limitation directly contributes to all three identified issues:

1. **Issue #1 (Single Source Face):**
   - No global view of all faces across video
   - Can't identify unique faces for mapping
   - Each frame processed independently

2. **Issue #2 (Face Recognition Fails):**
   - **No temporal context** - frames processed independently
   - Reference frame is single frame (`reference_frame_number`)
   - No memory of previous successful matches
   - No tracking continuity between frames

3. **Issue #3 (Fixed Settings):**
   - No video-wide analysis to inform adaptive settings
   - Can't optimize settings based on global face characteristics
   - Each frame uses same settings without context

### Potential Optimization: Preprocessing Step

A preprocessing step that scans all frames could:

- **Build global face database:**
  - Extract all faces from all frames upfront
  - Create embeddings for each unique face
  - Identify face identities across entire video

- **Enable better tracking:**
  - Pre-compute face trajectories
  - Identify when faces appear/disappear
  - Map face identities across frames

- **Support adaptive settings:**
  - Analyze video-wide face characteristics
  - Detect scenes with different requirements
  - Optimize settings per scene/face

- **Improve Issue #2 solutions:**
  - Multiple reference embeddings from different frames
  - Temporal tracking with pre-computed trajectories
  - Better matching with global context

### Face Identification and Clustering for Preprocessing

#### How FaceFusion Currently Identifies Faces

**Location:** `facefusion/face_recognizer.py` and `facefusion/face_selector.py`

1. **Face Embeddings:**
   - Uses **ArcFace model** (InsightFace) to generate 512-dimensional embeddings
   - Each face has `embedding` (raw) and `embedding_norm` (normalized L2)
   - Embeddings are computed from face landmarks (5-point) after warping to 112x112

2. **Distance Calculation:**
   ```python
   distance = 1 - numpy.dot(face.embedding_norm, reference_face.embedding_norm)
   ```
   - Range: **0.0** (identical) to **2.0** (completely different)
   - Interpolated to [0, 1] range for threshold comparison
   - Uses **cosine similarity** (dot product of normalized vectors)

3. **Current Threshold:**
   - Default: **0.3** (configurable 0.0-1.0 in steps of 0.05)
   - Stored in `reference_face_distance` state
   - Lower = stricter matching (fewer false positives, more false negatives)
   - Higher = looser matching (more false positives, fewer false negatives)

4. **Multiple Faces in Frame:**
   - `detect_faces()` returns multiple bounding boxes, scores, and landmarks
   - Each detected face gets its own embedding calculated independently
   - No grouping or clustering - just separate detections
   - Faces are identified by position/index, not by identity

#### Face Clustering Strategy for Preprocessing

**Goal:** Build a database of unique face identities across the entire video, allowing same person with different appearances (lighting, makeup, glasses) to be grouped together.

**Approach:**

1. **Collect All Faces:**
   - Scan all frames, extract all faces with embeddings
   - Store: `(frame_number, face_index, face_object, embedding_norm)`
   - Result: List of all face detections across video

2. **Clustering Algorithm Options:**

   **Option A: Hierarchical Clustering (Recommended)**
   - Use embedding distance to build clusters
   - Start with each face as own cluster
   - Merge clusters with average linkage (average distance between clusters)
   - Stop when minimum distance exceeds threshold
   - **Pros:** Handles varying cluster sizes, allows threshold tuning
   - **Cons:** O(n²) complexity for large videos

   **Option B: DBSCAN (Density-Based)**
   - Groups faces that are density-reachable
   - Parameters: `eps` (distance threshold), `min_samples` (min faces per cluster)
   - **Pros:** Handles noise/outliers, automatic cluster count
   - **Cons:** Requires tuning eps parameter, may split same person

   **Option C: K-Means with Elbow Method**
   - Estimate number of unique faces first
   - Use K-means to cluster
   - **Pros:** Fast, well-understood
   - **Cons:** Requires knowing/estimating K, assumes spherical clusters

3. **Threshold Selection for Clustering:**

   **Conservative Approach (Fewer Clusters):**
   - Use **higher threshold** (0.4-0.6) for clustering
   - Same person with different appearances grouped together
   - **Risk:** Different people might be merged if similar-looking
   - **Benefit:** User assigns source face once per person

   **Strict Approach (More Clusters):**
   - Use **lower threshold** (0.2-0.3) for clustering
   - Same person with different appearances creates separate clusters
   - **Risk:** Too many clusters, user must assign source face multiple times
   - **Benefit:** Less risk of merging different people

   **Recommended: Adaptive Threshold**
   - Start with moderate threshold (0.35-0.45)
   - Allow user to **merge clusters** if same person
   - Allow user to **split clusters** if different people
   - UI shows cluster representatives (best quality face per cluster)

4. **Cluster Representation:**
   - Each cluster needs a **representative face** (centroid or best quality)
   - Store: `cluster_id, representative_face, all_face_instances[]`
   - Face instances: `(frame_number, face_index, embedding_distance_to_centroid)`

5. **Handling Edge Cases:**
   - **Same person, different appearances:** Multiple clusters, user can merge
   - **Different people, similar appearance:** Single cluster, user can split
   - **Partial faces/occlusions:** May create separate clusters, can merge
   - **Face entering/leaving:** Tracked as separate instances of same cluster

#### Implementation Details

**Data Structure:**
```python
FaceCluster = TypedDict('FaceCluster', {
    'cluster_id': int,
    'representative_face': Face,  # Best quality or centroid
    'all_instances': List[Tuple[int, int, float]],  # (frame, index, distance)
    'average_embedding': Embedding,  # Cluster centroid
    'frame_range': Tuple[int, int]  # First and last appearance
})

VideoFaceDatabase = TypedDict('VideoFaceDatabase', {
    'clusters': List[FaceCluster],
    'face_to_cluster': Dict[Tuple[int, int], int],  # (frame, index) -> cluster_id
    'clustering_threshold': float
})
```

**Clustering Function:**
```python
def cluster_video_faces(all_faces: List[Tuple[int, int, Face]]) -> VideoFaceDatabase:
    # 1. Extract embeddings
    # 2. Compute distance matrix (or use approximate for large sets)
    # 3. Apply clustering algorithm
    # 4. Create cluster representatives
    # 5. Build mapping: face -> cluster
    # 6. Return database
```

**User Interface:**
- Show cluster gallery with representative faces
- Allow merging clusters (drag-drop or checkbox)
- Allow splitting clusters (select instances to remove)
- Preview which frames/instances belong to each cluster
- Adjust clustering threshold and re-cluster

#### Threshold Balancing Strategy

**The Challenge:**
- Too strict: Same person with glasses/makeup = separate clusters (user assigns source multiple times)
- Too loose: Different people = same cluster (wrong face replacement)

**Solution: Multi-Level Approach:**

1. **Initial Clustering:** Use moderate threshold (0.35-0.4)
   - Groups obvious same-person cases
   - May create separate clusters for appearance variations

2. **User Review & Merge:**
   - UI shows clusters with representative faces
   - User can visually identify "these are the same person"
   - Merge clusters with one click
   - System learns: merged clusters = same person despite distance

3. **Smart Suggestions:**
   - If user merges clusters A and B, suggest merging similar clusters
   - Track user corrections to improve future suggestions
   - Show confidence scores: "These might be the same person (85% confidence)"

4. **Per-Cluster Source Assignment:**
   - User assigns source face to each cluster
   - If same person has multiple clusters (merged or not), can assign same source
   - System applies source to all instances in cluster(s)

#### Design Decision: Same Person, Multiple Clusters - Settings Strategy

**Scenario:** Face detection has identified two clusters (face1, face2) from the same actual person because embedding distance exceeded threshold (e.g., person with/without glasses, different lighting, makeup changes).

**Question:** Should we use different replacement settings per cluster, or rely on temporal tracking?

**Analysis:**

**Option A: Per-Cluster Settings (Different Settings for Each Appearance)**
- **Rationale:** Different appearances may require different processing settings
  - Face1 (no glasses, good lighting): Use model A, bounding box detection
  - Face2 (glasses, low light): Use model B, region-based detection, higher detector score
- **Pros:**
  - Optimizes settings for each appearance variant
  - Handles cases where one appearance is harder to process
  - Can use different models optimized for different conditions
- **Cons:**
  - More complex configuration
  - User must understand why settings differ
  - May create inconsistent results if settings vary too much

**Option B: Temporal Tracking (Maintain Identity Across Appearance Changes)**
- **Rationale:** Use tracking to recognize same person despite appearance changes
- **Pros:**
  - Maintains identity continuity
  - Simpler user experience (one source face assignment)
  - Handles gradual appearance changes smoothly
- **Cons:**
  - Doesn't address that different appearances might need different settings
  - Still need to handle the settings question separately

**Option C: Hybrid Approach ⭐ RECOMMENDED**
- **Combine both strategies:**
  1. **Temporal tracking** maintains identity across appearance changes
  2. **Per-cluster/per-appearance settings** optimize processing for each variant
  3. **Same source face** applied to all clusters of same person
  4. **Different settings** can be applied per cluster if needed

**Implementation:**
```python
# Same person, multiple clusters (merged or linked via tracking)
person_identity = {
    'clusters': [cluster_1, cluster_2],  # Same person, different appearances
    'source_face': source_face_A,  # Same source for all
    'settings': {
        'cluster_1': {  # Appearance 1: no glasses, good lighting
            'face_detector_model': 'retinaface',
            'face_detector_score': 0.5,
            'face_swapper_model': 'model_a'
        },
        'cluster_2': {  # Appearance 2: glasses, low light
            'face_detector_model': 'scrfd',
            'face_detector_score': 0.7,  # Higher threshold for harder detection
            'face_swapper_model': 'model_b'  # Different model for glasses
        }
    }
}
```

**Why This Makes Sense:**
- **Same source face:** User wants same person replaced, regardless of appearance
- **Different settings:** Different appearances may need different processing
  - Glasses: Higher detector score, different model
  - Low light: Different detector, enhancement settings
  - Side profile: Different angle detection, landmark settings
- **Temporal tracking:** Maintains continuity when appearance changes gradually
- **User control:** User can choose to use same settings or optimize per-appearance

**UI Design:**
- Show clusters grouped by person identity (if merged or tracked)
- Allow per-cluster settings override
- Default: Same settings for all clusters of same person
- Advanced: "Optimize settings per appearance" checkbox
  - If checked: System suggests different settings per cluster
  - User can review and adjust

**Example Use Cases:**

1. **Person with/without glasses:**
   - Same source face for both
   - Cluster 1 (no glasses): Standard settings
   - Cluster 2 (glasses): Higher detector score, different model

2. **Person in different lighting:**
   - Same source face for both
   - Cluster 1 (bright): Standard settings
   - Cluster 2 (dark): Lower detector score, enhancement enabled

3. **Person at different angles:**
   - Same source face for all
   - Cluster 1 (front): Standard detection
   - Cluster 2 (profile): Different angle detection, adjusted landmarks

**Recommendation:**
- **Use temporal tracking** to maintain identity and enable same source assignment
- **Allow per-cluster settings** as an optional optimization
- **Default to same settings** for simplicity, but enable per-cluster optimization when needed
- **UI should make it clear:** "Same person, different appearances - optimize settings?"

### Implementation Considerations

**Option A: Add Preprocessing Step (Recommended)**
- New function: `preprocess_video_faces()` called before `process_video()`
- Scans all frames, extracts all faces, builds face database
- Clusters faces using embedding distance
- Stores results in extended `face_store` or new `video_face_database`
- Existing code can query database instead of detecting on-demand
- **UI:** Cluster gallery for review and source face assignment

**Option B: Maintain On-Demand with Stateful Tracking**
- Keep current architecture but add stateful tracking
- Maintain face tracker state across frames
- More complex but less disruptive
- **Limitation:** No global view, harder to cluster

**Option C: Hybrid Approach**
- Preprocess key frames (every Nth frame) for global context
- Use on-demand detection for intermediate frames
- Balance between performance and functionality
- **Limitation:** May miss faces that only appear in skipped frames

### Impact on Solution Design

This discovery means:
- **Issue #2 solutions** (temporal tracking) become more critical - they're not just nice-to-have, they're necessary to overcome the lack of preprocessing
- **Issue #3 solutions** (adaptive settings) could benefit from preprocessing to analyze video characteristics
- **Issue #1 solutions** (multi-face) could use preprocessing to identify all unique faces upfront

---

## Issue 1: Single Source Face Limitation

### Problem Statement
Users can only replace one face in a video at a time, despite the system detecting multiple faces. The UI only provides space for one source replacement face.

### Technical Root Cause

**Location:** `facefusion/processors/modules/face_swapper/core.py`

1. **Single Source Face Extraction:**
   - `extract_source_face()` (line 746) processes all source images but averages them into a single `source_face`
   - Function extracts one face per source image, then calls `get_average_face()` to combine them
   - Result: Only one source face object exists regardless of input

2. **Processing Architecture:**
   - `process_frame()` (line 760) receives one `source_face` and applies it to all `target_faces`
   - Loop structure: `for target_face in target_faces: swap_face(source_face, target_face, ...)`
   - All target faces get the same source face swapped onto them

3. **UI Limitation:**
   - `facefusion/uis/components/source.py` accepts multiple files but only displays one source image
   - No mechanism to map multiple source faces to multiple target faces

### Solution Options

#### Option A: Multiple Source Faces with Explicit Mapping ⭐ **RECOMMENDED**
**Complexity:** Medium | **Impact:** High

**Approach:**
- Allow multiple source images, each providing a distinct source face
- Create a mapping system: `source_face_map: Dict[target_face_id, source_face]`
- Modify `process_frame()` to select appropriate source face per target

**Implementation:**
1. Modify `extract_source_face()` to return `List[Face]` instead of single averaged face
2. Add UI component for source→target face mapping (gallery with drag-drop or selection)
3. Update `process_frame()` to accept mapping and apply correct source per target
4. Add face identification system (position-based, embedding-based, or manual assignment)

**Pros:**
- Maximum flexibility
- Supports complex scenarios (different faces, different sources)
- Clear user control

**Cons:**
- Requires UI redesign
- More complex state management
- User must understand mapping concept

---

#### Option B: Position-Based Automatic Matching
**Complexity:** Low | **Impact:** Medium

**Approach:**
- Match source faces to target faces by position (left-to-right, largest-to-smallest)
- Automatic assignment based on ordering rules

**Implementation:**
1. Sort source faces and target faces by same criteria
2. Map by index: `source_faces[i] → target_faces[i]`

**Pros:**
- Simple to implement
- No UI changes needed
- Works well for consistent positioning

**Cons:**
- Breaks if face order changes
- No user control
- Assumes same number of faces

---

#### Option C: Embedding-Based Automatic Matching
**Complexity:** Medium | **Impact:** Medium

**Approach:**
- Match each target face to closest source face by embedding distance
- Automatic assignment based on similarity

**Implementation:**
1. Compute embedding distances between all source-target pairs
2. Use Hungarian algorithm or greedy matching for optimal assignment
3. Apply matched pairs

**Pros:**
- More robust than position-based
- Handles varying face counts
- Automatic and intelligent

**Cons:**
- May match incorrectly if faces are similar
- No user override
- Requires distance computation overhead

---

## Issue 2: Face Recognition Fails with Appearance Changes

### Problem Statement
During video processing, a character's face may change enough (makeup, lighting, sunglasses) that they are no longer associated with the "reference" image, causing face replacement to fail.

### Technical Root Cause

**Location:** `facefusion/face_selector.py` and `facefusion/face_recognizer.py`

1. **Fixed Threshold Matching:**
   - `compare_faces()` (line 44) uses cosine similarity on face embeddings
   - Single `reference_face_distance` threshold applied to all frames
   - If embedding distance exceeds threshold, face is not matched

2. **Single Reference Frame:**
   - Reference face extracted from one frame (`reference_frame_number`)
   - No adaptation to appearance changes over time
   - Embeddings sensitive to: lighting, makeup, glasses, angle, expression, age

3. **No Temporal Context:**
   - Each frame processed independently
   - No memory of previous successful matches
   - No tracking continuity between frames

4. **Embedding Sensitivity:**
   - Face recognition models (ArcFace) are sensitive to appearance variations
   - Cosine similarity can vary significantly with same person under different conditions

### Solution Options

#### Option A: Adaptive Threshold with Temporal Smoothing ⭐ **RECOMMENDED**
**Complexity:** Medium | **Impact:** High

**Approach:**
- Maintain sliding window of recent successful matches
- Temporarily relax threshold if face was matched in recent frames
- Use position/velocity cues to maintain continuity

**Implementation:**
1. Add `FaceTracker` class to maintain per-face tracking state
2. Store recent match history (last N frames)
3. If face matched in recent frames, use relaxed threshold
4. Use bounding box position/size to predict next location
5. Combine embedding distance with spatial proximity

**Pros:**
- Maintains continuity during temporary appearance changes
- Handles gradual changes well
- Relatively simple to implement

**Cons:**
- May continue tracking wrong face if appearance changes too much
- Requires state management across frames

---

#### Option B: Multiple Reference Embeddings
**Complexity:** Medium | **Impact:** High

**Approach:**
- Collect multiple reference embeddings from different frames/conditions
- Match if target face is close to ANY reference embedding
- Build reference set from various appearances

**Implementation:**
1. Allow user to select multiple reference frames
2. Extract embeddings from each reference frame
3. Store as `List[Face]` instead of single `Face`
4. Match if `min(distance(target, ref)) < threshold` for any reference

**Pros:**
- Handles multiple appearance conditions
- More robust matching
- User can provide diverse reference images

**Cons:**
- Requires UI changes for multiple reference selection
- More embeddings to compare (slight performance cost)
- User must identify good reference frames

---

#### Option C: Temporal Tracking with Kalman Filtering
**Complexity:** High | **Impact:** Very High

**Approach:**
- Track faces across frames using position, size, and motion
- Combine tracking with embedding matching
- Use Kalman filter or optical flow for position prediction

**Implementation:**
1. Implement face tracker with position/size/velocity state
2. Predict next frame position using motion model
3. Match faces by combining embedding distance with spatial proximity
4. Maintain identity across frames even when embedding temporarily fails

**Pros:**
- Very robust to appearance changes
- Handles occlusions and temporary failures
- Professional-grade tracking

**Cons:**
- Complex implementation
- Requires significant architecture changes
- May need tuning for different video types

---

#### Option D: Per-Frame Reference Updates
**Complexity:** Low | **Impact:** Medium

**Approach:**
- Periodically update reference embedding from recent successful matches
- Adapts to gradual appearance changes

**Implementation:**
1. Track confidence of matches
2. If match confidence high for N consecutive frames, update reference embedding
3. Use exponential moving average: `ref_embedding = 0.9 * ref_embedding + 0.1 * matched_embedding`

**Pros:**
- Simple to implement
- Adapts automatically
- No UI changes needed

**Cons:**
- May drift if wrong face matched
- Requires confidence scoring
- May adapt too slowly for rapid changes

---

## Issue 3: Fixed Settings Throughout Video

### Problem Statement
Face replacement options (detector, enhancement, model, etc.) are fixed for the duration of the video. Settings that work well at some points may not work at others, but there's no way to vary settings.

### Technical Root Cause

**Location:** `facefusion/workflows/image_to_video.py` and `facefusion/state_manager.py`

1. **Global State Management:**
   - Settings stored in `state_manager` as global state
   - Read once at video processing start
   - `process_temp_frame()` (line 155) uses same settings for every frame

2. **No Frame-Specific Configuration:**
   - No mechanism to vary settings by frame number
   - No scene detection or adaptive logic
   - Settings applied uniformly: `state_manager.get_item('face_detector_score')` etc.

3. **Stateless Processing:**
   - Each frame processed independently
   - No context about previous frames or scenes
   - No way to detect when different settings would help

### Solution Options

#### Option A: Frame-Range-Based Settings ⭐ **RECOMMENDED**
**Complexity:** Medium | **Impact:** High

**Approach:**
- Allow different parameter values for different frame ranges
- UI: Timeline with keyframes where settings can change
- Store as: `List[Tuple[frame_range, settings_dict]]`

**Implementation:**
1. Add `frame_settings: List[FrameRangeSettings]` to state
2. `FrameRangeSettings = TypedDict('FrameRangeSettings', {'start': int, 'end': int, 'settings': Dict})`
3. Modify `process_temp_frame()` to lookup settings for current frame
4. Add UI timeline component with keyframe editing
5. Interpolate or step-change between ranges

**Pros:**
- Maximum user control
- Handles known problem areas
- Familiar interface (like video editors)

**Cons:**
- Requires UI timeline component
- User must identify frame ranges
- More complex state management

---

#### Option B: Scene-Based Settings
**Complexity:** High | **Impact:** High

**Approach:**
- Auto-detect scene changes (histogram differences, optical flow)
- Apply different settings per scene
- UI: Show detected scenes, allow per-scene overrides

**Implementation:**
1. Implement scene detection algorithm
2. Segment video into scenes
3. Store settings per scene
4. UI shows scene boundaries and allows editing

**Pros:**
- Automatic detection
- Intuitive (scenes are natural boundaries)
- Can work well for many videos

**Cons:**
- Scene detection may be inaccurate
- Requires scene detection algorithm
- More complex than frame ranges

---

#### Option C: Adaptive Settings Based on Face Characteristics
**Complexity:** High | **Impact:** Medium

**Approach:**
- Automatically adjust settings based on detected face properties
- Example: Higher detector score for side profiles, different model for low-light frames

**Implementation:**
1. Analyze face properties per frame (angle, size, lighting estimate)
2. Define rules: `if face_angle > 45: use detector_score = 0.7 else 0.5`
3. Apply rules automatically

**Pros:**
- Fully automatic
- Adapts to content
- No user intervention needed

**Cons:**
- Rules may not work for all cases
- Requires property detection
- May need tuning per video type

---

#### Option D: Manual Keyframe System
**Complexity:** Medium | **Impact:** High

**Approach:**
- UI: Timeline with keyframes where settings can change
- Interpolate between keyframes or use step changes
- Similar to video editing software

**Implementation:**
1. Add keyframe data structure
2. Timeline UI component
3. Interpolation logic between keyframes
4. Settings lookup by frame number

**Pros:**
- Precise control
- Familiar interface
- Flexible interpolation options

**Cons:**
- Requires timeline UI
- User must set keyframes
- More complex than frame ranges

---

#### Option E: Per-Face Settings
**Complexity:** Medium | **Impact:** Medium

**Approach:**
- Different settings per face identity
- Useful when multiple faces need different treatment

**Implementation:**
1. Identify faces by embedding or tracking
2. Store settings per face ID
3. Apply appropriate settings when processing each face

**Pros:**
- Handles multi-face scenarios well
- Can optimize per person
- Useful for Issue #1 solutions

**Cons:**
- Requires face identity tracking
- More complex state
- May not address single-face variation issues

---

## Recommended Implementation Priority

### Phase 1: High Impact, Moderate Complexity
1. **Issue #1 - Option A (Multiple Source Faces with Mapping)**
   - Enables multi-face replacement
   - Foundation for other improvements
   - Clear user value

2. **Issue #2 - Option A (Adaptive Threshold with Temporal Smoothing)**
   - Significantly improves reliability
   - Handles common failure cases
   - Moderate implementation effort

### Phase 2: High Impact, Higher Complexity
3. **Issue #3 - Option A (Frame-Range-Based Settings)**
   - Provides needed flexibility
   - Requires UI work but high value
   - Can be built incrementally

### Phase 3: Advanced Features
4. **Issue #2 - Option C (Temporal Tracking with Kalman Filtering)**
   - Professional-grade tracking
   - Handles edge cases
   - Significant architecture changes

5. **Issue #3 - Option B (Scene-Based Settings)**
   - Automatic detection
   - Complements frame-range approach

---

## Architecture Considerations

### Current Design
- **Stateless per-frame processing** with global state
- Each frame processed independently
- Settings read from `state_manager` once
- **No preprocessing step** - face detection is on-demand per frame (see Architectural Discovery section above)
- Face detection results cached per-frame by hash, but no global video analysis

### Required Changes

1. **Optional Preprocessing Step (Recommended):**
   - Add `preprocess_video_faces()` function to scan all frames upfront
   - Build global face database with embeddings and identities
   - Store in extended `face_store` or new `video_face_database`
   - Enables better tracking, multi-face mapping, and adaptive settings

2. **Stateful Tracking:**
   - Add `FaceTracker` class to maintain per-face state across frames
   - Store in frame processing context or separate tracking module
   - **Critical** if preprocessing is not implemented (see Architectural Discovery)

3. **Per-Face/Per-Range Configuration:**
   - Extend state management to support frame-specific settings
   - Add lookup functions: `get_settings_for_frame(frame_number)`

4. **Multi-Source Architecture:**
   - Change `source_face` from single `Face` to `Dict[face_id, Face]` or `List[Face]`
   - Add mapping logic in `process_frame()`
   - Can leverage preprocessing results for face identification

5. **UI Enhancements:**
   - Timeline component for frame-range settings
   - Source→target face mapping interface
   - Multiple reference frame selection

### Files Requiring Modification

**Core Processing:**
- `facefusion/processors/modules/face_swapper/core.py` - Multi-source, per-face settings
- `facefusion/workflows/image_to_video.py` - Frame-specific settings lookup
- `facefusion/face_selector.py` - Temporal tracking, adaptive thresholds

**State Management:**
- `facefusion/state_manager.py` - Extended state types
- `facefusion/types.py` - New type definitions

**UI Components:**
- `facefusion/uis/components/source.py` - Multi-source interface
- New: `facefusion/uis/components/timeline.py` - Settings timeline
- New: `facefusion/uis/components/face_mapping.py` - Source→target mapping

**New Modules:**
- `facefusion/face_tracker.py` - Temporal face tracking
- `facefusion/settings_manager.py` - Frame-range settings management
- `facefusion/video_face_database.py` - (Optional) Global face database from preprocessing

**Workflow Changes:**
- `facefusion/workflows/image_to_video.py` - Add optional `preprocess_video_faces()` call before `process_video()`

---

## Testing Considerations

### Test Cases Needed

1. **Multi-Face Replacement:**
   - Video with 2+ faces, 2+ source images
   - Verify correct mapping
   - Test position-based and embedding-based matching

2. **Appearance Changes:**
   - Video with lighting changes
   - Video with makeup/glasses changes
   - Verify tracking continuity

3. **Frame-Range Settings:**
   - Video with different settings for different ranges
   - Verify correct settings applied per frame
   - Test interpolation between ranges

4. **Edge Cases:**
   - Faces entering/leaving frame
   - Occlusions
   - Rapid appearance changes
   - Scene transitions

---

## Notes

- All solutions maintain backward compatibility where possible
- Consider migration path for existing workflows
- UI changes should be optional/gradual to not overwhelm users
- Performance impact should be minimal (tracking overhead is acceptable)

---

## Future Considerations

- Machine learning-based scene detection
- Automatic optimal settings detection
- Face re-identification after long occlusions
- Real-time processing optimizations

---

## Manual Face Replacement Override

**Status:** Planning Phase  
**See:** `PlanningDocs/manual-face-override.md` for detailed planning

### Overview
Feature to manually force face detection and replacement for specific frames where automatic detection fails. Users can highlight an area, select a source face, and force replacement that persists across processing runs.

### Key Features
- Frame navigation and selection
- Area highlighting (bounding box selection)
- Forced detection with relaxed settings (score ~0.1)
- Source face selection
- Persistence across re-runs

### Implementation Status
- Planning complete
- Ready for implementation

