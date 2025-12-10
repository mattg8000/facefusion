# Manual Face Replacement Override - Planning Document

**Date:** 2024  
**Status:** Planning Phase  
**Priority:** High (Addresses edge cases where automatic detection fails)

## Overview

This feature allows users to manually force face detection and replacement for specific frames where automatic detection fails. Users can highlight an area of a frame, select a source face, and force replacement that persists across processing runs.

---

## Problem Statement

### Current Limitation
- Automatic face detection may fail in edge cases (poor lighting, occlusions, unusual angles)
- When detection fails, no replacement occurs for that frame
- No way to manually override detection for specific problematic frames
- Manual fixes are lost when re-running processing

### Use Cases
1. **Partial Occlusion:** Face partially hidden but still visible
2. **Poor Lighting:** Face visible but detector score too low
3. **Unusual Angles:** Face at extreme angle not detected
4. **Edge Cases:** Faces that don't meet automatic detection criteria

---

## Solution Design

### Core Concept
Allow users to manually specify:
- **Frame number(s)** where forced replacement should occur
- **Bounding box area** to search for face (user highlights region)
- **Source face** to use for replacement
- **Relaxed detector settings** to find face in highlighted area

### Key Features
1. **Frame Selection:** Navigate to specific frame in video
2. **Area Highlighting:** Draw/select bounding box on frame
3. **Forced Detection:** Use very relaxed detector (score ~0.1) in highlighted area
4. **Source Selection:** Choose which source face to use
5. **Persistence:** Save forced replacements to be applied on every "Continue" run

---

## Data Structures

### Forced Face Replacement
```python
ForcedFaceReplacement = TypedDict('ForcedFaceReplacement', {
    'frame_number': int,  # Frame where replacement should occur
    'bounding_box': Tuple[float, float, float, float],  # (x1, y1, x2, y2) - user-selected area
    'source_face_index': int,  # Which source face to use (0, 1, 2, ...)
    'detector_score': float,  # Very relaxed score (e.g., 0.1) for forced detection
    'detector_model': str,  # Optional: specific detector to use
    'search_region': Optional[Tuple[float, float, float, float]],  # Optional: expanded search area
})

# Store as: frame_number -> List[ForcedFaceReplacement]
# Multiple forced replacements per frame allowed (multiple faces)
ForcedReplacements = Dict[int, List[ForcedFaceReplacement]]
```

### Storage
- Store in `state_manager` as `forced_face_replacements`
- Persist to file (JSON) alongside video database
- Load on video load/preprocess

---

## Implementation Plan

### Phase 1: Data Structure & Storage

#### 1.1 Create Data Types
**File:** `facefusion/types.py`
- Add `ForcedFaceReplacement` TypedDict
- Add `forced_face_replacements` to `State` TypedDict

#### 1.2 Storage Module
**File:** `facefusion/forced_replacements.py` (NEW)
```python
def save_forced_replacements(video_path: str, replacements: ForcedReplacements) -> bool:
    """Save forced replacements to JSON file"""
    
def load_forced_replacements(video_path: str) -> ForcedReplacements:
    """Load forced replacements from JSON file"""
    
def get_forced_replacements_for_frame(frame_number: int) -> List[ForcedFaceReplacement]:
    """Get forced replacements for a specific frame"""
```

**Storage Location:**
- Same directory as video database
- File: `{video_name}_forced_replacements.json`
- Format: JSON with frame_number as keys

---

### Phase 2: Forced Detection Logic

#### 2.1 Enhanced Face Detection
**File:** `facefusion/face_analyser.py`

**New Function:**
```python
def detect_face_in_region(
    vision_frame: VisionFrame,
    search_region: Tuple[float, float, float, float],  # (x1, y1, x2, y2)
    detector_score: float = 0.1,  # Very relaxed
    detector_model: Optional[str] = None
) -> Optional[Face]:
    """
    Force face detection in a specific region with relaxed settings.
    
    Args:
        vision_frame: Frame to search
        search_region: Bounding box area to search (can be expanded)
        detector_score: Very low score threshold (default 0.1)
        detector_model: Optional specific detector model
    
    Returns:
        Detected face or None
    """
    # 1. Crop frame to search region (with padding)
    # 2. Temporarily override detector_score in state_manager
    # 3. Run detection on cropped region
    # 4. Adjust bounding box coordinates back to full frame
    # 5. Restore original detector_score
    # 6. Return face if found
```

**Modify `get_many_faces()`:**
- Check for forced replacements for current frame
- If found, try forced detection first
- Merge forced faces with automatically detected faces

#### 2.2 Integration Point
**File:** `facefusion/workflows/image_to_video.py`

**In `process_temp_frame()`:**
- Before calling processors, check for forced replacements
- If forced replacement exists for this frame:
  - Use forced detection to find face
  - Ensure forced face is included in target_faces
  - Map to correct source face

---

### Phase 3: UI Components

#### 3.1 Frame Navigation & Selection
**File:** `facefusion/uis/components/forced_replacement.py` (NEW)

**Components:**
1. **Frame Navigator:**
   - Frame slider/number input
   - Previous/Next buttons
   - Frame preview image

2. **Area Selection:**
   - Interactive image component with drawing/selection
   - Allow user to draw bounding box
   - Show selected area visually

3. **Source Face Selector:**
   - Gallery of available source faces
   - Select which source to use

4. **Forced Replacement List:**
   - Table/list of all forced replacements
   - Show: frame number, bounding box, source face
   - Edit/Delete buttons

5. **Save/Load Controls:**
   - Save forced replacements
   - Load from file
   - Clear all

#### 3.2 UI Layout Integration
**File:** `facefusion/uis/layouts/default.py`
- Add `forced_replacement` component to right column
- Position after `face_mapping` component

---

### Phase 4: Processing Integration

#### 4.1 Face Swapper Integration
**File:** `facefusion/processors/modules/face_swapper/core.py`

**Modify `process_frame()`:**
```python
def process_frame(inputs: FaceSwapperInputs) -> ProcessorOutputs:
    frame_number = inputs.get('frame_number', 0)
    
    # Check for forced replacements for this frame
    forced_replacements = get_forced_replacements_for_frame(frame_number)
    
    if forced_replacements:
        # Force detection for each forced replacement
        for forced_repl in forced_replacements:
            forced_face = detect_face_in_region(
                target_vision_frame,
                forced_repl['bounding_box'],
                forced_repl['detector_score'],
                forced_repl.get('detector_model')
            )
            
            if forced_face:
                # Add to target_faces with special marker
                # Map to correct source face
                # Process replacement
```

#### 4.2 Face Selector Integration
**File:** `facefusion/face_selector.py`

**Modify `select_faces()`:**
- Check for forced faces first
- Include forced faces in result
- Ensure forced faces are not filtered out

---

## User Workflow

### Step 1: Navigate to Problem Frame
1. User navigates to frame where detection failed
2. Frame preview shows current frame
3. User can see where face should be

### Step 2: Highlight Area
1. User draws/selects bounding box around face area
2. Visual feedback shows selected region
3. User can adjust bounding box coordinates

### Step 3: Force Detection
1. User clicks "Detect Face in Area"
2. System uses relaxed detector (score 0.1) in highlighted area
3. If face found, shows preview
4. If not found, allows manual adjustment

### Step 4: Select Source Face
1. User selects which source face to use
2. Preview shows replacement result
3. User confirms

### Step 5: Save
1. User clicks "Save Forced Replacement"
2. Replacement saved to state and file
3. Appears in forced replacements list

### Step 6: Process
1. When user clicks "Continue", forced replacements are applied
2. Forced faces detected and replaced automatically
3. Persists across re-runs

---

## Technical Details

### Forced Detection Algorithm

```python
def detect_face_in_region(vision_frame, search_region, detector_score=0.1):
    # 1. Expand search region by 20% (padding)
    x1, y1, x2, y2 = search_region
    width = x2 - x1
    height = y2 - y1
    padding_x = width * 0.2
    padding_y = height * 0.2
    
    expanded_region = (
        max(0, x1 - padding_x),
        max(0, y1 - padding_y),
        min(vision_frame.shape[1], x2 + padding_x),
        min(vision_frame.shape[0], y2 + padding_y)
    )
    
    # 2. Crop frame to expanded region
    cropped_frame = vision_frame[
        int(expanded_region[1]):int(expanded_region[3]),
        int(expanded_region[0]):int(expanded_region[2])
    ]
    
    # 3. Save original detector score
    original_score = state_manager.get_item('face_detector_score')
    
    # 4. Temporarily set very relaxed score
    state_manager.set_item('face_detector_score', detector_score)
    
    # 5. Detect faces in cropped region
    faces = get_many_faces([cropped_frame])
    
    # 6. Find face closest to center of original search region
    center_x = (x1 + x2) / 2 - expanded_region[0]
    center_y = (y1 + y2) / 2 - expanded_region[1]
    
    best_face = None
    best_distance = float('inf')
    
    for face in faces:
        bbox = face.bounding_box
        face_center_x = (bbox[0] + bbox[2]) / 2
        face_center_y = (bbox[1] + bbox[3]) / 2
        distance = ((face_center_x - center_x)**2 + (face_center_y - center_y)**2)**0.5
        
        if distance < best_distance:
            best_distance = distance
            best_face = face
    
    # 7. Adjust bounding box back to full frame coordinates
    if best_face:
        adjusted_bbox = numpy.array([
            best_face.bounding_box[0] + expanded_region[0],
            best_face.bounding_box[1] + expanded_region[1],
            best_face.bounding_box[2] + expanded_region[0],
            best_face.bounding_box[3] + expanded_region[1]
        ])
        # Create new Face with adjusted bbox
        best_face = Face(
            bounding_box=adjusted_bbox,
            score_set=best_face.score_set,
            landmark_set=best_face.landmark_set,
            # ... other fields
        )
    
    # 8. Restore original detector score
    state_manager.set_item('face_detector_score', original_score)
    
    return best_face
```

### Persistence Format

**JSON Structure:**
```json
{
  "video_path": "/path/to/video.mp4",
  "forced_replacements": {
    "42": [
      {
        "frame_number": 42,
        "bounding_box": [100, 150, 250, 300],
        "source_face_index": 0,
        "detector_score": 0.1,
        "detector_model": "retinaface"
      }
    ],
    "105": [
      {
        "frame_number": 105,
        "bounding_box": [200, 100, 350, 250],
        "source_face_index": 1,
        "detector_score": 0.1
      }
    ]
  }
}
```

---

## Files to Create/Modify

### New Files
1. `facefusion/forced_replacements.py` - Storage and management
2. `facefusion/uis/components/forced_replacement.py` - UI component

### Modified Files
1. `facefusion/types.py` - Add data types
2. `facefusion/face_analyser.py` - Add forced detection function
3. `facefusion/face_selector.py` - Include forced faces
4. `facefusion/processors/modules/face_swapper/core.py` - Apply forced replacements
5. `facefusion/workflows/image_to_video.py` - Check for forced replacements
6. `facefusion/uis/layouts/default.py` - Add UI component
7. `facefusion/state_manager.py` - Add state item

---

## Edge Cases & Considerations

### Multiple Forced Replacements Per Frame
- Allow multiple forced faces in same frame
- Each can have different source face
- Process all forced faces

### Forced Face Conflicts with Automatic Detection
- **Priority:** Forced faces take precedence
- If automatic detection finds face in same area, use forced face
- Merge forced and automatic faces (forced first)

### Frame Range Support
- Future: Allow frame ranges (e.g., frames 42-50)
- For now: Single frame only

### Bounding Box Validation
- Ensure bounding box is within frame bounds
- Validate coordinates
- Handle edge cases (negative, out of bounds)

### Detector Model Selection
- Allow user to choose detector model for forced detection
- Some models may work better for specific cases
- Default: Use current detector model with relaxed score

---

## Testing Considerations

### Test Cases
1. **Single Frame Forced Replacement:**
   - Add forced replacement for one frame
   - Verify it's applied during processing
   - Verify persistence across re-runs

2. **Multiple Frames:**
   - Add forced replacements for multiple frames
   - Verify all are applied correctly

3. **Multiple Faces Per Frame:**
   - Add multiple forced replacements for same frame
   - Verify all are processed

4. **Edge Cases:**
   - Bounding box at frame edges
   - Very small bounding box
   - Very large bounding box
   - Overlapping forced and automatic detections

5. **Persistence:**
   - Save forced replacements
   - Close and reopen video
   - Verify forced replacements are loaded
   - Re-run processing, verify they're applied

---

## Future Enhancements

1. **Frame Range Support:** Apply forced replacement to frame ranges
2. **Visual Preview:** Show preview of forced replacement before saving
3. **Copy/Paste:** Copy forced replacement to nearby frames
4. **Auto-Suggest:** Suggest forced replacements based on detection failures
5. **Batch Operations:** Apply same forced replacement to multiple frames
6. **Temporal Interpolation:** Interpolate forced replacements between frames

---

## Implementation Priority

### Phase 1 (Core Functionality) - HIGH PRIORITY
1. Data structures and storage
2. Forced detection function
3. Basic UI for frame selection and area highlighting
4. Integration into processing pipeline

### Phase 2 (Enhanced UI) - MEDIUM PRIORITY
1. Improved area selection (drawing tool)
2. Preview of forced replacement
3. Better visualization of forced replacements list

### Phase 3 (Advanced Features) - LOW PRIORITY
1. Frame range support
2. Copy/paste functionality
3. Auto-suggestions

---

## Notes

- This feature addresses edge cases, so it should be optional/advanced
- UI should be clear but not overwhelming
- Persistence is critical - users don't want to re-enter forced replacements
- Forced detection should be conservative (very relaxed settings) to maximize success
- Consider performance impact of forced detection (should be minimal for edge cases)

