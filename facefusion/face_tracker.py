"""
Face Tracker - Temporal tracking and adaptive threshold management

This module provides temporal tracking of faces across frames to enable:
- Adaptive threshold adjustment based on recent match history
- Spatial continuity tracking using bounding box position
- Multiple reference embeddings from cluster database
- Thread-safe operation for parallel frame processing
"""

from typing import Dict, List, Optional, Tuple
from collections import deque
import threading
import numpy

from facefusion import logger
from facefusion.types import Face


class FaceMatchHistory:
	"""Tracks match history for a single face identity (thread-safe)"""
	
	def __init__(self, max_history: int = 20):
		self.max_history = max_history
		# Store (frame_number, matched) tuples - sorted by frame_number for temporal queries
		self.match_history: List[Tuple[int, bool]] = []
		self._lock = threading.Lock()  # Thread safety for parallel frame processing
		self.last_position: Optional[Tuple[float, float, float, float]] = None  # (x1, y1, x2, y2)
		self.last_frame: Optional[int] = None
	
	def record_match(self, frame_number: int, bounding_box: Tuple[float, float, float, float]) -> None:
		"""Record a successful match (thread-safe)"""
		with self._lock:
			# Insert in sorted order by frame_number (handles out-of-order processing)
			self._insert_sorted((frame_number, True))
			self.last_position = bounding_box
			self.last_frame = frame_number
	
	def record_miss(self, frame_number: int) -> None:
		"""Record a failed match (thread-safe)"""
		with self._lock:
			self._insert_sorted((frame_number, False))
	
	def _insert_sorted(self, entry: Tuple[int, bool]) -> None:
		"""Insert entry maintaining sorted order by frame_number"""
		frame_number, _ = entry
		
		# Find insertion point (binary search would be better, but list is small)
		insert_pos = len(self.match_history)
		for i, (fn, _) in enumerate(self.match_history):
			if fn > frame_number:
				insert_pos = i
				break
			elif fn == frame_number:
				# Update existing entry
				self.match_history[i] = entry
				return
		
		self.match_history.insert(insert_pos, entry)
		
		# Keep only recent history
		if len(self.match_history) > self.max_history:
			self.match_history = self.match_history[-self.max_history:]
	
	def get_temporal_match_count(self, current_frame: int, window_size: int = 5) -> int:
		"""
		Get number of matches in temporal window before current_frame.
		This works correctly even with out-of-order frame processing.
		"""
		with self._lock:
			if not self.match_history:
				return 0
			
			# Find frames in the temporal window [current_frame - window_size, current_frame)
			window_start = max(0, current_frame - window_size)
			matches = sum(1 for fn, matched in self.match_history 
			              if window_start <= fn < current_frame and matched)
			return matches
	
	def get_adaptive_threshold_multiplier(self, current_frame: int, base_threshold: float, window_size: int = 5) -> float:
		"""
		Calculate adaptive threshold multiplier based on temporal match history.
		Uses actual frame numbers, not completion order, so it works with parallel processing.
		
		Args:
			current_frame: The frame number being processed
			base_threshold: Base distance threshold
			window_size: Number of previous frames to consider
		
		Returns:
			A multiplier (e.g., 1.5 means use 1.5x the base threshold).
		"""
		with self._lock:
			matches = self.get_temporal_match_count(current_frame, window_size)
			
			if matches == 0:
				return 1.0  # Use base threshold
			
			# If recently matched in temporal window, relax threshold
			# More matches = more relaxation (up to 1.5x threshold)
			relaxation = 1.0 + (matches / window_size) * 0.5
			return min(relaxation, 1.5)  # Cap at 1.5x
	
	def get_spatial_distance(self, bounding_box: Tuple[float, float, float, float]) -> float:
		"""Calculate spatial distance from last known position (thread-safe)"""
		with self._lock:
			if self.last_position is None:
				return float('inf')
			
			# Calculate center distance
			last_center = (
				(self.last_position[0] + self.last_position[2]) / 2,
				(self.last_position[1] + self.last_position[3]) / 2
			)
			current_center = (
				(bounding_box[0] + bounding_box[2]) / 2,
				(bounding_box[1] + bounding_box[3]) / 2
			)
			
			distance = ((last_center[0] - current_center[0])**2 + (last_center[1] - current_center[1])**2)**0.5
			return distance


class FaceTracker:
	"""Tracks multiple face identities across frames (thread-safe)"""
	
	def __init__(self):
		# Map cluster_id -> FaceMatchHistory
		self.cluster_histories: Dict[int, FaceMatchHistory] = {}
		# Map (frame_number, face_index) -> cluster_id (for reverse lookup)
		self.frame_face_to_cluster: Dict[Tuple[int, int], int] = {}
		self._lock = threading.Lock()  # Thread safety for parallel frame processing
	
	def record_match(self, frame_number: int, face_index: int, cluster_id: int, bounding_box: Tuple[float, float, float, float]) -> None:
		"""Record a successful match (thread-safe)"""
		with self._lock:
			if cluster_id not in self.cluster_histories:
				self.cluster_histories[cluster_id] = FaceMatchHistory()
			
			self.cluster_histories[cluster_id].record_match(frame_number, bounding_box)
			self.frame_face_to_cluster[(frame_number, face_index)] = cluster_id
	
	def record_miss(self, frame_number: int, cluster_id: int) -> None:
		"""Record a failed match attempt (thread-safe)"""
		with self._lock:
			if cluster_id in self.cluster_histories:
				self.cluster_histories[cluster_id].record_miss(frame_number)
	
	def get_adaptive_threshold(self, cluster_id: int, current_frame: int, base_threshold: float, window_size: int = 5) -> float:
		"""
		Get adaptive threshold for a cluster based on its temporal match history.
		
		Args:
			cluster_id: The cluster ID to get threshold for
			current_frame: The frame number being processed (for temporal window)
			base_threshold: Base distance threshold
			window_size: Number of previous frames to consider for temporal smoothing
		
		Returns:
			Adaptive threshold (base_threshold * multiplier)
		"""
		with self._lock:
			if cluster_id not in self.cluster_histories:
				return base_threshold
		
		# Get multiplier outside lock (FaceMatchHistory has its own lock)
		multiplier = self.cluster_histories[cluster_id].get_adaptive_threshold_multiplier(
			current_frame, base_threshold, window_size
		)
		return base_threshold * multiplier
	
	def get_cluster_for_face(self, frame_number: int, face_index: int) -> Optional[int]:
		"""Get cluster ID for a face in a specific frame (from recent history, thread-safe)"""
		with self._lock:
			return self.frame_face_to_cluster.get((frame_number, face_index))
	
	def clear(self) -> None:
		"""Clear all tracking history (thread-safe)"""
		with self._lock:
			self.cluster_histories.clear()
			self.frame_face_to_cluster.clear()


# Global face tracker instance
_face_tracker: Optional[FaceTracker] = None


def get_face_tracker() -> FaceTracker:
	"""Get or create the global face tracker"""
	global _face_tracker
	if _face_tracker is None:
		_face_tracker = FaceTracker()
	return _face_tracker


def clear_face_tracker() -> None:
	"""Clear the face tracker (e.g., when starting a new video)"""
	global _face_tracker
	if _face_tracker is not None:
		_face_tracker.clear()
	_face_tracker = None

