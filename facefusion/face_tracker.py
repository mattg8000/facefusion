"""
Face Tracker - Temporal tracking and adaptive threshold management

This module provides temporal tracking of faces across frames to enable:
- Adaptive threshold adjustment based on recent match history
- Spatial continuity tracking using bounding box position
- Multiple reference embeddings from cluster database
"""

from typing import Dict, List, Optional, Tuple
from collections import deque
import numpy

from facefusion import logger
from facefusion.types import Face


class FaceMatchHistory:
	"""Tracks match history for a single face identity"""
	
	def __init__(self, max_history: int = 10):
		self.max_history = max_history
		self.match_history: deque = deque(maxlen=max_history)  # List of (frame_number, matched) tuples
		self.last_position: Optional[Tuple[float, float, float, float]] = None  # (x1, y1, x2, y2)
		self.last_frame: Optional[int] = None
	
	def record_match(self, frame_number: int, bounding_box: Tuple[float, float, float, float]) -> None:
		"""Record a successful match"""
		self.match_history.append((frame_number, True))
		self.last_position = bounding_box
		self.last_frame = frame_number
	
	def record_miss(self, frame_number: int) -> None:
		"""Record a failed match"""
		self.match_history.append((frame_number, False))
	
	def get_recent_match_count(self, window_size: int = 5) -> int:
		"""Get number of matches in recent window"""
		if not self.match_history:
			return 0
		recent = list(self.match_history)[-window_size:]
		return sum(1 for _, matched in recent if matched)
	
	def was_recently_matched(self, window_size: int = 5) -> bool:
		"""Check if face was matched in recent frames"""
		return self.get_recent_match_count(window_size) > 0
	
	def get_adaptive_threshold_multiplier(self, base_threshold: float, window_size: int = 5) -> float:
		"""
		Calculate adaptive threshold multiplier based on recent match history.
		Returns a multiplier (e.g., 1.5 means use 1.5x the base threshold).
		"""
		if not self.was_recently_matched(window_size):
			return 1.0  # Use base threshold
		
		# If recently matched, relax threshold by up to 50%
		recent_matches = self.get_recent_match_count(window_size)
		# More recent matches = more relaxation (up to 1.5x threshold)
		relaxation = 1.0 + (recent_matches / window_size) * 0.5
		return min(relaxation, 1.5)  # Cap at 1.5x
	
	def get_spatial_distance(self, bounding_box: Tuple[float, float, float, float]) -> float:
		"""Calculate spatial distance from last known position"""
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
	"""Tracks multiple face identities across frames"""
	
	def __init__(self):
		# Map cluster_id -> FaceMatchHistory
		self.cluster_histories: Dict[int, FaceMatchHistory] = {}
		# Map (frame_number, face_index) -> cluster_id (for reverse lookup)
		self.frame_face_to_cluster: Dict[Tuple[int, int], int] = {}
	
	def record_match(self, frame_number: int, face_index: int, cluster_id: int, bounding_box: Tuple[float, float, float, float]) -> None:
		"""Record a successful match"""
		if cluster_id not in self.cluster_histories:
			self.cluster_histories[cluster_id] = FaceMatchHistory()
		
		self.cluster_histories[cluster_id].record_match(frame_number, bounding_box)
		self.frame_face_to_cluster[(frame_number, face_index)] = cluster_id
	
	def record_miss(self, frame_number: int, cluster_id: int) -> None:
		"""Record a failed match attempt"""
		if cluster_id in self.cluster_histories:
			self.cluster_histories[cluster_id].record_miss(frame_number)
	
	def get_adaptive_threshold(self, cluster_id: int, base_threshold: float) -> float:
		"""Get adaptive threshold for a cluster based on its match history"""
		if cluster_id not in self.cluster_histories:
			return base_threshold
		
		multiplier = self.cluster_histories[cluster_id].get_adaptive_threshold_multiplier(base_threshold)
		return base_threshold * multiplier
	
	def get_cluster_for_face(self, frame_number: int, face_index: int) -> Optional[int]:
		"""Get cluster ID for a face in a specific frame (from recent history)"""
		return self.frame_face_to_cluster.get((frame_number, face_index))
	
	def clear(self) -> None:
		"""Clear all tracking history"""
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

