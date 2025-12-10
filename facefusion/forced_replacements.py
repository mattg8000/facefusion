"""
Forced Face Replacements - Storage and Management

This module handles storage and retrieval of forced face replacements,
which allow users to manually force face detection and replacement for
specific frames where automatic detection fails.
"""

import json
import os
from typing import Dict, List, Optional

from facefusion import logger
from facefusion.filesystem import get_file_name
from facefusion.types import ForcedFaceReplacement, ForcedReplacements


def get_forced_replacements_file_path(video_path: str) -> str:
	"""
	Get the file path for storing forced replacements for a video.
	
	Args:
		video_path: Path to the video file
	
	Returns:
		Path to the forced replacements JSON file
	"""
	video_name = get_file_name(video_path)
	video_dir = os.path.dirname(video_path)
	if not video_dir:
		video_dir = '.'
	return os.path.join(video_dir, f'{video_name}_forced_replacements.json')


def save_forced_replacements(video_path: str, replacements: ForcedReplacements) -> bool:
	"""
	Save forced replacements to JSON file.
	
	Args:
		video_path: Path to the video file
		replacements: Dictionary mapping frame_number to list of forced replacements
	
	Returns:
		True if saved successfully, False otherwise
	"""
	try:
		file_path = get_forced_replacements_file_path(video_path)
		
		# Convert to JSON-serializable format
		json_data = {
			'video_path': video_path,
			'forced_replacements': {
				str(frame_num): [
					{
						'frame_number': repl['frame_number'],
						'bounding_box': list(repl['bounding_box']),
						'source_face_index': repl['source_face_index'],
						'detector_score': repl['detector_score'],
						'detector_model': repl.get('detector_model')
					}
					for repl in frame_replacements
				]
				for frame_num, frame_replacements in replacements.items()
			}
		}
		
		with open(file_path, 'w') as f:
			json.dump(json_data, f, indent=2)
		
		logger.debug(f'Saved {len(replacements)} forced replacement frame(s) to {file_path}', __name__)
		return True
	except Exception as e:
		logger.error(f'Failed to save forced replacements: {e}', __name__)
		return False


def load_forced_replacements(video_path: str) -> ForcedReplacements:
	"""
	Load forced replacements from JSON file.
	
	Args:
		video_path: Path to the video file
	
	Returns:
		Dictionary mapping frame_number to list of forced replacements
	"""
	try:
		file_path = get_forced_replacements_file_path(video_path)
		
		if not os.path.exists(file_path):
			return {}
		
		with open(file_path, 'r') as f:
			json_data = json.load(f)
		
		# Convert from JSON format back to ForcedReplacements
		replacements: ForcedReplacements = {}
		for frame_str, frame_replacements in json_data.get('forced_replacements', {}).items():
			frame_num = int(frame_str)
			replacements[frame_num] = [
				{
					'frame_number': repl['frame_number'],
					'bounding_box': tuple(repl['bounding_box']),
					'source_face_index': repl['source_face_index'],
					'detector_score': repl['detector_score'],
					'detector_model': repl.get('detector_model')
				}
				for repl in frame_replacements
			]
		
		logger.debug(f'Loaded {len(replacements)} forced replacement frame(s) from {file_path}', __name__)
		return replacements
	except Exception as e:
		logger.error(f'Failed to load forced replacements: {e}', __name__)
		return {}


def get_forced_replacements_for_frame(frame_number: int) -> List[ForcedFaceReplacement]:
	"""
	Get forced replacements for a specific frame from state_manager.
	
	Args:
		frame_number: Frame number to get replacements for
	
	Returns:
		List of forced replacements for the frame
	"""
	from facefusion import state_manager
	
	replacements = state_manager.get_item('forced_face_replacements')
	if not replacements:
		return []
	
	return replacements.get(frame_number, [])


def add_forced_replacement(replacement: ForcedFaceReplacement) -> bool:
	"""
	Add a forced replacement to state_manager.
	
	Args:
		replacement: The forced replacement to add
	
	Returns:
		True if added successfully
	"""
	from facefusion import state_manager
	
	replacements = state_manager.get_item('forced_face_replacements') or {}
	frame_num = replacement['frame_number']
	
	if frame_num not in replacements:
		replacements[frame_num] = []
	
	replacements[frame_num].append(replacement)
	state_manager.set_item('forced_face_replacements', replacements)
	
	logger.debug(f'Added forced replacement for frame {frame_num}', __name__)
	return True


def remove_forced_replacement(frame_number: int, replacement_index: int) -> bool:
	"""
	Remove a forced replacement from state_manager.
	
	Args:
		frame_number: Frame number
		replacement_index: Index of replacement to remove
	
	Returns:
		True if removed successfully
	"""
	from facefusion import state_manager
	
	replacements = state_manager.get_item('forced_face_replacements')
	if not replacements or frame_number not in replacements:
		return False
	
	if 0 <= replacement_index < len(replacements[frame_number]):
		replacements[frame_number].pop(replacement_index)
		
		# Remove frame entry if no replacements left
		if not replacements[frame_number]:
			del replacements[frame_number]
		
		state_manager.set_item('forced_face_replacements', replacements)
		logger.debug(f'Removed forced replacement {replacement_index} for frame {frame_number}', __name__)
		return True
	
	return False


def clear_forced_replacements() -> None:
	"""Clear all forced replacements from state_manager."""
	from facefusion import state_manager
	state_manager.set_item('forced_face_replacements', {})
	logger.debug('Cleared all forced replacements', __name__)

