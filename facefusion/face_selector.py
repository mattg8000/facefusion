from typing import List, Optional

import numpy

from facefusion import state_manager
from facefusion.face_analyser import get_many_faces, get_one_face
from facefusion.types import Face, FaceSelectorOrder, Gender, Race, Score, VisionFrame


def select_faces(reference_vision_frame : VisionFrame, target_vision_frame : VisionFrame, frame_number : Optional[int] = None) -> List[Face]:
	# Check for forced replacements first
	forced_faces = []
	if frame_number is not None:
		from facefusion.forced_replacements import get_forced_replacements_for_frame
		from facefusion.face_analyser import detect_face_in_region
		
		forced_replacements = get_forced_replacements_for_frame(frame_number)
		for forced_repl in forced_replacements:
			forced_face = detect_face_in_region(
				target_vision_frame,
				forced_repl['bounding_box'],
				forced_repl.get('detector_score', 0.1),
				forced_repl.get('detector_model')
			)
			if forced_face:
				forced_faces.append(forced_face)
	
	# Get automatically detected faces
	target_faces = get_many_faces([ target_vision_frame ])
	
	# Merge forced faces with automatically detected faces (forced first)
	# Remove duplicates based on bounding box overlap
	if forced_faces:
		# Add forced faces first
		all_faces = forced_faces.copy()
		
		# Add automatic faces that don't overlap significantly with forced faces
		for auto_face in target_faces:
			is_duplicate = False
			auto_bbox = auto_face.bounding_box
			auto_center = ((auto_bbox[0] + auto_bbox[2]) / 2, (auto_bbox[1] + auto_bbox[3]) / 2)
			auto_area = (auto_bbox[2] - auto_bbox[0]) * (auto_bbox[3] - auto_bbox[1])
			
			for forced_face in forced_faces:
				forced_bbox = forced_face.bounding_box
				forced_center = ((forced_bbox[0] + forced_bbox[2]) / 2, (forced_bbox[1] + forced_bbox[3]) / 2)
				distance = ((auto_center[0] - forced_center[0])**2 + (auto_center[1] - forced_center[1])**2)**0.5
				
				# If centers are within 50 pixels, consider it a duplicate
				if distance < 50:
					is_duplicate = True
					break
			
			if not is_duplicate:
				all_faces.append(auto_face)
		
		target_faces = all_faces
	
	face_selector_mode = state_manager.get_item('face_selector_mode')
	
	# If we have cluster mappings, prefer 'many' mode to get all faces
	cluster_source_mapping = state_manager.get_item('cluster_source_mapping')
	if cluster_source_mapping and len(cluster_source_mapping) > 0:
		# Use 'many' mode when cluster mappings exist to ensure all faces are available
		return sort_and_filter_faces(target_faces)

	if face_selector_mode == 'many':
		return sort_and_filter_faces(target_faces)

	if face_selector_mode == 'one':
		target_face = get_one_face(sort_and_filter_faces(target_faces))
		if target_face:
			return [ target_face ]

	if face_selector_mode == 'reference':
		reference_faces = get_many_faces([ reference_vision_frame ])
		reference_faces = sort_and_filter_faces(reference_faces)
		reference_face = get_one_face(reference_faces, state_manager.get_item('reference_face_position'))
		if reference_face:
			# Use adaptive matching with cluster-based multiple references if available
			match_faces = find_match_faces_adaptive([ reference_face ], target_faces, state_manager.get_item('reference_face_distance'), frame_number)
			return match_faces

	# Default: if mode is None or unknown, return all faces (safer than empty list)
	from facefusion import logger
	if face_selector_mode is None:
		logger.warn('face_selector_mode is None, defaulting to all faces', __name__)
	return sort_and_filter_faces(target_faces)


def find_match_faces(reference_faces : List[Face], target_faces : List[Face], face_distance : float) -> List[Face]:
	"""Original matching function - for backward compatibility"""
	match_faces : List[Face] = []

	for reference_face in reference_faces:
		if reference_face:
			for index, target_face in enumerate(target_faces):
				if compare_faces(target_face, reference_face, face_distance):
					match_faces.append(target_faces[index])

	return match_faces


def find_match_faces_adaptive(reference_faces : List[Face], target_faces : List[Face], base_face_distance : float, frame_number : Optional[int] = None) -> List[Face]:
	"""
	Adaptive matching with temporal smoothing and cluster-based multiple references.
	
	Uses:
	- Multiple reference embeddings from cluster database (if available)
	- Adaptive threshold based on recent match history
	- Spatial continuity for better tracking
	"""
	from facefusion import logger
	from facefusion.face_tracker import get_face_tracker, clear_face_tracker
	from facefusion.video_face_database import get_video_face_database, get_cluster
	
	match_faces : List[Face] = []
	database = get_video_face_database()
	tracker = get_face_tracker()
	
	# If we have a database, try to find which cluster the reference face belongs to
	# and use multiple faces from that cluster as references
	cluster_reference_faces = []
	reference_cluster_id = None
	
	if database and reference_faces:
		reference_face = reference_faces[0]  # Use first reference face
		
		# Find which cluster this reference face belongs to by comparing with cluster centroids
		best_cluster_id = None
		best_distance = float('inf')
		
		for cluster in database['clusters']:
			cluster_centroid = cluster['average_embedding']
			ref_embedding = reference_face.embedding_norm
			
			# Calculate distance to cluster centroid
			distance = 1 - numpy.dot(ref_embedding, cluster_centroid)
			distance = float(numpy.interp(distance, [0, 2], [0, 1]))
			
			# Use a threshold to find matching cluster (e.g., 0.4)
			if distance < 0.4 and distance < best_distance:
				best_distance = distance
				best_cluster_id = cluster['cluster_id']
		
		# If we found a matching cluster, use multiple faces from it as references
		if best_cluster_id is not None:
			reference_cluster_id = best_cluster_id
			cluster = get_cluster(best_cluster_id)
			if cluster:
				# Get representative faces from the cluster (best quality ones)
				instances = sorted(cluster['all_instances'], 
				                  key=lambda inst: inst['face'].score_set.get('detector', 0), 
				                  reverse=True)
				# Use top 3-5 faces from cluster as references
				for inst in instances[:5]:
					cluster_reference_faces.append(inst['face'])
				
				if frame_number is None or frame_number < 3:
					logger.debug(f'[face_selector] Using {len(cluster_reference_faces)} reference faces from cluster {best_cluster_id}', __name__)
	
	# Use cluster references if available, otherwise use original reference
	effective_reference_faces = cluster_reference_faces if cluster_reference_faces else reference_faces
	
	# Match target faces using adaptive threshold
	for index, target_face in enumerate(target_faces):
		# Try to find which cluster this target face belongs to
		target_cluster_id = None
		if database and frame_number is not None:
			# Use database lookup for accurate cluster ID
			from facefusion.video_face_database import get_cluster_for_face
			target_cluster_id = get_cluster_for_face(frame_number, index)
		
		# Fallback: find closest cluster by embedding
		if target_cluster_id is None and database:
			best_cluster_id = None
			best_distance = float('inf')
			
			for cluster in database['clusters']:
				cluster_centroid = cluster['average_embedding']
				target_embedding = target_face.embedding_norm
				
				distance = 1 - numpy.dot(target_embedding, cluster_centroid)
				distance = float(numpy.interp(distance, [0, 2], [0, 1]))
				
				if distance < best_distance:
					best_distance = distance
					best_cluster_id = cluster['cluster_id']
			
			# If target is close to a cluster, use that cluster
			if best_cluster_id is not None and best_distance < 0.5:
				target_cluster_id = best_cluster_id
		
		# Get adaptive threshold for this cluster (using frame_number for temporal smoothing)
		if target_cluster_id is not None and frame_number is not None:
			# Use frame_number for proper temporal window calculation (works with parallel processing)
			adaptive_threshold = tracker.get_adaptive_threshold(target_cluster_id, frame_number, base_face_distance)
		else:
			adaptive_threshold = base_face_distance
		
		# Compare with all reference faces (use minimum distance)
		min_distance = float('inf')
		best_reference = None
		
		for ref_face in effective_reference_faces:
			if ref_face:
				distance = calculate_face_distance(target_face, ref_face)
				distance = float(numpy.interp(distance, [0, 2], [0, 1]))
				if distance < min_distance:
					min_distance = distance
					best_reference = ref_face
		
		# Check if match (using adaptive threshold)
		if min_distance < adaptive_threshold:
			match_faces.append(target_faces[index])
			
			# Record successful match for temporal tracking
			if target_cluster_id is not None and frame_number is not None:
				bbox = target_face.bounding_box
				tracker.record_match(frame_number, index, target_cluster_id, tuple(bbox))
		else:
			# Embedding matching failed - try position-based fallback
			# This helps maintain continuity when embedding matching temporarily fails
			use_position_fallback = False
			
			if frame_number is not None:
				bbox = target_face.bounding_box
				
				# If we have a database and reference cluster, check position-based match
				if database and reference_cluster_id is not None:
					position_cluster_id = tracker.find_cluster_by_position(
						tuple(bbox), 
						frame_number,
						max_distance=50.0,  # 50 pixels max distance
						window_size=3  # Check last 3 frames
					)
					
					# Only use position fallback if it matches the reference cluster
					# This prevents matching wrong faces
					if position_cluster_id == reference_cluster_id:
						use_position_fallback = True
						fallback_cluster_id = reference_cluster_id
						
						if frame_number < 5:
							logger.debug(f'[face_selector] Frame {frame_number}: Using position-based fallback for cluster {reference_cluster_id}', __name__)
				
				# If no database but we have target_cluster_id, still check position
				# This helps even when preprocessing wasn't run
				elif target_cluster_id is not None:
					# Check if this cluster was recently matched at nearby position
					# Use find_cluster_by_position which handles locking internally
					position_cluster_id = tracker.find_cluster_by_position(
						tuple(bbox),
						frame_number,
						max_distance=50.0,
						window_size=3
					)
					
					if position_cluster_id == target_cluster_id:
						use_position_fallback = True
						fallback_cluster_id = target_cluster_id
						
						if frame_number < 5:
							logger.debug(f'[face_selector] Frame {frame_number}: Using position-based fallback for cluster {target_cluster_id} (no database)', __name__)
				
				# Apply position-based fallback
				if use_position_fallback:
					match_faces.append(target_faces[index])
					
					# Record as position-based match (still record for tracking)
					tracker.record_match(frame_number, index, fallback_cluster_id, tuple(bbox))
				elif target_cluster_id is not None:
					# Record failed match attempt
					tracker.record_miss(frame_number, target_cluster_id)
			elif target_cluster_id is not None and frame_number is not None:
				# Record failed match attempt
				tracker.record_miss(frame_number, target_cluster_id)
	
	return match_faces


def compare_faces(face : Face, reference_face : Face, face_distance : float) -> bool:
	current_face_distance = calculate_face_distance(face, reference_face)
	current_face_distance = float(numpy.interp(current_face_distance, [ 0, 2 ], [ 0, 1 ]))
	return current_face_distance < face_distance


def calculate_face_distance(face : Face, reference_face : Face) -> float:
	if hasattr(face, 'embedding_norm') and hasattr(reference_face, 'embedding_norm'):
		return 1 - numpy.dot(face.embedding_norm, reference_face.embedding_norm)
	return 0


def sort_and_filter_faces(faces : List[Face]) -> List[Face]:
	if faces:
		if state_manager.get_item('face_selector_order'):
			faces = sort_faces_by_order(faces, state_manager.get_item('face_selector_order'))
		if state_manager.get_item('face_selector_gender'):
			faces = filter_faces_by_gender(faces, state_manager.get_item('face_selector_gender'))
		if state_manager.get_item('face_selector_race'):
			faces = filter_faces_by_race(faces, state_manager.get_item('face_selector_race'))
		if state_manager.get_item('face_selector_age_start') or state_manager.get_item('face_selector_age_end'):
			faces = filter_faces_by_age(faces, state_manager.get_item('face_selector_age_start'), state_manager.get_item('face_selector_age_end'))
	return faces


def sort_faces_by_order(faces : List[Face], order : FaceSelectorOrder) -> List[Face]:
	if order == 'left-right':
		return sorted(faces, key = get_bounding_box_left)
	if order == 'right-left':
		return sorted(faces, key = get_bounding_box_left, reverse = True)
	if order == 'top-bottom':
		return sorted(faces, key = get_bounding_box_top)
	if order == 'bottom-top':
		return sorted(faces, key = get_bounding_box_top, reverse = True)
	if order == 'small-large':
		return sorted(faces, key = get_bounding_box_area)
	if order == 'large-small':
		return sorted(faces, key = get_bounding_box_area, reverse = True)
	if order == 'best-worst':
		return sorted(faces, key = get_face_detector_score, reverse = True)
	if order == 'worst-best':
		return sorted(faces, key = get_face_detector_score)
	return faces


def get_bounding_box_left(face : Face) -> float:
	return face.bounding_box[0]


def get_bounding_box_top(face : Face) -> float:
	return face.bounding_box[1]


def get_bounding_box_area(face : Face) -> float:
	return (face.bounding_box[2] - face.bounding_box[0]) * (face.bounding_box[3] - face.bounding_box[1])


def get_face_detector_score(face : Face) -> Score:
	return face.score_set.get('detector')


def filter_faces_by_gender(faces : List[Face], gender : Gender) -> List[Face]:
	filter_faces = []

	for face in faces:
		if face.gender == gender:
			filter_faces.append(face)
	return filter_faces


def filter_faces_by_age(faces : List[Face], face_selector_age_start : int, face_selector_age_end : int) -> List[Face]:
	filter_faces = []
	age = range(face_selector_age_start, face_selector_age_end)

	for face in faces:
		if set(face.age) & set(age):
			filter_faces.append(face)
	return filter_faces


def filter_faces_by_race(faces : List[Face], race : Race) -> List[Face]:
	filter_faces = []

	for face in faces:
		if face.race == race:
			filter_faces.append(face)
	return filter_faces
