from typing import List, Optional, Tuple

import numpy

from facefusion import state_manager
from facefusion.common_helper import get_first
from facefusion.face_classifier import classify_face
from facefusion.face_detector import detect_faces, detect_faces_by_angle
from facefusion.face_helper import apply_nms, convert_to_face_landmark_5, estimate_face_angle, get_nms_threshold
from facefusion.face_landmarker import detect_face_landmark, estimate_face_landmark_68_5
from facefusion.face_recognizer import calculate_face_embedding
from facefusion.face_store import get_static_faces, set_static_faces
from facefusion.types import BoundingBox, Face, FaceLandmark5, FaceLandmarkSet, FaceScoreSet, Score, VisionFrame


def create_faces(vision_frame : VisionFrame, bounding_boxes : List[BoundingBox], face_scores : List[Score], face_landmarks_5 : List[FaceLandmark5]) -> List[Face]:
	faces = []
	nms_threshold = get_nms_threshold(state_manager.get_item('face_detector_model'), state_manager.get_item('face_detector_angles'))
	keep_indices = apply_nms(bounding_boxes, face_scores, state_manager.get_item('face_detector_score'), nms_threshold)

	for index in keep_indices:
		bounding_box = bounding_boxes[index]
		face_score = face_scores[index]
		face_landmark_5 = face_landmarks_5[index]
		face_landmark_5_68 = face_landmark_5
		face_landmark_68_5 = estimate_face_landmark_68_5(face_landmark_5_68)
		face_landmark_68 = face_landmark_68_5
		face_landmark_score_68 = 0.0
		face_angle = estimate_face_angle(face_landmark_68_5)

		if state_manager.get_item('face_landmarker_score') > 0:
			face_landmark_68, face_landmark_score_68 = detect_face_landmark(vision_frame, bounding_box, face_angle)
		if face_landmark_score_68 > state_manager.get_item('face_landmarker_score'):
			face_landmark_5_68 = convert_to_face_landmark_5(face_landmark_68)

		face_landmark_set : FaceLandmarkSet =\
		{
			'5': face_landmark_5,
			'5/68': face_landmark_5_68,
			'68': face_landmark_68,
			'68/5': face_landmark_68_5
		}
		face_score_set : FaceScoreSet =\
		{
			'detector': face_score,
			'landmarker': face_landmark_score_68
		}
		face_embedding, face_embedding_norm = calculate_face_embedding(vision_frame, face_landmark_set.get('5/68'))
		gender, age, race = classify_face(vision_frame, face_landmark_set.get('5/68'))
		faces.append(Face(
			bounding_box = bounding_box,
			score_set = face_score_set,
			landmark_set = face_landmark_set,
			angle = face_angle,
			embedding = face_embedding,
			embedding_norm = face_embedding_norm,
			gender = gender,
			age = age,
			race = race
		))
	return faces


def get_one_face(faces : List[Face], position : int = 0) -> Optional[Face]:
	if faces:
		position = min(position, len(faces) - 1)
		return faces[position]
	return None


def get_average_face(faces : List[Face]) -> Optional[Face]:
	face_embeddings = []
	face_embeddings_norm = []

	if faces:
		first_face = get_first(faces)

		for face in faces:
			face_embeddings.append(face.embedding)
			face_embeddings_norm.append(face.embedding_norm)

		return Face(
			bounding_box = first_face.bounding_box,
			score_set = first_face.score_set,
			landmark_set = first_face.landmark_set,
			angle = first_face.angle,
			embedding = numpy.mean(face_embeddings, axis = 0),
			embedding_norm = numpy.mean(face_embeddings_norm, axis = 0),
			gender = first_face.gender,
			age = first_face.age,
			race = first_face.race
		)
	return None


def get_many_faces(vision_frames : List[VisionFrame]) -> List[Face]:
	many_faces : List[Face] = []

	for vision_frame in vision_frames:
		if numpy.any(vision_frame):
			static_faces = get_static_faces(vision_frame)
			if static_faces:
				many_faces.extend(static_faces)
			else:
				all_bounding_boxes = []
				all_face_scores = []
				all_face_landmarks_5 = []

				for face_detector_angle in state_manager.get_item('face_detector_angles'):
					if face_detector_angle == 0:
						bounding_boxes, face_scores, face_landmarks_5 = detect_faces(vision_frame)
					else:
						bounding_boxes, face_scores, face_landmarks_5 = detect_faces_by_angle(vision_frame, face_detector_angle)
					all_bounding_boxes.extend(bounding_boxes)
					all_face_scores.extend(face_scores)
					all_face_landmarks_5.extend(face_landmarks_5)

				if all_bounding_boxes and all_face_scores and all_face_landmarks_5 and state_manager.get_item('face_detector_score') > 0:
					faces = create_faces(vision_frame, all_bounding_boxes, all_face_scores, all_face_landmarks_5)

					if faces:
						many_faces.extend(faces)
						set_static_faces(vision_frame, faces)
	return many_faces


def detect_face_in_region(vision_frame : VisionFrame, search_region : Tuple[float, float, float, float], detector_score : float = 0.1, detector_model : Optional[str] = None) -> Optional[Face]:
	"""
	Force face detection in a specific region with relaxed settings.
	
	Args:
		vision_frame: Frame to search
		search_region: Bounding box area to search (x1, y1, x2, y2)
		detector_score: Very low score threshold (default 0.1)
		detector_model: Optional specific detector model
	
	Returns:
		Detected face or None
	"""
	from facefusion import logger
	
	x1, y1, x2, y2 = search_region
	frame_height, frame_width = vision_frame.shape[:2]
	
	# Validate and clamp bounding box to frame bounds
	x1 = max(0, min(x1, frame_width))
	y1 = max(0, min(y1, frame_height))
	x2 = max(x1 + 1, min(x2, frame_width))
	y2 = max(y1 + 1, min(y2, frame_height))
	
	if x2 <= x1 or y2 <= y1:
		logger.warn(f'Invalid search region: {search_region}', __name__)
		return None
	
	# Expand search region by 20% (padding)
	width = x2 - x1
	height = y2 - y1
	padding_x = width * 0.2
	padding_y = height * 0.2
	
	expanded_region = (
		max(0, x1 - padding_x),
		max(0, y1 - padding_y),
		min(frame_width, x2 + padding_x),
		min(frame_height, y2 + padding_y)
	)
	
	# Crop frame to expanded region
	expanded_x1, expanded_y1, expanded_x2, expanded_y2 = map(int, expanded_region)
	cropped_frame = vision_frame[expanded_y1:expanded_y2, expanded_x1:expanded_x2]
	
	if not numpy.any(cropped_frame):
		logger.warn(f'Cropped frame is empty for region {expanded_region}', __name__)
		return None
	
	# Save original detector score and model
	original_score = state_manager.get_item('face_detector_score')
	original_model = state_manager.get_item('face_detector_model')
	
	# Temporarily set very relaxed score
	state_manager.set_item('face_detector_score', detector_score)
	if detector_model:
		state_manager.set_item('face_detector_model', detector_model)
	
	try:
		# Detect faces in cropped region
		all_bounding_boxes = []
		all_face_scores = []
		all_face_landmarks_5 = []
		
		for face_detector_angle in state_manager.get_item('face_detector_angles'):
			if face_detector_angle == 0:
				bounding_boxes, face_scores, face_landmarks_5 = detect_faces(cropped_frame)
			else:
				bounding_boxes, face_scores, face_landmarks_5 = detect_faces_by_angle(cropped_frame, face_detector_angle)
			all_bounding_boxes.extend(bounding_boxes)
			all_face_scores.extend(face_scores)
			all_face_landmarks_5.extend(face_landmarks_5)
		
		# Find face closest to center of original search region
		center_x = (x1 + x2) / 2 - expanded_x1
		center_y = (y1 + y2) / 2 - expanded_y1
		
		best_face = None
		best_distance = float('inf')
		
		if all_bounding_boxes and all_face_scores and all_face_landmarks_5:
			faces = create_faces(cropped_frame, all_bounding_boxes, all_face_scores, all_face_landmarks_5)
			
			for face in faces:
				bbox = face.bounding_box
				face_center_x = (bbox[0] + bbox[2]) / 2
				face_center_y = (bbox[1] + bbox[3]) / 2
				distance = ((face_center_x - center_x)**2 + (face_center_y - center_y)**2)**0.5
				
				if distance < best_distance:
					best_distance = distance
					best_face = face
		
		# Adjust bounding box back to full frame coordinates
		if best_face:
			adjusted_bbox = numpy.array([
				best_face.bounding_box[0] + expanded_x1,
				best_face.bounding_box[1] + expanded_y1,
				best_face.bounding_box[2] + expanded_x1,
				best_face.bounding_box[3] + expanded_y1
			])
			
			# Adjust landmarks back to full frame coordinates
			adjusted_landmark_set = {}
			for key, landmarks in best_face.landmark_set.items():
				if landmarks is not None and len(landmarks) > 0:
					adjusted_landmarks = landmarks.copy()
					adjusted_landmarks[:, 0] += expanded_x1
					adjusted_landmarks[:, 1] += expanded_y1
					adjusted_landmark_set[key] = adjusted_landmarks
				else:
					adjusted_landmark_set[key] = landmarks
			
			# Create new Face with adjusted coordinates
			best_face = Face(
				bounding_box = adjusted_bbox,
				score_set = best_face.score_set,
				landmark_set = adjusted_landmark_set,
				angle = best_face.angle,
				embedding = best_face.embedding,
				embedding_norm = best_face.embedding_norm,
				gender = best_face.gender,
				age = best_face.age,
				race = best_face.race
			)
			
			# Recalculate embedding on full frame for accuracy
			if best_face.landmark_set.get('5/68') is not None:
				face_embedding, face_embedding_norm = calculate_face_embedding(vision_frame, best_face.landmark_set.get('5/68'))
				best_face = best_face._replace(
					embedding = face_embedding,
					embedding_norm = face_embedding_norm
				)
		
		return best_face
	finally:
		# Restore original detector score and model
		state_manager.set_item('face_detector_score', original_score)
		if detector_model:
			state_manager.set_item('face_detector_model', original_model)


def scale_face(target_face : Face, target_vision_frame : VisionFrame, temp_vision_frame : VisionFrame) -> Face:
	scale_x = temp_vision_frame.shape[1] / target_vision_frame.shape[1]
	scale_y = temp_vision_frame.shape[0] / target_vision_frame.shape[0]

	bounding_box = target_face.bounding_box * [ scale_x, scale_y, scale_x, scale_y ]
	landmark_set =\
	{
		'5': target_face.landmark_set.get('5') * numpy.array([ scale_x, scale_y ]),
		'5/68': target_face.landmark_set.get('5/68') * numpy.array([ scale_x, scale_y ]),
		'68': target_face.landmark_set.get('68') * numpy.array([ scale_x, scale_y ]),
		'68/5': target_face.landmark_set.get('68/5') * numpy.array([ scale_x, scale_y ])
	}

	return target_face._replace(
		bounding_box = bounding_box,
		landmark_set = landmark_set
	)
