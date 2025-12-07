"""
Video Face Database - Preprocessing module for face clustering and identification

This module scans all video frames upfront, extracts faces, and clusters them
by similarity to build a global face database. This enables:
- Multi-face replacement with proper face identification
- Temporal tracking across appearance changes
- Per-cluster settings optimization
"""

from typing import Dict, List, Optional, Tuple, TypedDict
import os
import pickle

import cv2
import numpy
from tqdm import tqdm

from facefusion import logger, state_manager
from facefusion.face_analyser import get_many_faces
# calculate_face_distance imported from face_selector if needed, but we calculate directly here
from facefusion.types import Face, VisionFrame
from facefusion.vision import count_video_frame_total, read_static_image, read_video_frame


# Data structures for face database
FaceInstance = TypedDict('FaceInstance',
{
	'frame_number': int,
	'face_index': int,
	'face': Face,
	'distance_to_centroid': float,
	# Spatial information
	'bounding_box': Tuple[float, float, float, float],  # [x1, y1, x2, y2]
	'center_x': float,
	'center_y': float,
	'width': float,
	'height': float,
	'area': float
})

FaceCluster = TypedDict('FaceCluster',
{
	'cluster_id': int,
	'representative_face': Face,  # Best quality or centroid
	'all_instances': List[FaceInstance],
	'average_embedding': numpy.ndarray,  # Cluster centroid
	'frame_range': Tuple[int, int],  # First and last appearance
	'instance_count': int
})

VideoFaceDatabase = TypedDict('VideoFaceDatabase',
{
	'clusters': List[FaceCluster],
	'face_to_cluster': Dict[Tuple[int, int], int],  # (frame_number, face_index) -> cluster_id
	'clustering_threshold': float,
	'total_faces': int
})

# Global database storage
VIDEO_FACE_DATABASE: Optional[VideoFaceDatabase] = None


def get_video_face_database() -> Optional[VideoFaceDatabase]:
	"""Get the current video face database if it exists"""
	return VIDEO_FACE_DATABASE


def clear_video_face_database() -> None:
	"""Clear the video face database"""
	global VIDEO_FACE_DATABASE
	VIDEO_FACE_DATABASE = None


def preprocess_video_faces(video_path: str, trim_frame_start: int, trim_frame_end: int, clustering_threshold: Optional[float] = None) -> VideoFaceDatabase:
	"""
	Preprocess video: scan all frames, extract faces, and cluster by similarity.
	
	Args:
		video_path: Path to the video file
		trim_frame_start: First frame to process
		trim_frame_end: Last frame to process
		clustering_threshold: Distance threshold for clustering (0.0-1.0). If None, uses state_manager value or default 0.35
	
	Returns:
		VideoFaceDatabase with clustered faces
	"""
	global VIDEO_FACE_DATABASE
	
	# Get threshold from state_manager if not provided
	if clustering_threshold is None:
		clustering_threshold = state_manager.get_item('face_clustering_threshold')
		if clustering_threshold is None:
			clustering_threshold = 0.35  # Default
			logger.debug(f'Using default clustering threshold: {clustering_threshold}', __name__)
		else:
			logger.debug(f'Using clustering threshold from state_manager: {clustering_threshold}', __name__)
	
	logger.info('Preprocessing video faces...', __name__)
	
	# Step 1: Scan all frames and extract faces
	all_face_instances = scan_video_faces(video_path, trim_frame_start, trim_frame_end)
	
	if not all_face_instances:
		logger.warn('No faces found in video', __name__)
		VIDEO_FACE_DATABASE = {
			'clusters': [],
			'face_to_cluster': {},
			'clustering_threshold': clustering_threshold,
			'total_faces': 0
		}
		return VIDEO_FACE_DATABASE
	
	# Step 2: Cluster faces by embedding similarity
	clusters = cluster_faces(all_face_instances, clustering_threshold)
	
	# Step 3: Build mapping from face instances to clusters
	face_to_cluster = {}
	for cluster in clusters:
		for instance in cluster['all_instances']:
			face_to_cluster[(instance['frame_number'], instance['face_index'])] = cluster['cluster_id']
	
	VIDEO_FACE_DATABASE = {
		'clusters': clusters,
		'face_to_cluster': face_to_cluster,
		'clustering_threshold': clustering_threshold,
		'total_faces': len(all_face_instances)
	}
	
	logger.info(f'Preprocessing complete: {len(clusters)} clusters, {len(all_face_instances)} total faces', __name__)
	
	# Log database summary for inspection
	summary = get_database_summary()
	if summary:
		logger.info(f'Database summary: {summary["total_clusters"]} clusters, largest: {summary["largest_cluster"]} instances, smallest: {summary["smallest_cluster"]} instances, avg: {summary["average_cluster_size"]:.1f} instances', __name__)
		
		# Warn if too many clusters (suggests threshold too strict)
		if summary['total_clusters'] > len(all_face_instances) * 0.1:
			logger.warn(f'Many clusters detected ({summary["total_clusters"]}). Consider increasing clustering threshold from {clustering_threshold} to 0.4-0.5', __name__)
	
	return VIDEO_FACE_DATABASE


def preprocess_extracted_frames(temp_frame_paths: List[str], clustering_threshold: Optional[float] = None) -> VideoFaceDatabase:
	"""
	Preprocess extracted frames: scan all frame files, extract faces, and cluster by similarity.
	This is faster than preprocess_video_faces because it reads from disk instead of video seeking.
	
	Args:
		temp_frame_paths: List of paths to extracted frame image files (sorted by frame number)
		clustering_threshold: Distance threshold for clustering (0.0-1.0). If None, uses state_manager value or default 0.35
	
	Returns:
		VideoFaceDatabase with clustered faces
	"""
	global VIDEO_FACE_DATABASE
	
	# Get threshold from state_manager if not provided
	if clustering_threshold is None:
		clustering_threshold = state_manager.get_item('face_clustering_threshold')
		if clustering_threshold is None:
			clustering_threshold = 0.35  # Default
			logger.debug(f'Using default clustering threshold: {clustering_threshold}', __name__)
		else:
			logger.debug(f'Using clustering threshold from state_manager: {clustering_threshold}', __name__)
	
	logger.info('Preprocessing extracted frames...', __name__)
	
	# Step 1: Scan all extracted frames and extract faces
	all_face_instances = scan_extracted_frames(temp_frame_paths)
	
	if not all_face_instances:
		logger.warn('No faces found in extracted frames', __name__)
		VIDEO_FACE_DATABASE = {
			'clusters': [],
			'face_to_cluster': {},
			'clustering_threshold': clustering_threshold,
			'total_faces': 0
		}
		return VIDEO_FACE_DATABASE
	
	# Step 2: Cluster faces by embedding similarity
	clusters = cluster_faces(all_face_instances, clustering_threshold)
	
	# Step 3: Build mapping from face instances to clusters
	face_to_cluster = {}
	for cluster in clusters:
		for instance in cluster['all_instances']:
			face_to_cluster[(instance['frame_number'], instance['face_index'])] = cluster['cluster_id']
	
	VIDEO_FACE_DATABASE = {
		'clusters': clusters,
		'face_to_cluster': face_to_cluster,
		'clustering_threshold': clustering_threshold,
		'total_faces': len(all_face_instances)
	}
	
	logger.info(f'Preprocessing complete: {len(clusters)} clusters, {len(all_face_instances)} total faces', __name__)
	
	# Log database summary for inspection
	summary = get_database_summary()
	if summary:
		logger.info(f'Database summary: {summary["total_clusters"]} clusters, largest: {summary["largest_cluster"]} instances, smallest: {summary["smallest_cluster"]} instances, avg: {summary["average_cluster_size"]:.1f} instances', __name__)
		
		# Warn if too many clusters (suggests threshold too strict)
		if summary['total_clusters'] > len(all_face_instances) * 0.1:
			logger.warn(f'Many clusters detected ({summary["total_clusters"]}). Consider increasing clustering threshold from {clustering_threshold} to 0.4-0.5', __name__)
	
	return VIDEO_FACE_DATABASE


def scan_extracted_frames(temp_frame_paths: List[str]) -> List[FaceInstance]:
	"""
	Scan all extracted frame files and extract all faces with their embeddings.
	This is faster than scan_video_faces because it reads from disk images instead of video seeking.
	
	Args:
		temp_frame_paths: List of paths to extracted frame image files (sorted by frame number)
	
	Returns:
		List of FaceInstance objects
	"""
	all_face_instances: List[FaceInstance] = []
	
	logger.debug(f'Scanning {len(temp_frame_paths)} extracted frames for faces', __name__)
	
	with tqdm(total=len(temp_frame_paths), desc='Scanning faces', unit='frame', 
	          ascii=' =', disable=state_manager.get_item('log_level') in ['warn', 'error']) as progress:
		
		for frame_number, frame_path in enumerate(temp_frame_paths):
			# Read frame from disk (much faster than video seeking)
			vision_frame = read_static_image(frame_path)
			
			if vision_frame is not None and numpy.any(vision_frame):
				# Extract faces from this frame
				faces = get_many_faces([vision_frame])
				
				# Store each face with its frame and index
				for face_index, face in enumerate(faces):
					# Extract spatial information from bounding box
					bbox = face.bounding_box
					x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
					width = x2 - x1
					height = y2 - y1
					center_x = (x1 + x2) / 2.0
					center_y = (y1 + y2) / 2.0
					area = width * height
					
					all_face_instances.append({
						'frame_number': frame_number,
						'face_index': face_index,
						'face': face,
						'distance_to_centroid': 0.0,  # Will be calculated during clustering
						'bounding_box': (x1, y1, x2, y2),
						'center_x': center_x,
						'center_y': center_y,
						'width': width,
						'height': height,
						'area': area
					})
			
			progress.update()
	
	return all_face_instances


def scan_video_faces(video_path: str, trim_frame_start: int, trim_frame_end: int) -> List[FaceInstance]:
	"""
	Scan all frames in video and extract all faces with their embeddings.
	
	Returns:
		List of FaceInstance tuples: (frame_number, face_index, face_object)
	"""
	all_face_instances: List[FaceInstance] = []
	frame_total = count_video_frame_total(video_path)
	frame_range = range(trim_frame_start, min(trim_frame_end, frame_total))
	
	logger.debug(f'Scanning {len(frame_range)} frames for faces', __name__)
	
	with tqdm(total=len(frame_range), desc='Scanning faces', unit='frame', 
	          ascii=' =', disable=state_manager.get_item('log_level') in ['warn', 'error']) as progress:
		
		for frame_number in frame_range:
			vision_frame = read_video_frame(video_path, frame_number)
			
			if vision_frame is not None and numpy.any(vision_frame):
				# Extract faces from this frame
				faces = get_many_faces([vision_frame])
				
				# Store each face with its frame and index
				for face_index, face in enumerate(faces):
					# Extract spatial information from bounding box
					bbox = face.bounding_box
					x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
					width = x2 - x1
					height = y2 - y1
					center_x = (x1 + x2) / 2.0
					center_y = (y1 + y2) / 2.0
					area = width * height
					
					all_face_instances.append({
						'frame_number': frame_number,
						'face_index': face_index,
						'face': face,
						'distance_to_centroid': 0.0,  # Will be calculated during clustering
						'bounding_box': (x1, y1, x2, y2),
						'center_x': center_x,
						'center_y': center_y,
						'width': width,
						'height': height,
						'area': area
					})
			
			progress.update()
	
	return all_face_instances


def cluster_faces(all_face_instances: List[FaceInstance], threshold: float) -> List[FaceCluster]:
	"""
	Cluster faces by embedding similarity using hierarchical clustering approach.
	
	Args:
		all_face_instances: List of all face instances from video
		threshold: Distance threshold for clustering (0.0-1.0)
	
	Returns:
		List of FaceCluster objects
	"""
	if not all_face_instances:
		return []
	
	logger.debug(f'Clustering {len(all_face_instances)} face instances', __name__)
	
	# Simple hierarchical clustering: start with each face as own cluster, merge similar ones
	clusters: List[FaceCluster] = []
	cluster_id_counter = 0
	
	# Initialize: each face starts as its own cluster
	for instance in all_face_instances:
		face = instance['face']
		clusters.append({
			'cluster_id': cluster_id_counter,
			'representative_face': face,  # Start with first face as representative
			'all_instances': [instance],
			'average_embedding': face.embedding_norm.copy(),  # Start with first embedding
			'frame_range': (instance['frame_number'], instance['frame_number']),
			'instance_count': 1
		})
		cluster_id_counter += 1
	
	# Merge clusters that are similar enough
	merged = True
	while merged:
		merged = False
		clusters_to_remove = []
		
		for i in range(len(clusters)):
			if i in clusters_to_remove:
				continue
			
			for j in range(i + 1, len(clusters)):
				if j in clusters_to_remove:
					continue
				
				# Calculate distance between cluster centroids
				cluster_i_centroid = clusters[i]['average_embedding']
				cluster_j_centroid = clusters[j]['average_embedding']
				
				# Use same distance calculation as face_selector
				distance = 1 - numpy.dot(cluster_i_centroid, cluster_j_centroid)
				distance = float(numpy.interp(distance, [0, 2], [0, 1]))
				
				# If clusters are similar enough, merge them
				if distance < threshold:
					# Merge cluster j into cluster i
					clusters[i]['all_instances'].extend(clusters[j]['all_instances'])
					clusters[i]['instance_count'] = len(clusters[i]['all_instances'])
					
					# Update frame range
					frame_numbers = [inst['frame_number'] for inst in clusters[i]['all_instances']]
					clusters[i]['frame_range'] = (min(frame_numbers), max(frame_numbers))
					
					# Recalculate centroid (average of all embeddings)
					all_embeddings = [inst['face'].embedding_norm for inst in clusters[i]['all_instances']]
					clusters[i]['average_embedding'] = numpy.mean(all_embeddings, axis=0)
					clusters[i]['average_embedding'] = clusters[i]['average_embedding'] / numpy.linalg.norm(clusters[i]['average_embedding'])
					
					# Update representative face (use face with highest detector score)
					best_face = max(clusters[i]['all_instances'], 
					                key=lambda inst: inst['face'].score_set.get('detector', 0))
					clusters[i]['representative_face'] = best_face['face']
					
					# Mark cluster j for removal
					clusters_to_remove.append(j)
					merged = True
					break
		
		# Remove merged clusters (in reverse order to maintain indices)
		for idx in sorted(clusters_to_remove, reverse=True):
			clusters.pop(idx)
	
	# Update cluster IDs to be sequential
	for idx, cluster in enumerate(clusters):
		cluster['cluster_id'] = idx
	
	logger.debug(f'Clustering complete: {len(clusters)} clusters', __name__)
	
	return clusters


def get_cluster_for_face(frame_number: int, face_index: int) -> Optional[int]:
	"""
	Get the cluster ID for a specific face instance.
	
	Args:
		frame_number: Frame number where face appears
		face_index: Index of face in that frame
	
	Returns:
		Cluster ID if found, None otherwise
	"""
	if VIDEO_FACE_DATABASE is None:
		return None
	
	return VIDEO_FACE_DATABASE['face_to_cluster'].get((frame_number, face_index))


def get_cluster(cluster_id: int) -> Optional[FaceCluster]:
	"""
	Get a specific cluster by ID.
	
	Args:
		cluster_id: ID of the cluster
	
	Returns:
		FaceCluster if found, None otherwise
	"""
	if VIDEO_FACE_DATABASE is None:
		return None
	
	for cluster in VIDEO_FACE_DATABASE['clusters']:
		if cluster['cluster_id'] == cluster_id:
			return cluster
	
	return None


def get_database_summary() -> Optional[Dict]:
	"""
	Get a summary of the face database for inspection/debugging.
	
	Returns:
		Dictionary with database statistics, or None if database doesn't exist
	"""
	if VIDEO_FACE_DATABASE is None:
		return None
	
	clusters = VIDEO_FACE_DATABASE['clusters']
	
	# Calculate statistics
	cluster_sizes = [c['instance_count'] for c in clusters]
	
	return {
		'total_clusters': len(clusters),
		'total_faces': VIDEO_FACE_DATABASE['total_faces'],
		'clustering_threshold': VIDEO_FACE_DATABASE['clustering_threshold'],
		'largest_cluster': max(cluster_sizes) if cluster_sizes else 0,
		'smallest_cluster': min(cluster_sizes) if cluster_sizes else 0,
		'average_cluster_size': sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0,
		'cluster_details': [
			{
				'cluster_id': c['cluster_id'],
				'instance_count': c['instance_count'],
				'frame_range': c['frame_range']
			}
			for c in clusters
		]
	}


def get_all_clusters() -> List[FaceCluster]:
	"""
	Get all clusters from the database.
	
	Returns:
		List of all FaceCluster objects, or empty list if database doesn't exist
	"""
	if VIDEO_FACE_DATABASE is None:
		return []
	
	return VIDEO_FACE_DATABASE['clusters']


def save_database(file_path: str) -> bool:
	"""
	Save the face database to a file using pickle.
	
	Args:
		file_path: Path where to save the database
	
	Returns:
		True if successful, False otherwise
	"""
	global VIDEO_FACE_DATABASE
	
	if VIDEO_FACE_DATABASE is None:
		logger.warn('No database to save', __name__)
		return False
	
	try:
		# Create directory if it doesn't exist
		os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
		
		with open(file_path, 'wb') as f:
			pickle.dump(VIDEO_FACE_DATABASE, f)
		
		logger.info(f'Database saved to {file_path}', __name__)
		return True
	except Exception as e:
		logger.error(f'Failed to save database: {e}', __name__)
		return False


def load_database(file_path: str) -> bool:
	"""
	Load a face database from a file.
	
	Args:
		file_path: Path to the database file
	
	Returns:
		True if successful, False otherwise
	"""
	global VIDEO_FACE_DATABASE
	
	if not os.path.exists(file_path):
		logger.warn(f'Database file not found: {file_path}', __name__)
		return False
	
	try:
		with open(file_path, 'rb') as f:
			VIDEO_FACE_DATABASE = pickle.load(f)
		
		logger.info(f'Database loaded from {file_path}', __name__)
		logger.info(f'Loaded: {len(VIDEO_FACE_DATABASE["clusters"])} clusters, {VIDEO_FACE_DATABASE["total_faces"]} total faces', __name__)
		return True
	except Exception as e:
		logger.error(f'Failed to load database: {e}', __name__)
		return False


def extract_face_crop(video_path: str, frame_number: int, face_instance: FaceInstance, crop_size: Tuple[int, int] = (512, 512)) -> Optional[numpy.ndarray]:
	"""
	Extract a cropped face image from a video frame.
	
	Args:
		video_path: Path to the video file
		frame_number: Frame number to extract from
		face_instance: FaceInstance containing the face to extract
		crop_size: Size of the output crop (width, height)
	
	Returns:
		Cropped face image as numpy array, or None if extraction fails
	"""
	try:
		# Read the frame
		vision_frame = read_video_frame(video_path, frame_number)
		if vision_frame is None or not numpy.any(vision_frame):
			return None
		
		# Get bounding box
		bbox = face_instance['bounding_box']
		x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
		
		# Ensure coordinates are within frame bounds
		height, width = vision_frame.shape[:2]
		x1 = max(0, min(x1, width))
		y1 = max(0, min(y1, height))
		x2 = max(0, min(x2, width))
		y2 = max(0, min(y2, height))
		
		if x2 <= x1 or y2 <= y1:
			return None
		
		# Extract crop
		face_crop = vision_frame[y1:y2, x1:x2]
		
		# Resize to crop_size if needed
		if crop_size:
			face_crop = cv2.resize(face_crop, crop_size, interpolation=cv2.INTER_LINEAR)
		
		return face_crop
	except Exception as e:
		logger.error(f'Failed to extract face crop: {e}', __name__)
		return None


def get_cluster_face_crop(video_path: str, cluster_id: int, crop_size: Tuple[int, int] = (512, 512)) -> Optional[numpy.ndarray]:
	"""
	Extract a face crop for a cluster's representative face.
	
	Args:
		video_path: Path to the video file
		cluster_id: ID of the cluster
		crop_size: Size of the output crop (width, height)
	
	Returns:
		Cropped face image as numpy array, or None if extraction fails
	"""
	cluster = get_cluster(cluster_id)
	if cluster is None:
		return None
	
	# Use the representative face instance
	# Find the instance that matches the representative face
	representative_face = cluster['representative_face']
	
	# Find the instance with this face (or use first instance)
	instance = cluster['all_instances'][0]
	
	return extract_face_crop(video_path, instance['frame_number'], instance, crop_size)


def save_cluster_face_crops(video_path: str, output_dir: str, crop_size: Tuple[int, int] = (512, 512)) -> int:
	"""
	Save face crops for all clusters to disk.
	
	Args:
		video_path: Path to the video file
		output_dir: Directory where to save the crops
		crop_size: Size of each crop (width, height)
	
	Returns:
		Number of crops saved
	"""
	if VIDEO_FACE_DATABASE is None:
		logger.warn('No database available', __name__)
		return 0
	
	os.makedirs(output_dir, exist_ok=True)
	
	saved_count = 0
	clusters = VIDEO_FACE_DATABASE['clusters']
	
	for cluster in clusters:
		cluster_id = cluster['cluster_id']
		
		# Get representative face crop
		face_crop = get_cluster_face_crop(video_path, cluster_id, crop_size)
		
		if face_crop is not None:
			output_path = os.path.join(output_dir, f'cluster_{cluster_id:03d}.jpg')
			cv2.imwrite(output_path, face_crop)
			saved_count += 1
	
	logger.info(f'Saved {saved_count} face crops to {output_dir}', __name__)
	return saved_count

