from typing import Dict, List, Optional, Tuple

import cv2
import gradio
import numpy

from facefusion import logger, state_manager, translator
from facefusion.common_helper import get_first
from facefusion.face_analyser import get_one_face
from facefusion.filesystem import filter_image_paths, has_image, is_video
from facefusion.face_analyser import get_many_faces
from facefusion.face_selector import sort_faces_by_order
from facefusion.uis.core import get_ui_component, register_ui_component
from facefusion.vision import read_static_image, read_static_images
from facefusion.video_face_database import (
	get_all_clusters,
	get_cluster,
	get_cluster_face_crop,
	get_video_face_database
)

FACE_MAPPING_WRAPPER : Optional[gradio.Column] = None
FACE_MAPPING_SOURCE_GALLERY : Optional[gradio.Gallery] = None
FACE_MAPPING_TARGET_GALLERY : Optional[gradio.Gallery] = None
FACE_MAPPING_ASSIGN_BUTTON : Optional[gradio.Button] = None
FACE_MAPPING_CLEAR_BUTTON : Optional[gradio.Button] = None
FACE_MAPPING_STATUS_TEXT : Optional[gradio.Textbox] = None

# Store selected indices
_selected_source_index: Optional[int] = None
_selected_target_cluster_id: Optional[int] = None


def render() -> None:
	global FACE_MAPPING_WRAPPER
	global FACE_MAPPING_SOURCE_GALLERY
	global FACE_MAPPING_TARGET_GALLERY
	global FACE_MAPPING_ASSIGN_BUTTON
	global FACE_MAPPING_CLEAR_BUTTON
	global FACE_MAPPING_STATUS_TEXT
	
	# Only show mapping UI for videos with database
	database = get_video_face_database()
	source_paths = state_manager.get_item('source_paths') or []
	source_image_paths = filter_image_paths(source_paths)
	has_multiple_sources = len(source_image_paths) > 1
	is_video_target = is_video(state_manager.get_item('target_path'))
	show_mapping = database is not None and has_multiple_sources and is_video_target
	
	logger.debug(f'Rendering face_mapping: show={show_mapping}, database={database is not None}, source_paths={len(source_paths)}, source_images={len(source_image_paths)}, has_multiple={has_multiple_sources}, is_video={is_video_target}', __name__)
	
	with gradio.Column(visible = show_mapping) as FACE_MAPPING_WRAPPER:
		FACE_MAPPING_STATUS_TEXT = gradio.Textbox(
			label = translator.get('uis.face_mapping_status') or 'Face Mapping Status',
			value = get_mapping_status_text(),
			lines = 2,
			interactive = False
		)
		
		with gradio.Row():
			with gradio.Column(scale = 1):
				source_images = get_source_face_images()
				logger.debug(f'Rendering source gallery with {len(source_images)} images', __name__)
				FACE_MAPPING_SOURCE_GALLERY = gradio.Gallery(
					label = translator.get('uis.face_mapping_source_gallery') or 'Source Faces (Click to select)',
					value = source_images if source_images else None,
					columns = 2,
					rows = 2,
					height = 'auto',
					allow_preview = True,
					show_label = True
				)
			
			with gradio.Column(scale = 1):
				target_images = get_target_cluster_images()
				logger.debug(f'Rendering target gallery with {len(target_images)} images', __name__)
				FACE_MAPPING_TARGET_GALLERY = gradio.Gallery(
					label = translator.get('uis.face_mapping_target_gallery') or 'Target Face Clusters (Click to select)',
					value = target_images if target_images else None,
					columns = 2,
					rows = 2,
					height = 'auto',
					allow_preview = True,
					show_label = True
				)
		
		with gradio.Row():
			FACE_MAPPING_ASSIGN_BUTTON = gradio.Button(
				value = translator.get('uis.face_mapping_assign') or 'Assign Source to Cluster',
				variant = 'primary',
				size = 'sm'
			)
			FACE_MAPPING_CLEAR_BUTTON = gradio.Button(
				value = translator.get('uis.face_mapping_clear') or 'Clear All Mappings',
				variant = 'secondary',
				size = 'sm'
			)
	
	register_ui_component('face_mapping_wrapper', FACE_MAPPING_WRAPPER)
	register_ui_component('face_mapping_source_gallery', FACE_MAPPING_SOURCE_GALLERY)
	register_ui_component('face_mapping_target_gallery', FACE_MAPPING_TARGET_GALLERY)
	register_ui_component('face_mapping_assign_button', FACE_MAPPING_ASSIGN_BUTTON)
	register_ui_component('face_mapping_clear_button', FACE_MAPPING_CLEAR_BUTTON)
	register_ui_component('face_mapping_status_text', FACE_MAPPING_STATUS_TEXT)


def listen() -> None:
	FACE_MAPPING_SOURCE_GALLERY.select(select_source_face, outputs = FACE_MAPPING_STATUS_TEXT)
	FACE_MAPPING_TARGET_GALLERY.select(select_target_cluster, outputs = FACE_MAPPING_STATUS_TEXT)
	FACE_MAPPING_ASSIGN_BUTTON.click(
		assign_source_to_cluster,
		outputs = [FACE_MAPPING_STATUS_TEXT, FACE_MAPPING_TARGET_GALLERY]
	)
	FACE_MAPPING_CLEAR_BUTTON.click(
		clear_all_mappings,
		outputs = [FACE_MAPPING_STATUS_TEXT, FACE_MAPPING_TARGET_GALLERY]
	)
	
	# Update when source or target changes
	source_file = get_ui_component('source_file')
	if source_file:
		source_file.change(refresh_source_gallery, outputs = FACE_MAPPING_SOURCE_GALLERY)
	
	target_video = get_ui_component('target_video')
	if target_video:
		target_video.change(refresh_target_gallery, outputs = FACE_MAPPING_TARGET_GALLERY)
	
	# Update when database is refreshed - use .then() to chain after face_database refresh
	face_database_refresh = get_ui_component('face_database_refresh_button')
	if face_database_refresh:
		# Chain after the face_database refresh completes
		face_database_refresh.click(
			refresh_after_database_update,
			outputs = [FACE_MAPPING_WRAPPER, FACE_MAPPING_SOURCE_GALLERY, FACE_MAPPING_TARGET_GALLERY, FACE_MAPPING_STATUS_TEXT],
			show_progress = 'hidden'
		)


def refresh_source_gallery() -> gradio.Gallery:
	"""Refresh the source face gallery"""
	source_images = get_source_face_images()
	logger.debug(f'Refreshing source gallery: {len(source_images)} images', __name__)
	return gradio.Gallery(value = source_images if source_images else None)


def refresh_target_gallery() -> gradio.Gallery:
	"""Refresh the target cluster gallery"""
	target_images = get_target_cluster_images()
	logger.debug(f'Refreshing target gallery: {len(target_images)} images', __name__)
	return gradio.Gallery(value = target_images if target_images else None)


def refresh_after_database_update() -> Tuple[gradio.Column, gradio.Gallery, gradio.Gallery, gradio.Textbox]:
	"""Refresh the entire mapping UI after database is updated"""
	database = get_video_face_database()
	has_multiple_sources = len(filter_image_paths(state_manager.get_item('source_paths') or [])) > 1
	is_video_target = is_video(state_manager.get_item('target_path'))
	show_mapping = database is not None and has_multiple_sources and is_video_target
	
	logger.debug(f'Refreshing mapping UI after database update: show={show_mapping}, database={database is not None}, sources={has_multiple_sources}, video={is_video_target}', __name__)
	
	source_images = get_source_face_images() if show_mapping else None
	target_images = get_target_cluster_images() if show_mapping else None
	
	return (
		gradio.Column(visible = show_mapping),
		gradio.Gallery(value = source_images if source_images else None),
		gradio.Gallery(value = target_images if target_images else None),
		gradio.Textbox(value = get_mapping_status_text())
	)


def update_mapping_ui() -> Tuple[gradio.Column, gradio.Gallery, gradio.Textbox]:
	"""Update mapping UI visibility and content"""
	database = get_video_face_database()
	has_multiple_sources = len(filter_image_paths(state_manager.get_item('source_paths') or [])) > 1
	is_video_target = is_video(state_manager.get_item('target_path'))
	show_mapping = database is not None and has_multiple_sources and is_video_target
	
	source_images = get_source_face_images() if show_mapping else None
	logger.debug(f'Updating mapping UI: show={show_mapping}, sources={len(source_images) if source_images else 0}', __name__)
	
	return (
		gradio.Column(visible = show_mapping),
		gradio.Gallery(value = source_images),
		gradio.Textbox(value = get_mapping_status_text())
	)


def select_source_face(select_data: gradio.SelectData) -> gradio.Textbox:
	"""Handle source face selection"""
	global _selected_source_index
	
	if select_data and select_data.index is not None:
		_selected_source_index = select_data.index
		source_paths = filter_image_paths(state_manager.get_item('source_paths') or [])
		if _selected_source_index < len(source_paths):
			source_name = source_paths[_selected_source_index].split('/')[-1]
			return gradio.Textbox(value = f'Selected source: {source_name}\nNow select a target cluster to assign it to.')
	
	return gradio.Textbox(value = get_mapping_status_text())


def select_target_cluster(select_data: gradio.SelectData) -> gradio.Textbox:
	"""Handle target cluster selection"""
	global _selected_target_cluster_id
	
	if select_data and select_data.index is not None:
		clusters = get_all_clusters()
		if clusters and select_data.index < len(clusters):
			_selected_target_cluster_id = clusters[select_data.index]['cluster_id']
			cluster = clusters[select_data.index]
			return gradio.Textbox(value = f'Selected cluster: {cluster["cluster_id"]} ({cluster["instance_count"]} instances)\nClick "Assign" to map the selected source to this cluster.')
	
	return gradio.Textbox(value = get_mapping_status_text())


def assign_source_to_cluster() -> Tuple[gradio.Textbox, gradio.Gallery]:
	"""Assign selected source face to selected target cluster"""
	global _selected_source_index, _selected_target_cluster_id
	
	if _selected_source_index is None or _selected_target_cluster_id is None:
		return gradio.Textbox(value = 'Please select both a source face and a target cluster.'), gradio.Gallery()
	
	# Get or create mapping
	cluster_source_mapping = state_manager.get_item('cluster_source_mapping') or {}
	# Ensure cluster_id is stored as integer (not string) for consistent lookup
	cluster_source_mapping[int(_selected_target_cluster_id)] = int(_selected_source_index)
	state_manager.set_item('cluster_source_mapping', cluster_source_mapping)
	
	logger.info(f'[face_mapping] Assigned cluster {_selected_target_cluster_id} → source {_selected_source_index}. Mapping: {cluster_source_mapping}', __name__)
	
	# Get cluster info for status
	cluster = get_cluster(_selected_target_cluster_id)
	cluster_info = f'Cluster {_selected_target_cluster_id} ({cluster["instance_count"]} instances)' if cluster else f'Cluster {_selected_target_cluster_id}'
	
	source_paths = filter_image_paths(state_manager.get_item('source_paths') or [])
	source_name = source_paths[_selected_source_index].split('/')[-1] if _selected_source_index < len(source_paths) else f'Source {_selected_source_index}'
	
	status_text = f'✓ Assigned: {source_name} → {cluster_info}\n{get_mapping_status_text()}'
	
	# Reset selections
	_selected_source_index = None
	_selected_target_cluster_id = None
	
	# Update target gallery to show assigned status
	updated_target_gallery = get_target_cluster_images()
	
	return gradio.Textbox(value = status_text), gradio.Gallery(value = updated_target_gallery)


def clear_all_mappings() -> Tuple[gradio.Textbox, gradio.Gallery]:
	"""Clear all cluster-to-source mappings"""
	state_manager.set_item('cluster_source_mapping', {})
	_selected_source_index = None
	_selected_target_cluster_id = None
	
	updated_target_gallery = get_target_cluster_images()
	return gradio.Textbox(value = 'All mappings cleared.'), gradio.Gallery(value = updated_target_gallery)


def get_mapping_status_text() -> str:
	"""Get current mapping status text"""
	cluster_source_mapping = state_manager.get_item('cluster_source_mapping') or {}
	
	if not cluster_source_mapping:
		return 'No mappings assigned. Select a source face, then a target cluster, then click "Assign".'
	
	mapping_count = len(cluster_source_mapping)
	return f'{mapping_count} mapping(s) assigned. Ready to process.'


def get_source_face_images() -> List[numpy.ndarray]:
	"""Get face crop images from source images"""
	source_paths = filter_image_paths(state_manager.get_item('source_paths') or [])
	if not source_paths:
		logger.debug('No source image paths found', __name__)
		return []
	
	logger.debug(f'Extracting faces from {len(source_paths)} source images', __name__)
	source_images = []
	for source_path in source_paths:
		try:
			source_frame = read_static_image(source_path)
			if source_frame is None:
				logger.warn(f'Could not read source image: {source_path}', __name__)
				continue
			
			faces = get_many_faces([source_frame])
			if not faces:
				logger.warn(f'No faces detected in source image: {source_path}', __name__)
				continue
			
			faces = sort_faces_by_order(faces, 'large-small')
			
			# Extract face crop
			face = get_first(faces)
			bbox = face.bounding_box
			x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
			
			# Ensure coordinates are within bounds
			height, width = source_frame.shape[:2]
			x1 = max(0, min(x1, width - 1))
			y1 = max(0, min(y1, height - 1))
			x2 = max(x1 + 1, min(x2, width))
			y2 = max(y1 + 1, min(y2, height))
			
			if x2 > x1 and y2 > y1:
				face_crop = source_frame[y1:y2, x1:x2]
				if face_crop.size > 0:
					face_crop = cv2.resize(face_crop, (256, 256), interpolation=cv2.INTER_LINEAR)
					face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
					source_images.append(face_crop_rgb)
					logger.debug(f'Successfully extracted face from {source_path}', __name__)
				else:
					logger.warn(f'Empty face crop from {source_path}', __name__)
			else:
				logger.warn(f'Invalid bounding box from {source_path}: ({x1}, {y1}, {x2}, {y2})', __name__)
		except Exception as e:
			logger.error(f'Failed to extract face from source {source_path}: {e}', __name__)
			import traceback
			logger.error(traceback.format_exc(), __name__)
			continue
	
	logger.debug(f'Extracted {len(source_images)} source faces', __name__)
	return source_images


def get_target_cluster_images() -> List[numpy.ndarray]:
	"""Get face crop images for target clusters"""
	clusters = get_all_clusters()
	if not clusters:
		logger.debug('No clusters found in database', __name__)
		return []
	
	logger.debug(f'Getting images for {len(clusters)} clusters', __name__)
	
	# Sort by cluster ID
	sorted_clusters = sorted(clusters, key=lambda c: c['cluster_id'])
	
	cluster_images = []
	target_path = state_manager.get_item('target_path')
	if not target_path:
		logger.warn('No target path found', __name__)
		return []
	
	cluster_source_mapping = state_manager.get_item('cluster_source_mapping') or {}
	
	for cluster in sorted_clusters:
		try:
			cluster_id = cluster['cluster_id']
			
			# Get representative face crop
			face_crop = get_cluster_face_crop(target_path, cluster_id, crop_size=(256, 256))
			
			if face_crop is None:
				logger.warn(f'Could not get face crop for cluster {cluster_id}', __name__)
				continue
			
			# Convert BGR to RGB
			face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
			
			# Draw indicator if mapped
			if cluster_id in cluster_source_mapping:
				source_index = cluster_source_mapping[cluster_id]
				# Draw green border or indicator
				cv2.rectangle(face_crop_rgb, (0, 0), (255, 255), (0, 255, 0), 5)
			
			cluster_images.append(face_crop_rgb)
		except Exception as e:
			logger.error(f'Failed to get cluster {cluster["cluster_id"]} image: {e}', __name__)
			import traceback
			logger.error(traceback.format_exc(), __name__)
			continue
	
	logger.debug(f'Returning {len(cluster_images)} cluster images', __name__)
	return cluster_images

