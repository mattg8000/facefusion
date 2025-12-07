from typing import List, Optional, Tuple

import cv2
import gradio
import numpy

from facefusion import logger, state_manager, translator
from facefusion.common_helper import calculate_float_step
from facefusion.filesystem import is_video
from facefusion.temp_helper import resolve_temp_frame_paths
from facefusion.uis.core import get_ui_component, register_ui_component
from facefusion.vision import read_static_image
from facefusion.video_face_database import (
	get_all_clusters,
	get_cluster,
	get_cluster_face_crop,
	get_database_summary,
	get_video_face_database
)

FACE_CLUSTERING_THRESHOLD_SLIDER : Optional[gradio.Slider] = None
FACE_DATABASE_SUMMARY_TEXTBOX : Optional[gradio.Textbox] = None
FACE_DATABASE_CLUSTER_GALLERY : Optional[gradio.Gallery] = None
FACE_DATABASE_FRAME_VIEW_MODAL : Optional[gradio.Column] = None
FACE_DATABASE_FRAME_VIEW_IMAGE : Optional[gradio.Image] = None
FACE_DATABASE_FRAME_VIEW_CLOSE : Optional[gradio.Button] = None
FACE_DATABASE_REFRESH_BUTTON : Optional[gradio.Button] = None
FACE_DATABASE_CLUSTER_DROPDOWN : Optional[gradio.Dropdown] = None

# Store current cluster and instance info for frame view
_current_cluster_id: Optional[int] = None
_current_cluster_instances: List = []


def render() -> None:
	global FACE_CLUSTERING_THRESHOLD_SLIDER
	global FACE_DATABASE_SUMMARY_TEXTBOX
	global FACE_DATABASE_CLUSTER_GALLERY
	global FACE_DATABASE_FRAME_VIEW_MODAL
	global FACE_DATABASE_FRAME_VIEW_IMAGE
	global FACE_DATABASE_FRAME_VIEW_CLOSE
	global FACE_DATABASE_REFRESH_BUTTON
	global FACE_DATABASE_CLUSTER_DROPDOWN

	# Clustering threshold slider
	threshold = state_manager.get_item('face_clustering_threshold')
	if threshold is None:
		threshold = 0.35
	
	FACE_CLUSTERING_THRESHOLD_SLIDER = gradio.Slider(
		label = translator.get('uis.face_clustering_threshold_slider') or 'Face Clustering Threshold',
		value = threshold,
		minimum = 0.2,
		maximum = 0.6,
		step = 0.05,
		info = 'Lower = more clusters (stricter), Higher = fewer clusters (looser). Default: 0.35'
	)
	
	# Database summary textbox
	FACE_DATABASE_SUMMARY_TEXTBOX = gradio.Textbox(
		label = translator.get('uis.face_database_summary') or 'Face Database Summary',
		value = get_database_summary_text(),
		lines = 8,
		interactive = False
	)
	
	# Cluster selection dropdown
	FACE_DATABASE_CLUSTER_DROPDOWN = gradio.Dropdown(
		label = translator.get('uis.face_database_cluster_dropdown') or 'Select Cluster',
		choices = get_cluster_choices(),
		value = None,
		interactive = True,
		info = 'Select a cluster to view its face crop'
	)
	
	# Cluster face gallery
	FACE_DATABASE_CLUSTER_GALLERY = gradio.Gallery(
		label = translator.get('uis.face_database_cluster_gallery') or 'Cluster Face Crops',
		value = None,
		columns = 4,
		rows = 2,
		height = 'auto',
		allow_preview = True,
		show_label = True
	)
	
	# Fullscreen modal for frame view with bounding box
	with gradio.Column(visible = False, elem_classes = ['face-database-frame-modal']) as FACE_DATABASE_FRAME_VIEW_MODAL:
		FACE_DATABASE_FRAME_VIEW_CLOSE = gradio.Button(
			value = '✕ Close',
			variant = 'secondary',
			size = 'sm',
			elem_classes = ['face-database-frame-close']
		)
		FACE_DATABASE_FRAME_VIEW_IMAGE = gradio.Image(
			label = translator.get('uis.face_database_frame_view') or 'Full Frame with Face Bounding Box',
			value = None,
			height = 'auto',
			container = True
		)
	
	# Refresh button
	FACE_DATABASE_REFRESH_BUTTON = gradio.Button(
		value = translator.get('uis.face_database_refresh') or 'Refresh Database View',
		variant = 'secondary'
	)
	
	register_ui_component('face_clustering_threshold_slider', FACE_CLUSTERING_THRESHOLD_SLIDER)
	register_ui_component('face_database_summary_textbox', FACE_DATABASE_SUMMARY_TEXTBOX)
	register_ui_component('face_database_cluster_gallery', FACE_DATABASE_CLUSTER_GALLERY)
	register_ui_component('face_database_frame_view_modal', FACE_DATABASE_FRAME_VIEW_MODAL)
	register_ui_component('face_database_frame_view_image', FACE_DATABASE_FRAME_VIEW_IMAGE)
	register_ui_component('face_database_frame_view_close', FACE_DATABASE_FRAME_VIEW_CLOSE)
	register_ui_component('face_database_refresh_button', FACE_DATABASE_REFRESH_BUTTON)
	register_ui_component('face_database_cluster_dropdown', FACE_DATABASE_CLUSTER_DROPDOWN)


def listen() -> None:
	FACE_CLUSTERING_THRESHOLD_SLIDER.release(update_clustering_threshold, inputs = FACE_CLUSTERING_THRESHOLD_SLIDER)
	FACE_DATABASE_REFRESH_BUTTON.click(
		refresh_database_view,
		outputs = [FACE_DATABASE_SUMMARY_TEXTBOX, FACE_DATABASE_CLUSTER_DROPDOWN, FACE_DATABASE_CLUSTER_GALLERY]
	)
	FACE_DATABASE_CLUSTER_DROPDOWN.change(
		update_cluster_gallery,
		inputs = FACE_DATABASE_CLUSTER_DROPDOWN,
		outputs = FACE_DATABASE_CLUSTER_GALLERY
	)
	
	# When a face crop is selected in the gallery, show full frame with bounding box in modal
	if FACE_DATABASE_CLUSTER_GALLERY:
		FACE_DATABASE_CLUSTER_GALLERY.select(
			show_full_frame_with_bbox_modal,
			inputs = FACE_DATABASE_CLUSTER_DROPDOWN,
			outputs = [FACE_DATABASE_FRAME_VIEW_MODAL, FACE_DATABASE_FRAME_VIEW_IMAGE]
		)
	
	# Close button for modal
	if FACE_DATABASE_FRAME_VIEW_CLOSE:
		FACE_DATABASE_FRAME_VIEW_CLOSE.click(
			close_frame_modal,
			outputs = FACE_DATABASE_FRAME_VIEW_MODAL
		)
	
	# Update database view when target video changes
	target_video = get_ui_component('target_video')
	if target_video:
		for method in ['change', 'clear']:
			getattr(target_video, method)(clear_database_view, outputs = [FACE_DATABASE_SUMMARY_TEXTBOX, FACE_DATABASE_CLUSTER_DROPDOWN, FACE_DATABASE_CLUSTER_GALLERY])
	
	# Auto-refresh when output video is updated (processing completed)
	output_video = get_ui_component('output_video')
	if output_video:
		output_video.change(
			refresh_database_view,
			outputs = [FACE_DATABASE_SUMMARY_TEXTBOX, FACE_DATABASE_CLUSTER_DROPDOWN, FACE_DATABASE_CLUSTER_GALLERY],
			show_progress = 'hidden'
		)


def update_clustering_threshold(threshold : float) -> None:
	state_manager.set_item('face_clustering_threshold', threshold)
	logger.debug(f'Clustering threshold updated to {threshold}', __name__)


def refresh_database_view() -> Tuple[gradio.Textbox, gradio.Dropdown, gradio.Gallery]:
	summary_text = get_database_summary_text()
	cluster_choices = get_cluster_choices()
	
	# Update summary
	summary_textbox = gradio.Textbox(value = summary_text)
	
	# Update cluster dropdown
	cluster_dropdown = gradio.Dropdown(
		choices = cluster_choices,
		value = cluster_choices[0] if cluster_choices else None
	)
	
	# Update gallery with first cluster if available
	gallery_value = None
	if cluster_choices:
		cluster_id = int(cluster_choices[0].split(':')[0])
		gallery_value = get_cluster_gallery_images(cluster_id)
	
	return summary_textbox, cluster_dropdown, gradio.Gallery(value = gallery_value)


def update_cluster_gallery(cluster_choice : Optional[str]) -> gradio.Gallery:
	if not cluster_choice:
		return gradio.Gallery(value = None)
	
	try:
		# Extract cluster ID from choice string (format: "0: 123 instances")
		cluster_id = int(cluster_choice.split(':')[0])
		gallery_images = get_cluster_gallery_images(cluster_id)
		return gradio.Gallery(value = gallery_images)
	except (ValueError, IndexError):
		logger.error(f'Failed to parse cluster choice: {cluster_choice}', __name__)
		return gradio.Gallery(value = None)


def clear_database_view() -> Tuple[gradio.Textbox, gradio.Dropdown, gradio.Gallery]:
	return (
		gradio.Textbox(value = 'No database available. Process a video first.'),
		gradio.Dropdown(choices = [], value = None),
		gradio.Gallery(value = None)
	)


def get_database_summary_text() -> str:
	database = get_video_face_database()
	if database is None:
		return 'No database available. Process a video first to build the face database.'
	
	summary = get_database_summary()
	if summary is None:
		return 'Database exists but summary unavailable.'
	
	text = f"""Face Database Summary
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total Clusters: {summary['total_clusters']}
Total Face Instances: {summary['total_faces']}
Clustering Threshold: {summary['clustering_threshold']:.2f}

Cluster Statistics:
  • Largest cluster: {summary['largest_cluster']} instances
  • Smallest cluster: {summary['smallest_cluster']} instances
  • Average cluster size: {summary['average_cluster_size']:.1f} instances

Note: Adjust threshold and reprocess video to change clustering.
"""
	return text


def get_cluster_choices() -> List[str]:
	clusters = get_all_clusters()
	if not clusters:
		return []
	
	# Sort by size (largest first)
	sorted_clusters = sorted(clusters, key=lambda c: c['instance_count'], reverse=True)
	
	choices = []
	for cluster in sorted_clusters:
		cluster_id = cluster['cluster_id']
		instance_count = cluster['instance_count']
		frame_range = cluster['frame_range']
		choices.append(f"{cluster_id}: {instance_count} instances (frames {frame_range[0]}-{frame_range[1]})")
	
	return choices


def get_cluster_gallery_images(cluster_id: int) -> List[numpy.ndarray]:
	"""Get face crop images for a specific cluster"""
	global _current_cluster_id, _current_cluster_instances
	
	cluster = get_cluster(cluster_id)
	if cluster is None:
		return []
	
	target_path = state_manager.get_item('target_path')
	if not target_path or not is_video(target_path):
		return []
	
	# Store cluster info for frame view
	_current_cluster_id = cluster_id
	_current_cluster_instances = cluster['all_instances']
	
	# Get face crops from multiple instances in the cluster
	# Limit to first 8 instances to avoid too many images
	instances = cluster['all_instances'][:8]
	gallery_images = []
	
	for instance in instances:
		try:
			from facefusion.video_face_database import extract_face_crop
			face_crop = extract_face_crop(target_path, instance['frame_number'], instance, crop_size=(256, 256))
			
			if face_crop is not None:
				# Convert BGR to RGB for display
				face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
				gallery_images.append(face_crop_rgb)
		except Exception as e:
			logger.debug(f'Failed to extract face crop for instance: {e}', __name__)
			continue
	
	# If no images extracted, try representative face
	if not gallery_images:
		face_crop = get_cluster_face_crop(target_path, cluster_id, crop_size=(256, 256))
		if face_crop is not None:
			face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
			gallery_images.append(face_crop_rgb)
	
	return gallery_images


def close_frame_modal() -> gradio.Column:
	"""Close the frame view modal"""
	return gradio.Column(visible = False)


def show_full_frame_with_bbox_modal(select_data: gradio.SelectData, cluster_dropdown: Optional[str] = None) -> Tuple[gradio.Column, gradio.Image]:
	"""Show full frame with red bounding box in a fullscreen modal"""
	"""Show full frame with red bounding box around the selected face"""
	global _current_cluster_id, _current_cluster_instances
	
	if select_data is None or select_data.index is None:
		return gradio.Column(visible = False), gradio.Image(value = None)
	
	# Get the selected instance index from gallery
	selected_index = select_data.index
	
	# Get cluster ID from dropdown or stored value
	cluster_id = None
	if cluster_dropdown:
		try:
			cluster_id = int(cluster_dropdown.split(':')[0])
		except (ValueError, IndexError):
			pass
	
	if cluster_id is None:
		cluster_id = _current_cluster_id
	
	if cluster_id is None:
		return gradio.Column(visible = False), gradio.Image(value = None)
	
	# Get cluster and instances
	cluster = get_cluster(cluster_id)
	if cluster is None:
		return gradio.Column(visible = False), gradio.Image(value = None)
	
	instances = cluster['all_instances'][:8]  # Same limit as gallery
	if selected_index >= len(instances):
		return gradio.Column(visible = False), gradio.Image(value = None)
	
	instance = instances[selected_index]
	frame_number = instance['frame_number']
	bbox = instance['bounding_box']
	
	# Get the full frame from extracted frames
	target_path = state_manager.get_item('target_path')
	if not target_path or not is_video(target_path):
		return gradio.Column(visible = False), gradio.Image(value = None)
	
	# Try to read from extracted frames first (faster)
	temp_frame_paths = resolve_temp_frame_paths(target_path)
	full_frame = None
	
	if temp_frame_paths and frame_number < len(temp_frame_paths):
		full_frame = read_static_image(temp_frame_paths[frame_number])
	
	# If not available from extracted frames, fall back to video
	if full_frame is None:
		from facefusion.vision import read_video_frame
		full_frame = read_video_frame(target_path, frame_number)
	
	if full_frame is None:
		return gradio.Column(visible = False), gradio.Image(value = None)
	
	# Draw red bounding box
	x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
	
	# Ensure coordinates are within frame bounds
	height, width = full_frame.shape[:2]
	x1 = max(0, min(x1, width - 1))
	y1 = max(0, min(y1, height - 1))
	x2 = max(0, min(x2, width - 1))
	y2 = max(0, min(y2, height - 1))
	
	# Draw rectangle (BGR format: red = (0, 0, 255))
	frame_with_bbox = full_frame.copy()
	cv2.rectangle(frame_with_bbox, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red box, 3px thick
	
	# Convert BGR to RGB for display
	frame_with_bbox_rgb = cv2.cvtColor(frame_with_bbox, cv2.COLOR_BGR2RGB)
	
	# Return modal (visible) and image
	return gradio.Column(visible = True), gradio.Image(value = frame_with_bbox_rgb)

