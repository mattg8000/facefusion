"""
Forced Face Replacement UI Component

Allows users to manually force face detection and replacement for specific frames
where automatic detection fails.
"""

import gradio
import numpy
import cv2
from typing import List, Optional, Tuple

from facefusion import logger, state_manager, translator
from facefusion.filesystem import is_video, filter_image_paths
from facefusion.forced_replacements import (
	add_forced_replacement,
	clear_forced_replacements,
	get_forced_replacements_for_frame,
	load_forced_replacements,
	remove_forced_replacement,
	save_forced_replacements
)
from facefusion.face_analyser import detect_face_in_region, get_many_faces
from facefusion.face_selector import sort_faces_by_order
from facefusion.common_helper import get_first
from facefusion.uis.core import get_ui_component, register_ui_component
from facefusion.vision import read_video_frame, read_static_image
from facefusion.temp_helper import resolve_temp_frame_paths

FORCED_REPLACEMENT_WRAPPER : Optional[gradio.Column] = None
FORCED_REPLACEMENT_FRAME_SLIDER : Optional[gradio.Slider] = None
FORCED_REPLACEMENT_FRAME_IMAGE : Optional[gradio.Image] = None
FORCED_REPLACEMENT_BBOX_INPUT : Optional[gradio.Textbox] = None
FORCED_REPLACEMENT_SOURCE_DROPDOWN : Optional[gradio.Dropdown] = None
FORCED_REPLACEMENT_DETECT_BUTTON : Optional[gradio.Button] = None
FORCED_REPLACEMENT_SAVE_BUTTON : Optional[gradio.Button] = None
FORCED_REPLACEMENT_LIST : Optional[gradio.Dataframe] = None
FORCED_REPLACEMENT_CLEAR_BUTTON : Optional[gradio.Button] = None
FORCED_REPLACEMENT_RESET_BBOX_BUTTON : Optional[gradio.Button] = None
FORCED_REPLACEMENT_SET_FROM_CLICK_BUTTON : Optional[gradio.Button] = None
FORCED_REPLACEMENT_BBOX_X1 : Optional[gradio.Number] = None
FORCED_REPLACEMENT_BBOX_Y1 : Optional[gradio.Number] = None
FORCED_REPLACEMENT_BBOX_X2 : Optional[gradio.Number] = None
FORCED_REPLACEMENT_BBOX_Y2 : Optional[gradio.Number] = None

# Store click coordinates for bounding box drawing
_bbox_start_point: Optional[Tuple[int, int]] = None
_click_mode_active: bool = False


def render() -> None:
	global FORCED_REPLACEMENT_WRAPPER
	global FORCED_REPLACEMENT_FRAME_SLIDER
	global FORCED_REPLACEMENT_FRAME_IMAGE
	global FORCED_REPLACEMENT_BBOX_INPUT
	global FORCED_REPLACEMENT_SOURCE_DROPDOWN
	global FORCED_REPLACEMENT_DETECT_BUTTON
	global FORCED_REPLACEMENT_SAVE_BUTTON
	global FORCED_REPLACEMENT_LIST
	global FORCED_REPLACEMENT_CLEAR_BUTTON
	global FORCED_REPLACEMENT_RESET_BBOX_BUTTON
	global FORCED_REPLACEMENT_SET_FROM_CLICK_BUTTON
	global FORCED_REPLACEMENT_BBOX_X1
	global FORCED_REPLACEMENT_BBOX_Y1
	global FORCED_REPLACEMENT_BBOX_X2
	global FORCED_REPLACEMENT_BBOX_Y2
	
	# Only show for videos
	is_video_target = is_video(state_manager.get_item('target_path'))
	
	with gradio.Column(visible=is_video_target) as FORCED_REPLACEMENT_WRAPPER:
		gradio.Markdown("### Manual Face Replacement Override")
		gradio.Markdown("Force face detection for frames where automatic detection fails.")
		
		# Get max frames from video if available
		max_frames = 100  # Default
		target_path = state_manager.get_item('target_path')
		if target_path and is_video(target_path):
			from facefusion.vision import count_video_frame_total
			try:
				max_frames = max(0, count_video_frame_total(target_path) - 1)
			except:
				pass
		
		FORCED_REPLACEMENT_FRAME_SLIDER = gradio.Slider(
			label=translator.get('uis.forced_replacement_frame') or 'Frame Number',
			minimum=0,
			maximum=max_frames,
			value=0,
			step=1,
			info='Navigate to the frame where detection failed'
		)
		
		FORCED_REPLACEMENT_FRAME_IMAGE = gradio.Image(
			label=translator.get('uis.forced_replacement_preview') or 'Frame Preview (Click to set bounding box corners)',
			value=None,
			height=400,
			type='numpy',
			show_download_button=False
		)
		
		with gradio.Row():
			FORCED_REPLACEMENT_BBOX_X1 = gradio.Number(
				label='X1 (left)',
				value=None,
				precision=0,
				minimum=0
			)
			FORCED_REPLACEMENT_BBOX_Y1 = gradio.Number(
				label='Y1 (top)',
				value=None,
				precision=0,
				minimum=0
			)
		with gradio.Row():
			FORCED_REPLACEMENT_BBOX_X2 = gradio.Number(
				label='X2 (right)',
				value=None,
				precision=0,
				minimum=0
			)
			FORCED_REPLACEMENT_BBOX_Y2 = gradio.Number(
				label='Y2 (bottom)',
				value=None,
				precision=0,
				minimum=0
			)
		
		FORCED_REPLACEMENT_BBOX_INPUT = gradio.Textbox(
			label=translator.get('uis.forced_replacement_bbox') or 'Bounding Box (x1, y1, x2, y2) - Auto-filled from inputs above',
			value='',
			placeholder='Auto-filled from coordinate inputs',
			interactive=False,
			info='Coordinates are automatically combined from the inputs above, or enter manually in format: x1, y1, x2, y2'
		)
		
		with gradio.Row():
			FORCED_REPLACEMENT_SET_FROM_CLICK_BUTTON = gradio.Button(
				value=translator.get('uis.forced_replacement_set_from_click') or 'Click Mode: Set Coordinates',
				variant='secondary',
				size='sm'
			)
			FORCED_REPLACEMENT_RESET_BBOX_BUTTON = gradio.Button(
				value=translator.get('uis.forced_replacement_reset_bbox') or 'Reset',
				variant='secondary',
				size='sm'
			)
		
		# Initialize dropdown with current source paths
		initial_choices = get_source_dropdown_choices()
		logger.debug(f'Initializing source dropdown with {len(initial_choices)} choices: {initial_choices}', __name__)
		FORCED_REPLACEMENT_SOURCE_DROPDOWN = gradio.Dropdown(
			label=translator.get('uis.forced_replacement_source') or 'Source Face',
			choices=initial_choices,
			value=initial_choices[0] if initial_choices else None,
			info='Select which source face to use for replacement'
		)
		
		with gradio.Row():
			FORCED_REPLACEMENT_DETECT_BUTTON = gradio.Button(
				value=translator.get('uis.forced_replacement_detect') or 'Detect Face in Area',
				variant='secondary',
				size='sm'
			)
			FORCED_REPLACEMENT_SAVE_BUTTON = gradio.Button(
				value=translator.get('uis.forced_replacement_save') or 'Save Forced Replacement',
				variant='primary',
				size='sm'
			)
		
		FORCED_REPLACEMENT_LIST = gradio.Dataframe(
			label=translator.get('uis.forced_replacement_list') or 'Saved Forced Replacements',
			value=None,
			headers=['Frame', 'Bounding Box', 'Source', 'Actions'],
			interactive=False
		)
		
		with gradio.Row():
			FORCED_REPLACEMENT_RESET_BBOX_BUTTON = gradio.Button(
				value=translator.get('uis.forced_replacement_reset_bbox') or 'Reset Selection',
				variant='secondary',
				size='sm'
			)
			FORCED_REPLACEMENT_CLEAR_BUTTON = gradio.Button(
				value=translator.get('uis.forced_replacement_clear') or 'Clear All',
				variant='secondary',
				size='sm'
			)
	
	register_ui_component('forced_replacement_wrapper', FORCED_REPLACEMENT_WRAPPER)
	register_ui_component('forced_replacement_frame_slider', FORCED_REPLACEMENT_FRAME_SLIDER)
	register_ui_component('forced_replacement_frame_image', FORCED_REPLACEMENT_FRAME_IMAGE)
	register_ui_component('forced_replacement_bbox_input', FORCED_REPLACEMENT_BBOX_INPUT)
	register_ui_component('forced_replacement_source_dropdown', FORCED_REPLACEMENT_SOURCE_DROPDOWN)
	register_ui_component('forced_replacement_detect_button', FORCED_REPLACEMENT_DETECT_BUTTON)
	register_ui_component('forced_replacement_save_button', FORCED_REPLACEMENT_SAVE_BUTTON)
	register_ui_component('forced_replacement_list', FORCED_REPLACEMENT_LIST)
	register_ui_component('forced_replacement_clear_button', FORCED_REPLACEMENT_CLEAR_BUTTON)
	register_ui_component('forced_replacement_reset_bbox_button', FORCED_REPLACEMENT_RESET_BBOX_BUTTON)
	register_ui_component('forced_replacement_set_from_click_button', FORCED_REPLACEMENT_SET_FROM_CLICK_BUTTON)
	register_ui_component('forced_replacement_bbox_x1', FORCED_REPLACEMENT_BBOX_X1)
	register_ui_component('forced_replacement_bbox_y1', FORCED_REPLACEMENT_BBOX_Y1)
	register_ui_component('forced_replacement_bbox_x2', FORCED_REPLACEMENT_BBOX_X2)
	register_ui_component('forced_replacement_bbox_y2', FORCED_REPLACEMENT_BBOX_Y2)


def listen() -> None:
	FORCED_REPLACEMENT_FRAME_SLIDER.change(
		update_frame_preview,
		inputs=FORCED_REPLACEMENT_FRAME_SLIDER,
		outputs=FORCED_REPLACEMENT_FRAME_IMAGE
	)
	
	# Handle bounding box selection from image clicks
	# Try using select event - if it doesn't work, we'll use coordinate inputs
	if FORCED_REPLACEMENT_FRAME_IMAGE:
		try:
			FORCED_REPLACEMENT_FRAME_IMAGE.select(
				handle_bbox_selection,
				inputs=FORCED_REPLACEMENT_FRAME_SLIDER,
				outputs=[FORCED_REPLACEMENT_BBOX_X1, FORCED_REPLACEMENT_BBOX_Y1, FORCED_REPLACEMENT_BBOX_X2, FORCED_REPLACEMENT_BBOX_Y2, FORCED_REPLACEMENT_FRAME_IMAGE]
			)
		except Exception as e:
			logger.warn(f'Image select event not available: {e}. Using coordinate inputs instead.', __name__)
	
	# Update bbox text input when coordinate inputs change
	if FORCED_REPLACEMENT_BBOX_X1 and FORCED_REPLACEMENT_BBOX_Y1 and FORCED_REPLACEMENT_BBOX_X2 and FORCED_REPLACEMENT_BBOX_Y2:
		def update_bbox_from_coords(x1, y1, x2, y2):
			if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
				bbox_str = f'{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}'
				updated_image = update_preview_with_bbox(
					state_manager.get_item('_forced_replacement_current_frame') or 0,
					bbox_str
				)
				return gradio.Textbox(value=bbox_str), updated_image
			return gradio.Textbox(value=''), update_frame_preview(state_manager.get_item('_forced_replacement_current_frame') or 0)
		
		FORCED_REPLACEMENT_BBOX_X1.change(
			lambda x1, y1, x2, y2: update_bbox_from_coords(x1, y1, x2, y2),
			inputs=[FORCED_REPLACEMENT_BBOX_X1, FORCED_REPLACEMENT_BBOX_Y1, FORCED_REPLACEMENT_BBOX_X2, FORCED_REPLACEMENT_BBOX_Y2],
			outputs=[FORCED_REPLACEMENT_BBOX_INPUT, FORCED_REPLACEMENT_FRAME_IMAGE]
		)
		FORCED_REPLACEMENT_BBOX_Y1.change(
			lambda y1, x1, x2, y2: update_bbox_from_coords(x1, y1, x2, y2),
			inputs=[FORCED_REPLACEMENT_BBOX_Y1, FORCED_REPLACEMENT_BBOX_X1, FORCED_REPLACEMENT_BBOX_X2, FORCED_REPLACEMENT_BBOX_Y2],
			outputs=[FORCED_REPLACEMENT_BBOX_INPUT, FORCED_REPLACEMENT_FRAME_IMAGE]
		)
		FORCED_REPLACEMENT_BBOX_X2.change(
			lambda x2, x1, y1, y2: update_bbox_from_coords(x1, y1, x2, y2),
			inputs=[FORCED_REPLACEMENT_BBOX_X2, FORCED_REPLACEMENT_BBOX_X1, FORCED_REPLACEMENT_BBOX_Y1, FORCED_REPLACEMENT_BBOX_Y2],
			outputs=[FORCED_REPLACEMENT_BBOX_INPUT, FORCED_REPLACEMENT_FRAME_IMAGE]
		)
		FORCED_REPLACEMENT_BBOX_Y2.change(
			lambda y2, x1, y1, x2: update_bbox_from_coords(x1, y1, x2, y2),
			inputs=[FORCED_REPLACEMENT_BBOX_Y2, FORCED_REPLACEMENT_BBOX_X1, FORCED_REPLACEMENT_BBOX_Y1, FORCED_REPLACEMENT_BBOX_X2],
			outputs=[FORCED_REPLACEMENT_BBOX_INPUT, FORCED_REPLACEMENT_FRAME_IMAGE]
		)
	
	# Update preview when bbox text input changes (manual entry)
	FORCED_REPLACEMENT_BBOX_INPUT.change(
		lambda bbox_str, fn: update_preview_with_bbox(fn, bbox_str) if bbox_str else update_frame_preview(fn),
		inputs=[FORCED_REPLACEMENT_BBOX_INPUT, FORCED_REPLACEMENT_FRAME_SLIDER],
		outputs=FORCED_REPLACEMENT_FRAME_IMAGE
	)
	
	# Toggle click mode
	FORCED_REPLACEMENT_SET_FROM_CLICK_BUTTON.click(
		toggle_click_mode,
		outputs=[FORCED_REPLACEMENT_SET_FROM_CLICK_BUTTON, FORCED_REPLACEMENT_FRAME_IMAGE]
	)
	
	FORCED_REPLACEMENT_DETECT_BUTTON.click(
		detect_face_in_area,
		inputs=[FORCED_REPLACEMENT_FRAME_SLIDER, FORCED_REPLACEMENT_BBOX_INPUT],
		outputs=[FORCED_REPLACEMENT_FRAME_IMAGE, FORCED_REPLACEMENT_BBOX_INPUT]
	)
	
	FORCED_REPLACEMENT_SAVE_BUTTON.click(
		save_forced_replacement,
		inputs=[FORCED_REPLACEMENT_FRAME_SLIDER, FORCED_REPLACEMENT_BBOX_INPUT, FORCED_REPLACEMENT_SOURCE_DROPDOWN],
		outputs=[FORCED_REPLACEMENT_LIST, FORCED_REPLACEMENT_BBOX_INPUT, FORCED_REPLACEMENT_SOURCE_DROPDOWN]
	)
	
	FORCED_REPLACEMENT_CLEAR_BUTTON.click(
		clear_all_replacements,
		outputs=FORCED_REPLACEMENT_LIST
	)
	
	FORCED_REPLACEMENT_RESET_BBOX_BUTTON.click(
		reset_bbox_selection,
		inputs=FORCED_REPLACEMENT_FRAME_SLIDER,
		outputs=[FORCED_REPLACEMENT_BBOX_INPUT, FORCED_REPLACEMENT_FRAME_IMAGE, FORCED_REPLACEMENT_BBOX_X1, FORCED_REPLACEMENT_BBOX_Y1, FORCED_REPLACEMENT_BBOX_X2, FORCED_REPLACEMENT_BBOX_Y2]
	)
	
	# Update source dropdown when source changes
	# Listen to source_image changes (which gets updated when source files are uploaded)
	source_image = get_ui_component('source_image')
	if source_image:
		source_image.change(
			update_source_dropdown,
			outputs=FORCED_REPLACEMENT_SOURCE_DROPDOWN
		)
		source_image.clear(
			update_source_dropdown,
			outputs=FORCED_REPLACEMENT_SOURCE_DROPDOWN
		)
	
	# Also try to get source_file directly (it might not be registered)
	# We'll use a workaround: check state_manager periodically or on other events
	# For now, the source_image change should handle most cases
	
	# Update frame preview when source dropdown changes (preview with swapped face)
	if FORCED_REPLACEMENT_SOURCE_DROPDOWN:
		FORCED_REPLACEMENT_SOURCE_DROPDOWN.change(
			update_preview_with_source_face,
			inputs=[FORCED_REPLACEMENT_FRAME_SLIDER, FORCED_REPLACEMENT_SOURCE_DROPDOWN, FORCED_REPLACEMENT_BBOX_INPUT],
			outputs=FORCED_REPLACEMENT_FRAME_IMAGE,
			show_progress='hidden'
		)
	
	# Also update preview when frame slider changes (if source is selected)
	if FORCED_REPLACEMENT_FRAME_SLIDER:
		FORCED_REPLACEMENT_FRAME_SLIDER.change(
			update_preview_with_source_face_if_selected,
			inputs=[FORCED_REPLACEMENT_FRAME_SLIDER, FORCED_REPLACEMENT_SOURCE_DROPDOWN, FORCED_REPLACEMENT_BBOX_INPUT],
			outputs=FORCED_REPLACEMENT_FRAME_IMAGE,
			show_progress='hidden'
		)
	
	# Also update preview when bbox changes (if source is selected)
	if FORCED_REPLACEMENT_BBOX_INPUT:
		FORCED_REPLACEMENT_BBOX_INPUT.change(
			update_preview_with_source_face_if_selected,
			inputs=[FORCED_REPLACEMENT_FRAME_SLIDER, FORCED_REPLACEMENT_SOURCE_DROPDOWN, FORCED_REPLACEMENT_BBOX_INPUT],
			outputs=FORCED_REPLACEMENT_FRAME_IMAGE,
			show_progress='hidden'
		)
	
	# Load forced replacements when target video changes
	target_video = get_ui_component('target_video')
	if target_video:
		target_video.change(
			load_replacements_for_video,
			outputs=[FORCED_REPLACEMENT_LIST, FORCED_REPLACEMENT_FRAME_SLIDER, FORCED_REPLACEMENT_WRAPPER]
		)


def update_frame_preview(frame_number: int) -> gradio.Image:
	"""Update frame preview when frame slider changes"""
	# Store current frame number for coordinate updates
	state_manager.set_item('_forced_replacement_current_frame', frame_number)
	
	target_path = state_manager.get_item('target_path')
	if not target_path or not is_video(target_path):
		return gradio.Image(value=None)
	
	# Try to read from extracted frames first
	temp_frame_paths = resolve_temp_frame_paths(target_path)
	if temp_frame_paths and frame_number < len(temp_frame_paths):
		frame = read_static_image(temp_frame_paths[frame_number])
	else:
		frame = read_video_frame(target_path, frame_number)
	
	if frame is not None:
		# Convert BGR to RGB for display
		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		
		# Draw instruction text if click mode is active
		global _click_mode_active
		if _click_mode_active:
			cv2.putText(frame_rgb, 'Click Mode Active - Click twice: top-left, then bottom-right', 
			           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
		
		return gradio.Image(value=frame_rgb)
	
	return gradio.Image(value=None)


def update_preview_with_bbox(frame_number: int, bbox_str: str) -> gradio.Image:
	"""Update frame preview with bounding box drawn on it"""
	target_path = state_manager.get_item('target_path')
	if not target_path or not is_video(target_path):
		return gradio.Image(value=None)
	
	# Read frame
	temp_frame_paths = resolve_temp_frame_paths(target_path)
	if temp_frame_paths and frame_number < len(temp_frame_paths):
		frame = read_static_image(temp_frame_paths[frame_number])
	else:
		frame = read_video_frame(target_path, frame_number)
	
	if frame is None:
		return gradio.Image(value=None)
	
	# Convert BGR to RGB
	frame_rgb = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
	
	# Draw bounding box if coordinates provided
	if bbox_str and bbox_str.strip():
		try:
			bbox_coords = [float(x.strip()) for x in bbox_str.split(',')]
			if len(bbox_coords) == 4:
				x1, y1, x2, y2 = map(int, bbox_coords)
				# Ensure coordinates are within frame bounds
				height, width = frame_rgb.shape[:2]
				x1 = max(0, min(x1, width))
				y1 = max(0, min(y1, height))
				x2 = max(x1 + 1, min(x2, width))
				y2 = max(y1 + 1, min(y2, height))
				
				# Draw bounding box in cyan
				cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Cyan
				cv2.putText(frame_rgb, 'Selected Region', (x1, max(10, y1 - 5)), 
				           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
		except Exception as e:
			logger.debug(f'Could not draw bbox: {e}', __name__)
	
	return gradio.Image(value=frame_rgb)


def toggle_click_mode() -> Tuple[gradio.Button, gradio.Image]:
	"""Toggle click mode on/off"""
	global _click_mode_active, _bbox_start_point
	_click_mode_active = not _click_mode_active
	_bbox_start_point = None
	
	frame_number = state_manager.get_item('_forced_replacement_current_frame') or 0
	updated_image = update_frame_preview(frame_number)
	
	if _click_mode_active:
		button_text = 'Click Mode: Active (Click image to set corners)'
		button_variant = 'primary'
	else:
		button_text = 'Click Mode: Set Coordinates'
		button_variant = 'secondary'
	
	return gradio.Button(value=button_text, variant=button_variant), updated_image


def handle_bbox_selection(select_data: gradio.SelectData, frame_number: int) -> Tuple[gradio.Number, gradio.Number, gradio.Number, gradio.Number, gradio.Image]:
	"""
	Handle bounding box selection from image clicking.
	Uses two-click method: first click sets top-left, second click sets bottom-right.
	
	Args:
		select_data: Gradio SelectData containing click coordinates
		frame_number: Current frame number
	
	Returns:
		Updated coordinate inputs and image
	"""
	global _bbox_start_point, _click_mode_active
	
	if not _click_mode_active:
		# Click mode not active, ignore clicks
		current_image = update_frame_preview(frame_number)
		return gradio.Number(value=None), gradio.Number(value=None), gradio.Number(value=None), gradio.Number(value=None), current_image
	
	if select_data is None:
		current_image = update_frame_preview(frame_number)
		return gradio.Number(value=None), gradio.Number(value=None), gradio.Number(value=None), gradio.Number(value=None), current_image
	
	try:
		# Get click coordinates from select_data
		# Gradio's select event provides index which contains (x, y) coordinates
		if hasattr(select_data, 'index'):
			# The index is typically a tuple (x, y) for click coordinates
			if isinstance(select_data.index, (tuple, list)) and len(select_data.index) >= 2:
				x, y = int(select_data.index[0]), int(select_data.index[1])
			else:
				# Try to get from value or other attributes
				x, y = None, None
				if hasattr(select_data, 'value'):
					if isinstance(select_data.value, (tuple, list)) and len(select_data.value) >= 2:
						x, y = int(select_data.value[0]), int(select_data.value[1])
				
				if x is None or y is None:
					logger.warn(f'Could not extract click coordinates from select_data: {select_data}', __name__)
					current_image = update_frame_preview(frame_number)
					return gradio.Number(value=None), gradio.Number(value=None), gradio.Number(value=None), gradio.Number(value=None), current_image
		else:
			# Try alternative methods to get coordinates
			if hasattr(select_data, 'x') and hasattr(select_data, 'y'):
				x, y = int(select_data.x), int(select_data.y)
			else:
				logger.warn(f'select_data does not have expected attributes: {dir(select_data)}', __name__)
				current_image = update_frame_preview(frame_number)
				return gradio.Number(value=None), gradio.Number(value=None), gradio.Number(value=None), gradio.Number(value=None), current_image
		
		# Two-click method: first click sets start, second click completes bbox
		if _bbox_start_point is None:
			# First click - set start point (top-left)
			_bbox_start_point = (x, y)
			# Update coordinate inputs with first point
			# Draw a marker at the click point
			target_path = state_manager.get_item('target_path')
			if target_path and is_video(target_path):
				# Read frame again to draw on it
				temp_frame_paths = resolve_temp_frame_paths(target_path)
				if temp_frame_paths and frame_number < len(temp_frame_paths):
					frame = read_static_image(temp_frame_paths[frame_number])
				else:
					frame = read_video_frame(target_path, frame_number)
				
				if frame is not None:
					frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
					cv2.circle(frame_rgb, (x, y), 8, (255, 0, 0), -1)  # Red dot
					cv2.putText(frame_rgb, 'Top-left set. Click for bottom-right', (x + 10, y - 10),
					           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
					current_image = gradio.Image(value=frame_rgb)
				else:
					current_image = update_frame_preview(frame_number)
			else:
				current_image = update_frame_preview(frame_number)
			return gradio.Number(value=x), gradio.Number(value=y), gradio.Number(value=None), gradio.Number(value=None), current_image
		else:
			# Second click - complete bounding box (bottom-right)
			x1, y1 = _bbox_start_point
			x2, y2 = x, y
			
			# Ensure correct order
			x1, x2 = min(x1, x2), max(x1, x2)
			y1, y2 = min(y1, y2), max(y1, y2)
			
			_bbox_start_point = None  # Reset for next selection
			
			logger.debug(f'Created bbox from two clicks: ({x1}, {y1}, {x2}, {y2})', __name__)
			bbox_str = f'{x1}, {y1}, {x2}, {y2}'
			updated_image = update_preview_with_bbox(frame_number, bbox_str)
			return gradio.Number(value=x1), gradio.Number(value=y1), gradio.Number(value=x2), gradio.Number(value=y2), updated_image
		
	except Exception as e:
		logger.error(f'Error handling bbox selection: {e}', __name__)
		import traceback
		logger.error(traceback.format_exc(), __name__)
		_bbox_start_point = None  # Reset on error
		current_image = update_frame_preview(frame_number)
		return gradio.Textbox(value=''), current_image


def detect_face_in_area(frame_number: int, bbox_str: str) -> Tuple[gradio.Image, gradio.Textbox]:
	"""Detect face in the specified bounding box area"""
	target_path = state_manager.get_item('target_path')
	if not target_path or not is_video(target_path):
		return gradio.Image(value=None), gradio.Textbox(value=bbox_str)
	
	# Parse bounding box - try to get from coordinate inputs if bbox_str is empty
	if not bbox_str or not bbox_str.strip():
		x1 = FORCED_REPLACEMENT_BBOX_X1.value if FORCED_REPLACEMENT_BBOX_X1 else None
		y1 = FORCED_REPLACEMENT_BBOX_Y1.value if FORCED_REPLACEMENT_BBOX_Y1 else None
		x2 = FORCED_REPLACEMENT_BBOX_X2.value if FORCED_REPLACEMENT_BBOX_X2 else None
		y2 = FORCED_REPLACEMENT_BBOX_Y2.value if FORCED_REPLACEMENT_BBOX_Y2 else None
		
		if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
			bbox_str = f'{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}'
		else:
			logger.warn('No bounding box coordinates provided', __name__)
			return gradio.Image(value=None), gradio.Textbox(value='')
	
	# Parse bounding box
	try:
		bbox_coords = [float(x.strip()) for x in bbox_str.split(',')]
		if len(bbox_coords) != 4:
			raise ValueError('Bounding box must have 4 coordinates')
		bbox = tuple(bbox_coords)
	except Exception as e:
		logger.error(f'Invalid bounding box format: {e}', __name__)
		return gradio.Image(value=None), gradio.Textbox(value=bbox_str)
	
	# Read frame
	temp_frame_paths = resolve_temp_frame_paths(target_path)
	if temp_frame_paths and frame_number < len(temp_frame_paths):
		frame = read_static_image(temp_frame_paths[frame_number])
	else:
		frame = read_video_frame(target_path, frame_number)
	
	if frame is None:
		return gradio.Image(value=None), gradio.Textbox(value=bbox_str)
	
	# Detect face in region
	detected_face = detect_face_in_region(frame, bbox, detector_score=0.1)
	
	# Draw bounding box on frame
	frame_rgb = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
	
	# Draw original search region in yellow
	x1, y1, x2, y2 = map(int, bbox)
	cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (255, 255, 0), 2)  # Yellow
	# Add label
	cv2.putText(frame_rgb, 'Search Region', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
	
	# Draw detected face in green if found
	if detected_face:
		det_bbox = detected_face.bounding_box
		dx1, dy1, dx2, dy2 = map(int, det_bbox)
		cv2.rectangle(frame_rgb, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)  # Green
		cv2.putText(frame_rgb, 'Detected Face', (dx1, dy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
		# Update bbox input with detected coordinates
		bbox_str = f'{dx1}, {dy1}, {dx2}, {dy2}'
	
	return gradio.Image(value=frame_rgb), gradio.Textbox(value=bbox_str)


def save_forced_replacement(frame_number: int, bbox_str: str, source_index: Optional[int]) -> Tuple[gradio.Dataframe, gradio.Textbox, gradio.Dropdown]:
	"""Save a forced replacement"""
	if source_index is None:
		logger.warn('No source face selected', __name__)
		return update_replacement_list(), gradio.Textbox(value=bbox_str), gradio.Dropdown()
	
	# Parse bounding box - try to get from coordinate inputs if bbox_str is empty
	if not bbox_str or not bbox_str.strip():
		x1 = FORCED_REPLACEMENT_BBOX_X1.value if FORCED_REPLACEMENT_BBOX_X1 else None
		y1 = FORCED_REPLACEMENT_BBOX_Y1.value if FORCED_REPLACEMENT_BBOX_Y1 else None
		x2 = FORCED_REPLACEMENT_BBOX_X2.value if FORCED_REPLACEMENT_BBOX_X2 else None
		y2 = FORCED_REPLACEMENT_BBOX_Y2.value if FORCED_REPLACEMENT_BBOX_Y2 else None
		
		if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
			bbox_str = f'{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}'
		else:
			logger.warn('No bounding box coordinates provided', __name__)
			return update_replacement_list(), gradio.Textbox(value=''), gradio.Dropdown()
	
	# Parse bounding box
	try:
		bbox_coords = [float(x.strip()) for x in bbox_str.split(',')]
		if len(bbox_coords) != 4:
			raise ValueError('Bounding box must have 4 coordinates')
		bbox = tuple(bbox_coords)
	except Exception as e:
		logger.error(f'Invalid bounding box format: {e}', __name__)
		return update_replacement_list(), gradio.Textbox(value=bbox_str), gradio.Dropdown()
	
	# Create forced replacement
	from facefusion.types import ForcedFaceReplacement
	replacement: ForcedFaceReplacement = {
		'frame_number': frame_number,
		'bounding_box': bbox,
		'source_face_index': int(source_index),
		'detector_score': 0.1,
		'detector_model': None
	}
	
	# Save to state
	add_forced_replacement(replacement)
	
	# Save to file
	target_path = state_manager.get_item('target_path')
	if target_path:
		replacements = state_manager.get_item('forced_face_replacements') or {}
		save_forced_replacements(target_path, replacements)
	
	logger.info(f'Saved forced replacement for frame {frame_number}', __name__)
	
	# Clear inputs
	return update_replacement_list(), gradio.Textbox(value=''), gradio.Dropdown(value=None)


def clear_all_replacements() -> gradio.Dataframe:
	"""Clear all forced replacements"""
	clear_forced_replacements()
	target_path = state_manager.get_item('target_path')
	if target_path:
		save_forced_replacements(target_path, {})
	return update_replacement_list()


def reset_bbox_selection(frame_number: int) -> Tuple[gradio.Textbox, gradio.Image, gradio.Number, gradio.Number, gradio.Number, gradio.Number]:
	"""Reset bounding box selection (clear the two-click state and coordinate inputs)"""
	global _bbox_start_point
	_bbox_start_point = None
	current_image = update_frame_preview(frame_number)
	return gradio.Textbox(value=''), current_image, gradio.Number(value=None), gradio.Number(value=None), gradio.Number(value=None), gradio.Number(value=None)


def update_replacement_list() -> gradio.Dataframe:
	"""Update the replacement list display"""
	replacements = state_manager.get_item('forced_face_replacements') or {}
	
	if not replacements:
		return gradio.Dataframe(value=None)
	
	rows = []
	for frame_num, frame_replacements in sorted(replacements.items()):
		for repl in frame_replacements:
			bbox = repl['bounding_box']
			rows.append([
				frame_num,
				f'{int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}',
				repl['source_face_index'],
				'Delete'  # Future: add delete button
			])
	
	return gradio.Dataframe(value=rows if rows else None)


def get_source_dropdown_choices() -> List[str]:
	"""Get choices for source face dropdown"""
	source_paths = filter_image_paths(state_manager.get_item('source_paths') or [])
	logger.debug(f'Getting source dropdown choices from {len(source_paths)} source paths: {source_paths}', __name__)
	
	if not source_paths:
		return []
	
	choices = []
	for i, source_path in enumerate(source_paths):
		# Handle both string paths and File objects
		if isinstance(source_path, str):
			path_str = source_path
		elif hasattr(source_path, 'name'):
			path_str = source_path.name
		elif hasattr(source_path, 'path'):
			path_str = source_path.path
		else:
			path_str = str(source_path)
		
		# Get filename for display
		filename = path_str.split('/')[-1] if '/' in path_str else path_str.split('\\')[-1]
		choices.append(f'Source {i}: {filename}')
	
	logger.debug(f'Generated {len(choices)} dropdown choices: {choices}', __name__)
	return choices


def update_source_dropdown() -> gradio.Dropdown:
	"""Update source face dropdown when sources change"""
	choices = get_source_dropdown_choices()
	logger.debug(f'Updating source dropdown with {len(choices)} choices: {choices}', __name__)
	return gradio.Dropdown(choices=choices, value=choices[0] if choices else None)




def load_replacements_for_video() -> Tuple[gradio.Dataframe, gradio.Slider, gradio.Column]:
	"""Load forced replacements when video changes"""
	target_path = state_manager.get_item('target_path')
	if not target_path or not is_video(target_path):
		return gradio.Dataframe(value=None), gradio.Slider(), gradio.Column(visible=False)
	
	# Load from file
	replacements = load_forced_replacements(target_path)
	state_manager.set_item('forced_face_replacements', replacements)
	
	# Update frame slider max
	from facefusion.vision import count_video_frame_total
	max_frames = max(0, count_video_frame_total(target_path) - 1)
	
	return update_replacement_list(), gradio.Slider(maximum=max_frames), gradio.Column(visible=True)


def update_preview_with_source_face_if_selected(frame_number: int, source_index: Optional[str], bbox_str: str) -> gradio.Image:
	"""Update preview with source face if a source is selected, otherwise just show frame"""
	if source_index is None or source_index == '':
		# No source selected, just show frame
		return update_frame_preview(frame_number)
	return update_preview_with_source_face(frame_number, source_index, bbox_str)


def update_preview_with_source_face(frame_number: int, source_index: Optional[str], bbox_str: str) -> gradio.Image:
	"""Update frame preview with source face swapped in"""
	if source_index is None or source_index == '':
		# No source selected, just show frame
		return update_frame_preview(frame_number)
	
	# Parse source index from dropdown choice (format: "Source 0: filename.jpg")
	try:
		if isinstance(source_index, str):
			# Extract index from string like "Source 0: filename.jpg"
			index_str = source_index.split(':')[0].replace('Source', '').strip()
			source_index_int = int(index_str)
		else:
			source_index_int = int(source_index)
	except (ValueError, AttributeError, TypeError) as e:
		logger.warn(f'Could not parse source index from: {source_index} ({type(source_index)}): {e}', __name__)
		return update_frame_preview(frame_number)
	
	source_index = source_index_int
	
	# Get source face
	source_paths = filter_image_paths(state_manager.get_item('source_paths') or [])
	if source_index >= len(source_paths):
		logger.warn(f'Source index {source_index} out of range (max: {len(source_paths)-1})', __name__)
		return update_frame_preview(frame_number)
	
	source_path = source_paths[source_index]
	source_frame = read_static_image(source_path)
	if source_frame is None:
		logger.warn(f'Could not read source image: {source_path}', __name__)
		return update_frame_preview(frame_number)
	
	# Extract source face
	source_faces = get_many_faces([source_frame])
	source_faces = sort_faces_by_order(source_faces, 'large-small')
	if not source_faces:
		logger.warn(f'No faces detected in source image: {source_path}', __name__)
		return update_frame_preview(frame_number)
	
	source_face = get_first(source_faces)
	
	# Get target frame
	target_path = state_manager.get_item('target_path')
	if not target_path or not is_video(target_path):
		return update_frame_preview(frame_number)
	
	temp_frame_paths = resolve_temp_frame_paths(target_path)
	if temp_frame_paths and frame_number < len(temp_frame_paths):
		target_frame = read_static_image(temp_frame_paths[frame_number])
	else:
		target_frame = read_video_frame(target_path, frame_number)
	
	if target_frame is None:
		return update_frame_preview(frame_number)
	
	# If bbox is provided, detect face in that region, otherwise use all faces
	if bbox_str and bbox_str.strip():
		try:
			bbox_coords = [float(x.strip()) for x in bbox_str.split(',')]
			if len(bbox_coords) == 4:
				bbox = tuple(bbox_coords)
				target_face = detect_face_in_region(target_frame, bbox, detector_score=0.1)
				if not target_face:
					logger.warn('Could not detect face in specified bounding box', __name__)
					return update_frame_preview(frame_number)
			else:
				# Invalid bbox, try to get all faces
				target_faces = get_many_faces([target_frame])
				if not target_faces:
					return update_frame_preview(frame_number)
				target_face = get_first(sort_faces_by_order(target_faces, 'large-small'))
		except Exception as e:
			logger.warn(f'Error parsing bbox: {e}, using all faces', __name__)
			target_faces = get_many_faces([target_frame])
			if not target_faces:
				return update_frame_preview(frame_number)
			target_face = get_first(sort_faces_by_order(target_faces, 'large-small'))
	else:
		# No bbox, get all faces
		target_faces = get_many_faces([target_frame])
		if not target_faces:
			return update_frame_preview(frame_number)
		target_face = get_first(sort_faces_by_order(target_faces, 'large-small'))
	
	# Swap face using face_swapper
	try:
		from facefusion.processors.modules.face_swapper.core import swap_face
		swapped_frame = swap_face(source_face, target_face, target_frame.copy())
		
		# Convert BGR to RGB for display
		frame_rgb = cv2.cvtColor(swapped_frame, cv2.COLOR_BGR2RGB)
		
		# Draw bounding box if available
		if bbox_str and bbox_str.strip():
			try:
				bbox_coords = [float(x.strip()) for x in bbox_str.split(',')]
				if len(bbox_coords) == 4:
					x1, y1, x2, y2 = map(int, bbox_coords)
					cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
			except:
				pass
		
		return gradio.Image(value=frame_rgb)
	except Exception as e:
		logger.error(f'Error swapping face in preview: {e}', __name__)
		import traceback
		logger.error(traceback.format_exc(), __name__)
		return update_frame_preview(frame_number)

