from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import numpy
from tqdm import tqdm

from facefusion import ffmpeg
from facefusion import logger, process_manager, state_manager, translator, video_manager
from facefusion.audio import create_empty_audio_frame, get_audio_frame, get_voice_frame
from facefusion.common_helper import get_first
from facefusion.content_analyser import analyse_video
from facefusion.filesystem import filter_audio_paths, is_video
from facefusion.processors.core import get_processors_modules
from facefusion.temp_helper import clear_temp_directory, create_temp_directory, move_temp_file, resolve_temp_frame_paths
from facefusion.time_helper import calculate_end_time
from facefusion.types import ErrorCode
from facefusion.vision import conditional_merge_vision_mask, detect_video_resolution, extract_vision_mask, pack_resolution, read_static_image, read_static_images, read_static_video_frame, restrict_trim_frame, restrict_video_fps, restrict_video_resolution, scale_resolution, write_image
from facefusion.workflows.core import is_process_stopping


def process(start_time : float) -> ErrorCode:
	"""Full workflow: setup -> extract -> preprocess -> process -> merge -> audio -> finalize"""
	tasks =\
	[
		setup,
		extract_frames,
		preprocess_faces,  # Preprocessing after frames are extracted (faster)
		process_video,
		merge_frames,
		restore_audio,
		partial(finalize_video, start_time)
	]
	process_manager.start()

	for task in tasks:
		error_code = task() # type:ignore[operator]

		if error_code > 0:
			process_manager.end()
			return error_code

	process_manager.end()
	return 0


def process_preprocess_only(start_time : float) -> ErrorCode:
	"""Preprocess-only workflow: setup -> extract -> preprocess (stops after building database)"""
	tasks =\
	[
		setup,
		extract_frames,
		preprocess_faces
	]
	process_manager.start()

	for task in tasks:
		error_code = task() # type:ignore[operator]

		if error_code > 0:
			process_manager.end()
			return error_code

	process_manager.end()
	return 0


def process_continue(start_time : float) -> ErrorCode:
	"""Continue workflow: checks if preprocessing done, then processes -> merge -> audio -> finalize"""
	# Check if preprocessing is already done (database exists and frames extracted)
	from facefusion.video_face_database import get_video_face_database
	from facefusion.temp_helper import resolve_temp_frame_paths
	
	# If preprocessing enabled, check if it's already done
	enable_preprocessing = state_manager.get_item('enable_face_preprocessing')
	needs_preprocessing = False
	
	if enable_preprocessing and is_video(state_manager.get_item('target_path')):
		temp_frame_paths = resolve_temp_frame_paths(state_manager.get_item('target_path'))
		database = get_video_face_database()
		
		# Need preprocessing if frames exist but database doesn't
		if temp_frame_paths and database is None:
			needs_preprocessing = True
	
	tasks = []
	
	# Only do setup/extract if frames don't exist
	temp_frame_paths = resolve_temp_frame_paths(state_manager.get_item('target_path'))
	if not temp_frame_paths:
		tasks.extend([setup, extract_frames])
	
	# Do preprocessing if needed
	if needs_preprocessing:
		tasks.append(preprocess_faces)
	
	# Continue with processing
	tasks.extend([
		process_video,
		merge_frames,
		restore_audio,
		partial(finalize_video, start_time)
	])
	
	process_manager.start()

	for task in tasks:
		error_code = task() # type:ignore[operator]

		if error_code > 0:
			process_manager.end()
			return error_code

	process_manager.end()
	return 0


def setup() -> ErrorCode:
	trim_frame_start, trim_frame_end = restrict_trim_frame(state_manager.get_item('target_path'), state_manager.get_item('trim_frame_start'), state_manager.get_item('trim_frame_end'))

	if analyse_video(state_manager.get_item('target_path'), trim_frame_start, trim_frame_end):
		return 3

	logger.debug(translator.get('clearing_temp'), __name__)
	clear_temp_directory(state_manager.get_item('target_path'))
	logger.debug(translator.get('creating_temp'), __name__)
	create_temp_directory(state_manager.get_item('target_path'))
	return 0


def preprocess_faces() -> ErrorCode:
	"""
	Preprocess video faces: scan all extracted frames and build face database with clustering.
	This runs AFTER frames are extracted to disk for better performance.
	This is optional and can be enabled/disabled via state_manager.
	"""
	# Check if preprocessing is enabled (default: True for now, can be made configurable)
	enable_preprocessing = state_manager.get_item('enable_face_preprocessing')
	if enable_preprocessing is False:
		logger.info('Face preprocessing disabled, skipping', __name__)
		return 0
	
	# Only preprocess for videos
	if not is_video(state_manager.get_item('target_path')):
		logger.debug('Not a video, skipping face preprocessing', __name__)
		return 0
	
	# Get extracted frame paths
	temp_frame_paths = resolve_temp_frame_paths(state_manager.get_item('target_path'))
	if not temp_frame_paths:
		logger.warn('No extracted frames found, skipping face preprocessing', __name__)
		return 0
	
	clustering_threshold = state_manager.get_item('face_clustering_threshold')
	if clustering_threshold is None:
		clustering_threshold = 0.35  # Default threshold
	
	logger.info('Starting face preprocessing on extracted frames...', __name__)
	try:
		from facefusion.video_face_database import preprocess_extracted_frames
		preprocess_extracted_frames(
			temp_frame_paths,
			clustering_threshold
		)
		logger.info('Face preprocessing completed successfully', __name__)
		return 0
	except Exception as e:
		logger.error(f'Face preprocessing failed: {e}', __name__)
		import traceback
		logger.error(traceback.format_exc(), __name__)
		# Don't fail the whole process if preprocessing fails
		# Just log and continue
		return 0


def extract_frames() -> ErrorCode:
	trim_frame_start, trim_frame_end = restrict_trim_frame(state_manager.get_item('target_path'), state_manager.get_item('trim_frame_start'), state_manager.get_item('trim_frame_end'))
	output_video_resolution = scale_resolution(detect_video_resolution(state_manager.get_item('target_path')), state_manager.get_item('output_video_scale'))
	temp_video_resolution = restrict_video_resolution(state_manager.get_item('target_path'), output_video_resolution)
	temp_video_fps = restrict_video_fps(state_manager.get_item('target_path'), state_manager.get_item('output_video_fps'))
	logger.info(translator.get('extracting_frames').format(resolution=pack_resolution(temp_video_resolution), fps=temp_video_fps), __name__)

	if ffmpeg.extract_frames(state_manager.get_item('target_path'), temp_video_resolution, temp_video_fps, trim_frame_start, trim_frame_end):
		logger.debug(translator.get('extracting_frames_succeeded'), __name__)
	else:
		if is_process_stopping():
			return 4
		logger.error(translator.get('extracting_frames_failed'), __name__)
		return 1
	return 0


def process_video() -> ErrorCode:
	# Clear face tracker at start of video processing for fresh tracking
	from facefusion.face_tracker import clear_face_tracker
	clear_face_tracker()
	
	# Load forced replacements for this video
	from facefusion.forced_replacements import load_forced_replacements
	target_path = state_manager.get_item('target_path')
	if target_path and is_video(target_path):
		replacements = load_forced_replacements(target_path)
		state_manager.set_item('forced_face_replacements', replacements)
	temp_frame_paths = resolve_temp_frame_paths(state_manager.get_item('target_path'))

	if temp_frame_paths:
		with tqdm(total = len(temp_frame_paths), desc = translator.get('processing'), unit = 'frame', ascii = ' =', disable = state_manager.get_item('log_level') in [ 'warn', 'error' ]) as progress:
			progress.set_postfix(execution_providers = state_manager.get_item('execution_providers'))

			with ThreadPoolExecutor(max_workers = state_manager.get_item('execution_thread_count')) as executor:
				futures = []

				for frame_number, temp_frame_path in enumerate(temp_frame_paths):
					future = executor.submit(process_temp_frame, temp_frame_path, frame_number)
					futures.append(future)

				for future in as_completed(futures):
					if is_process_stopping():
						for __future__ in futures:
							__future__.cancel()

					if not future.cancelled():
						future.result()
						progress.update()

		for processor_module in get_processors_modules(state_manager.get_item('processors')):
			processor_module.post_process()

		if is_process_stopping():
			return 4
	else:
		logger.error(translator.get('temp_frames_not_found'), __name__)
		return 1
	return 0


def merge_frames() -> ErrorCode:
	trim_frame_start, trim_frame_end = restrict_trim_frame(state_manager.get_item('target_path'), state_manager.get_item('trim_frame_start'), state_manager.get_item('trim_frame_end'))
	output_video_resolution = scale_resolution(detect_video_resolution(state_manager.get_item('target_path')), state_manager.get_item('output_video_scale'))
	temp_video_fps = restrict_video_fps(state_manager.get_item('target_path'), state_manager.get_item('output_video_fps'))

	logger.info(translator.get('merging_video').format(resolution = pack_resolution(output_video_resolution), fps = state_manager.get_item('output_video_fps')), __name__)
	if ffmpeg.merge_video(state_manager.get_item('target_path'), temp_video_fps, output_video_resolution, state_manager.get_item('output_video_fps'), trim_frame_start, trim_frame_end):
		logger.debug(translator.get('merging_video_succeeded'), __name__)
	else:
		if is_process_stopping():
			return 4
		logger.error(translator.get('merging_video_failed'), __name__)
		return 1
	return 0


def restore_audio() -> ErrorCode:
	trim_frame_start, trim_frame_end = restrict_trim_frame(state_manager.get_item('target_path'), state_manager.get_item('trim_frame_start'), state_manager.get_item('trim_frame_end'))

	if state_manager.get_item('output_audio_volume') == 0:
		logger.info(translator.get('skipping_audio'), __name__)
		move_temp_file(state_manager.get_item('target_path'), state_manager.get_item('output_path'))
	else:
		source_audio_path = get_first(filter_audio_paths(state_manager.get_item('source_paths')))
		if source_audio_path:
			if ffmpeg.replace_audio(state_manager.get_item('target_path'), source_audio_path, state_manager.get_item('output_path')):
				video_manager.clear_video_pool()
				logger.debug(translator.get('replacing_audio_succeeded'), __name__)
			else:
				video_manager.clear_video_pool()
				if is_process_stopping():
					return 4
				logger.warn(translator.get('replacing_audio_skipped'), __name__)
				move_temp_file(state_manager.get_item('target_path'), state_manager.get_item('output_path'))
		else:
			if ffmpeg.restore_audio(state_manager.get_item('target_path'), state_manager.get_item('output_path'), trim_frame_start, trim_frame_end):
				video_manager.clear_video_pool()
				logger.debug(translator.get('restoring_audio_succeeded'), __name__)
			else:
				video_manager.clear_video_pool()
				if is_process_stopping():
					return 4
				logger.warn(translator.get('restoring_audio_skipped'), __name__)
				move_temp_file(state_manager.get_item('target_path'), state_manager.get_item('output_path'))
	return 0


def process_temp_frame(temp_frame_path : str, frame_number : int) -> bool:
	reference_vision_frame = read_static_video_frame(state_manager.get_item('target_path'), state_manager.get_item('reference_frame_number'))
	source_vision_frames = read_static_images(state_manager.get_item('source_paths'))
	source_audio_path = get_first(filter_audio_paths(state_manager.get_item('source_paths')))
	temp_video_fps = restrict_video_fps(state_manager.get_item('target_path'), state_manager.get_item('output_video_fps'))
	target_vision_frame = read_static_image(temp_frame_path, 'rgba')
	temp_vision_frame = target_vision_frame.copy()
	temp_vision_mask = extract_vision_mask(temp_vision_frame)

	source_audio_frame = get_audio_frame(source_audio_path, temp_video_fps, frame_number)
	source_voice_frame = get_voice_frame(source_audio_path, temp_video_fps, frame_number)

	if not numpy.any(source_audio_frame):
		source_audio_frame = create_empty_audio_frame()
	if not numpy.any(source_voice_frame):
		source_voice_frame = create_empty_audio_frame()

	processors = state_manager.get_item('processors')
	if frame_number == 0:
		logger.info(f'[image_to_video] Processing frame {frame_number} with processors: {processors}', __name__)
	
	for processor_module in get_processors_modules(processors):
		if frame_number == 0:
			logger.info(f'[image_to_video] Calling processor: {processor_module.__name__}', __name__)
		temp_vision_frame, temp_vision_mask = processor_module.process_frame(
		{
			'reference_vision_frame': reference_vision_frame,
			'source_vision_frames': source_vision_frames,
			'source_audio_frame': source_audio_frame,
			'source_voice_frame': source_voice_frame,
			'target_vision_frame': target_vision_frame[:, :, :3],
			'temp_vision_frame': temp_vision_frame[:, :, :3],
			'temp_vision_mask': temp_vision_mask,
			'frame_number': frame_number
		})

	temp_vision_frame = conditional_merge_vision_mask(temp_vision_frame, temp_vision_mask)
	return write_image(temp_frame_path, temp_vision_frame)


def finalize_video(start_time : float) -> ErrorCode:
	logger.debug(translator.get('clearing_temp'), __name__)
	clear_temp_directory(state_manager.get_item('target_path'))

	if is_video(state_manager.get_item('output_path')):
		logger.info(translator.get('processing_video_succeeded').format(seconds = calculate_end_time(start_time)), __name__)
	else:
		logger.error(translator.get('processing_video_failed'), __name__)
		return 1
	return 0
