from time import sleep
from typing import Optional, Tuple

import gradio

from facefusion import logger, process_manager, state_manager, translator
from facefusion.args import collect_step_args
from facefusion.core import process_step
from facefusion.filesystem import is_directory, is_image, is_video
from facefusion.jobs import job_helper, job_manager, job_runner, job_store
from facefusion.temp_helper import clear_temp_directory
from facefusion.types import Args, UiWorkflow
from facefusion.uis.core import get_ui_component
from facefusion.uis.ui_helper import suggest_output_path

INSTANT_RUNNER_WRAPPER : Optional[gradio.Row] = None
INSTANT_RUNNER_START_BUTTON : Optional[gradio.Button] = None
INSTANT_RUNNER_PREPROCESS_BUTTON : Optional[gradio.Button] = None
INSTANT_RUNNER_CONTINUE_BUTTON : Optional[gradio.Button] = None
INSTANT_RUNNER_STOP_BUTTON : Optional[gradio.Button] = None
INSTANT_RUNNER_CLEAR_BUTTON : Optional[gradio.Button] = None


def render() -> None:
	global INSTANT_RUNNER_WRAPPER
	global INSTANT_RUNNER_START_BUTTON
	global INSTANT_RUNNER_PREPROCESS_BUTTON
	global INSTANT_RUNNER_CONTINUE_BUTTON
	global INSTANT_RUNNER_STOP_BUTTON
	global INSTANT_RUNNER_CLEAR_BUTTON

	if job_manager.init_jobs(state_manager.get_item('jobs_path')):
		is_instant_runner = state_manager.get_item('ui_workflow') == 'instant_runner'

		with gradio.Row(visible = is_instant_runner) as INSTANT_RUNNER_WRAPPER:
			INSTANT_RUNNER_START_BUTTON = gradio.Button(
				value = translator.get('uis.start_button') or 'Start',
				variant = 'primary',
				size = 'sm'
			)
			INSTANT_RUNNER_PREPROCESS_BUTTON = gradio.Button(
				value = translator.get('uis.preprocess_button') or 'Preprocess',
				variant = 'secondary',
				size = 'sm',
				visible = is_video(state_manager.get_item('target_path'))
			)
			INSTANT_RUNNER_CONTINUE_BUTTON = gradio.Button(
				value = translator.get('uis.continue_button') or 'Continue',
				variant = 'secondary',
				size = 'sm',
				visible = is_video(state_manager.get_item('target_path'))
			)
			INSTANT_RUNNER_STOP_BUTTON = gradio.Button(
				value = translator.get('uis.stop_button'),
				variant = 'primary',
				size = 'sm',
				visible = False
			)
			INSTANT_RUNNER_CLEAR_BUTTON = gradio.Button(
				value = translator.get('uis.clear_button'),
				size = 'sm'
			)


def listen() -> None:
	output_image = get_ui_component('output_image')
	output_video = get_ui_component('output_video')
	ui_workflow_dropdown = get_ui_component('ui_workflow_dropdown')

	if output_image and output_video:
		INSTANT_RUNNER_START_BUTTON.click(start, outputs = [ INSTANT_RUNNER_START_BUTTON, INSTANT_RUNNER_STOP_BUTTON ])
		INSTANT_RUNNER_START_BUTTON.click(run, outputs = [ INSTANT_RUNNER_START_BUTTON, INSTANT_RUNNER_STOP_BUTTON, output_image, output_video ])
		INSTANT_RUNNER_PREPROCESS_BUTTON.click(start, outputs = [ INSTANT_RUNNER_PREPROCESS_BUTTON, INSTANT_RUNNER_STOP_BUTTON ])
		INSTANT_RUNNER_PREPROCESS_BUTTON.click(run_preprocess, outputs = [ INSTANT_RUNNER_PREPROCESS_BUTTON, INSTANT_RUNNER_STOP_BUTTON, output_image, output_video ])
		INSTANT_RUNNER_CONTINUE_BUTTON.click(start, outputs = [ INSTANT_RUNNER_CONTINUE_BUTTON, INSTANT_RUNNER_STOP_BUTTON ])
		INSTANT_RUNNER_CONTINUE_BUTTON.click(run_continue, outputs = [ INSTANT_RUNNER_CONTINUE_BUTTON, INSTANT_RUNNER_STOP_BUTTON, output_image, output_video ])
		INSTANT_RUNNER_STOP_BUTTON.click(stop, outputs = [ INSTANT_RUNNER_START_BUTTON, INSTANT_RUNNER_STOP_BUTTON, output_image, output_video ])
		INSTANT_RUNNER_CLEAR_BUTTON.click(clear, outputs = [ output_image, output_video ])
		
		# Update button visibility when target changes
		target_video = get_ui_component('target_video')
		if target_video:
			for method in ['change', 'clear']:
				getattr(target_video, method)(update_button_visibility, outputs = [INSTANT_RUNNER_PREPROCESS_BUTTON, INSTANT_RUNNER_CONTINUE_BUTTON])
	if ui_workflow_dropdown:
		ui_workflow_dropdown.change(remote_update, inputs = ui_workflow_dropdown, outputs = INSTANT_RUNNER_WRAPPER)


def remote_update(ui_workflow : UiWorkflow) -> gradio.Row:
	is_instant_runner = ui_workflow == 'instant_runner'

	return gradio.Row(visible = is_instant_runner)


def start() -> Tuple[gradio.Button, gradio.Button]:
	while not process_manager.is_processing():
		sleep(0.5)
	return gradio.Button(visible = False), gradio.Button(visible = True)


def run() -> Tuple[gradio.Button, gradio.Button, gradio.Image, gradio.Video]:
	step_args = collect_step_args()
	output_path = step_args.get('output_path')

	if is_directory(step_args.get('output_path')):
		step_args['output_path'] = suggest_output_path(step_args.get('output_path'), state_manager.get_item('target_path'))
	if job_manager.init_jobs(state_manager.get_item('jobs_path')):
		create_and_run_job(step_args)
		state_manager.set_item('output_path', output_path)
	if is_image(step_args.get('output_path')):
		return gradio.Button(visible = True), gradio.Button(visible = False), gradio.Image(value = step_args.get('output_path'), visible = True), gradio.Video(value = None, visible = False)
	if is_video(step_args.get('output_path')):
		return gradio.Button(visible = True), gradio.Button(visible = False), gradio.Image(value = None, visible = False), gradio.Video(value = step_args.get('output_path'), visible = True)
	return gradio.Button(visible = True), gradio.Button(visible = False), gradio.Image(value = None), gradio.Video(value = None)


def create_and_run_job(step_args : Args) -> bool:
	job_id = job_helper.suggest_job_id('ui')

	for key in job_store.get_job_keys():
		state_manager.sync_item(key) #type:ignore[arg-type]

	return job_manager.create_job(job_id) and job_manager.add_step(job_id, step_args) and job_manager.submit_job(job_id) and job_runner.run_job(job_id, process_step)


def stop() -> Tuple[gradio.Button, gradio.Button, gradio.Image, gradio.Video]:
	process_manager.stop()
	return gradio.Button(visible = True), gradio.Button(visible = False), gradio.Image(value = None), gradio.Video(value = None)


def run_preprocess() -> Tuple[gradio.Button, gradio.Button, gradio.Image, gradio.Video]:
	"""Run preprocessing only: setup -> extract -> preprocess"""
	step_args = collect_step_args()
	output_path = step_args.get('output_path')
	
	if is_directory(step_args.get('output_path')):
		step_args['output_path'] = suggest_output_path(step_args.get('output_path'), state_manager.get_item('target_path'))
	
	# Set workflow mode to preprocess_only in step_args (so it persists during job execution)
	step_args['workflow_mode'] = 'preprocess_only'
	# Explicitly enable face preprocessing for this workflow
	step_args['enable_face_preprocessing'] = True
	
	if job_manager.init_jobs(state_manager.get_item('jobs_path')):
		create_and_run_job(step_args)
		state_manager.set_item('output_path', output_path)
	
	return gradio.Button(visible = True), gradio.Button(visible = False), gradio.Image(value = None), gradio.Video(value = None)


def run_continue() -> Tuple[gradio.Button, gradio.Button, gradio.Image, gradio.Video]:
	"""Continue processing: process -> merge -> audio -> finalize (skips preprocessing if already done)"""
	step_args = collect_step_args()
	output_path = step_args.get('output_path')

	if is_directory(step_args.get('output_path')):
		step_args['output_path'] = suggest_output_path(step_args.get('output_path'), state_manager.get_item('target_path'))
	
	# Set workflow mode to continue in step_args (so it persists during job execution)
	step_args['workflow_mode'] = 'continue'
	
	# Pass cluster_source_mapping through step_args to ensure it's available during processing
	cluster_source_mapping = state_manager.get_item('cluster_source_mapping')
	if cluster_source_mapping:
		step_args['cluster_source_mapping'] = cluster_source_mapping
		logger.info(f'[instant_runner] Passing cluster_source_mapping to job: {cluster_source_mapping}', __name__)
	
	if job_manager.init_jobs(state_manager.get_item('jobs_path')):
		create_and_run_job(step_args)
		state_manager.set_item('output_path', output_path)
	
	if is_image(step_args.get('output_path')):
		return gradio.Button(visible = True), gradio.Button(visible = False), gradio.Image(value = step_args.get('output_path'), visible = True), gradio.Video(value = None, visible = False)
	if is_video(step_args.get('output_path')):
		return gradio.Button(visible = True), gradio.Button(visible = False), gradio.Image(value = None, visible = False), gradio.Video(value = step_args.get('output_path'), visible = True)
	return gradio.Button(visible = True), gradio.Button(visible = False), gradio.Image(value = None), gradio.Video(value = None)


def update_button_visibility() -> Tuple[gradio.Button, gradio.Button]:
	"""Update visibility of preprocess/continue buttons based on target type"""
	is_video_target = is_video(state_manager.get_item('target_path'))
	return gradio.Button(visible = is_video_target), gradio.Button(visible = is_video_target)


def clear() -> Tuple[gradio.Image, gradio.Video]:
	while process_manager.is_processing():
		sleep(0.5)
	if state_manager.get_item('target_path'):
		clear_temp_directory(state_manager.get_item('target_path'))
		# Clear workflow mode
		state_manager.set_item('workflow_mode', None)
	return gradio.Image(value = None), gradio.Video(value = None)
