from argparse import ArgumentParser
from functools import lru_cache
from typing import List, Optional, Tuple

import cv2
import numpy

import facefusion.choices
import facefusion.jobs.job_manager
import facefusion.jobs.job_store
from facefusion import config, content_analyser, face_classifier, face_detector, face_landmarker, face_masker, face_recognizer, inference_manager, logger, state_manager, translator, video_manager
from facefusion.common_helper import get_first, is_macos
from facefusion.download import conditional_download_hashes, conditional_download_sources, resolve_download_url
from facefusion.execution import has_execution_provider
from facefusion.face_analyser import get_average_face, get_many_faces, get_one_face, scale_face
from facefusion.face_helper import paste_back, warp_face_by_face_landmark_5
from facefusion.face_masker import create_area_mask, create_box_mask, create_occlusion_mask, create_region_mask
from facefusion.face_selector import select_faces, sort_faces_by_order
from facefusion.filesystem import filter_image_paths, has_image, in_directory, is_image, is_video, resolve_relative_path, same_file_extension
from facefusion.model_helper import get_static_model_initializer
from facefusion.processors.modules.face_swapper import choices as face_swapper_choices
from facefusion.processors.modules.face_swapper.types import FaceSwapperInputs
from facefusion.processors.pixel_boost import explode_pixel_boost, implode_pixel_boost
from facefusion.processors.types import ProcessorOutputs
from facefusion.program_helper import find_argument_group
from facefusion.thread_helper import conditional_thread_semaphore
from facefusion.types import ApplyStateItem, Args, DownloadScope, Embedding, Face, InferencePool, ModelOptions, ModelSet, ProcessMode, VisionFrame
from facefusion.vision import read_static_image, read_static_images, read_static_video_frame, unpack_resolution


@lru_cache()
def create_static_model_set(download_scope : DownloadScope) -> ModelSet:
	return\
	{
		'blendswap_256':
		{
			'__metadata__':
			{
				'vendor': 'mapooon',
				'license': 'Non-Commercial',
				'year': 2023
			},
			'hashes':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'blendswap_256.hash'),
					'path': resolve_relative_path('../.assets/models/blendswap_256.hash')
				}
			},
			'sources':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'blendswap_256.onnx'),
					'path': resolve_relative_path('../.assets/models/blendswap_256.onnx')
				}
			},
			'type': 'blendswap',
			'template': 'ffhq_512',
			'size': (256, 256),
			'mean': [ 0.0, 0.0, 0.0 ],
			'standard_deviation': [ 1.0, 1.0, 1.0 ]
		},
		'ghost_1_256':
		{
			'__metadata__':
			{
				'vendor': 'ai-forever',
				'license': 'Apache-2.0',
				'year': 2022
			},
			'hashes':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'ghost_1_256.hash'),
					'path': resolve_relative_path('../.assets/models/ghost_1_256.hash')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.4.0', 'crossface_ghost.hash'),
					'path': resolve_relative_path('../.assets/models/crossface_ghost.hash')
				}
			},
			'sources':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'ghost_1_256.onnx'),
					'path': resolve_relative_path('../.assets/models/ghost_1_256.onnx')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.4.0', 'crossface_ghost.onnx'),
					'path': resolve_relative_path('../.assets/models/crossface_ghost.onnx')
				}
			},
			'type': 'ghost',
			'template': 'arcface_112_v1',
			'size': (256, 256),
			'mean': [ 0.5, 0.5, 0.5 ],
			'standard_deviation': [ 0.5, 0.5, 0.5 ]
		},
		'ghost_2_256':
		{
			'__metadata__':
			{
				'vendor': 'ai-forever',
				'license': 'Apache-2.0',
				'year': 2022
			},
			'hashes':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'ghost_2_256.hash'),
					'path': resolve_relative_path('../.assets/models/ghost_2_256.hash')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.4.0', 'crossface_ghost.hash'),
					'path': resolve_relative_path('../.assets/models/crossface_ghost.hash')
				}
			},
			'sources':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'ghost_2_256.onnx'),
					'path': resolve_relative_path('../.assets/models/ghost_2_256.onnx')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.4.0', 'crossface_ghost.onnx'),
					'path': resolve_relative_path('../.assets/models/crossface_ghost.onnx')
				}
			},
			'type': 'ghost',
			'template': 'arcface_112_v1',
			'size': (256, 256),
			'mean': [ 0.5, 0.5, 0.5 ],
			'standard_deviation': [ 0.5, 0.5, 0.5 ]
		},
		'ghost_3_256':
		{
			'__metadata__':
			{
				'vendor': 'ai-forever',
				'license': 'Apache-2.0',
				'year': 2022
			},
			'hashes':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'ghost_3_256.hash'),
					'path': resolve_relative_path('../.assets/models/ghost_3_256.hash')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.4.0', 'crossface_ghost.hash'),
					'path': resolve_relative_path('../.assets/models/crossface_ghost.hash')
				}
			},
			'sources':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'ghost_3_256.onnx'),
					'path': resolve_relative_path('../.assets/models/ghost_3_256.onnx')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.4.0', 'crossface_ghost.onnx'),
					'path': resolve_relative_path('../.assets/models/crossface_ghost.onnx')
				}
			},
			'type': 'ghost',
			'template': 'arcface_112_v1',
			'size': (256, 256),
			'mean': [ 0.5, 0.5, 0.5 ],
			'standard_deviation': [ 0.5, 0.5, 0.5 ]
		},
		'hififace_unofficial_256':
		{
			'__metadata__':
			{
				'vendor': 'GuijiAI',
				'license': 'Unknown',
				'year': 2021
			},
			'hashes':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.1.0', 'hififace_unofficial_256.hash'),
					'path': resolve_relative_path('../.assets/models/hififace_unofficial_256.hash')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.4.0', 'crossface_hififace.hash'),
					'path': resolve_relative_path('../.assets/models/crossface_hififace.hash')
				}
			},
			'sources':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.1.0', 'hififace_unofficial_256.onnx'),
					'path': resolve_relative_path('../.assets/models/hififace_unofficial_256.onnx')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.4.0', 'crossface_hififace.onnx'),
					'path': resolve_relative_path('../.assets/models/crossface_hififace.onnx')
				}
			},
			'type': 'hififace',
			'template': 'mtcnn_512',
			'size': (256, 256),
			'mean': [ 0.5, 0.5, 0.5 ],
			'standard_deviation': [ 0.5, 0.5, 0.5 ]
		},
		'hyperswap_1a_256':
		{
			'__metadata__':
			{
				'vendor': 'FaceFusion',
				'license': 'ResearchRAIL',
				'year': 2025
			},
			'hashes':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.3.0', 'hyperswap_1a_256.hash'),
					'path': resolve_relative_path('../.assets/models/hyperswap_1a_256.hash')
				}
			},
			'sources':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.3.0', 'hyperswap_1a_256.onnx'),
					'path': resolve_relative_path('../.assets/models/hyperswap_1a_256.onnx')
				}
			},
			'type': 'hyperswap',
			'template': 'arcface_128',
			'size': (256, 256),
			'mean': [ 0.5, 0.5, 0.5 ],
			'standard_deviation': [ 0.5, 0.5, 0.5 ]
		},
		'hyperswap_1b_256':
		{
			'__metadata__':
			{
				'vendor': 'FaceFusion',
				'license': 'ResearchRAIL',
				'year': 2025
			},
			'hashes':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.3.0', 'hyperswap_1b_256.hash'),
					'path': resolve_relative_path('../.assets/models/hyperswap_1b_256.hash')
				}
			},
			'sources':
				{
					'face_swapper':
					{
						'url': resolve_download_url('models-3.3.0', 'hyperswap_1b_256.onnx'),
						'path': resolve_relative_path('../.assets/models/hyperswap_1b_256.onnx')
					}
				},
			'type': 'hyperswap',
			'template': 'arcface_128',
			'size': (256, 256),
			'mean': [ 0.5, 0.5, 0.5 ],
			'standard_deviation': [ 0.5, 0.5, 0.5 ]
		},
		'hyperswap_1c_256':
		{
			'__metadata__':
			{
				'vendor': 'FaceFusion',
				'license': 'ResearchRAIL',
				'year': 2025
			},
			'hashes':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.3.0', 'hyperswap_1c_256.hash'),
					'path': resolve_relative_path('../.assets/models/hyperswap_1c_256.hash')
				}
			},
			'sources':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.3.0', 'hyperswap_1c_256.onnx'),
					'path': resolve_relative_path('../.assets/models/hyperswap_1c_256.onnx')
				}
			},
			'type': 'hyperswap',
			'template': 'arcface_128',
			'size': (256, 256),
			'mean': [ 0.5, 0.5, 0.5 ],
			'standard_deviation': [ 0.5, 0.5, 0.5 ]
		},
		'inswapper_128':
		{
			'__metadata__':
			{
				'vendor': 'InsightFace',
				'license': 'Non-Commercial',
				'year': 2023
			},
			'hashes':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'inswapper_128.hash'),
					'path': resolve_relative_path('../.assets/models/inswapper_128.hash')
				}
			},
			'sources':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'inswapper_128.onnx'),
					'path': resolve_relative_path('../.assets/models/inswapper_128.onnx')
				}
			},
			'type': 'inswapper',
			'template': 'arcface_128',
			'size': (128, 128),
			'mean': [ 0.0, 0.0, 0.0 ],
			'standard_deviation': [ 1.0, 1.0, 1.0 ]
		},
		'inswapper_128_fp16':
		{
			'__metadata__':
			{
				'vendor': 'InsightFace',
				'license': 'Non-Commercial',
				'year': 2023
			},
			'hashes':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'inswapper_128_fp16.hash'),
					'path': resolve_relative_path('../.assets/models/inswapper_128_fp16.hash')
				}
			},
			'sources':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'inswapper_128_fp16.onnx'),
					'path': resolve_relative_path('../.assets/models/inswapper_128_fp16.onnx')
				}
			},
			'type': 'inswapper',
			'template': 'arcface_128',
			'size': (128, 128),
			'mean': [ 0.0, 0.0, 0.0 ],
			'standard_deviation': [ 1.0, 1.0, 1.0 ]
		},
		'simswap_256':
		{
			'__metadata__':
			{
				'vendor': 'neuralchen',
				'license': 'Non-Commercial',
				'year': 2020
			},
			'hashes':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'simswap_256.hash'),
					'path': resolve_relative_path('../.assets/models/simswap_256.hash')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.4.0', 'crossface_simswap.hash'),
					'path': resolve_relative_path('../.assets/models/crossface_simswap.hash')
				}
			},
			'sources':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'simswap_256.onnx'),
					'path': resolve_relative_path('../.assets/models/simswap_256.onnx')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.4.0', 'crossface_simswap.onnx'),
					'path': resolve_relative_path('../.assets/models/crossface_simswap.onnx')
				}
			},
			'type': 'simswap',
			'template': 'arcface_112_v1',
			'size': (256, 256),
			'mean': [ 0.485, 0.456, 0.406 ],
			'standard_deviation': [ 0.229, 0.224, 0.225 ]
		},
		'simswap_unofficial_512':
		{
			'__metadata__':
			{
				'vendor': 'neuralchen',
				'license': 'Non-Commercial',
				'year': 2020
			},
			'hashes':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'simswap_unofficial_512.hash'),
					'path': resolve_relative_path('../.assets/models/simswap_unofficial_512.hash')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.4.0', 'crossface_simswap.hash'),
					'path': resolve_relative_path('../.assets/models/crossface_simswap.hash')
				}
			},
			'sources':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'simswap_unofficial_512.onnx'),
					'path': resolve_relative_path('../.assets/models/simswap_unofficial_512.onnx')
				},
				'embedding_converter':
				{
					'url': resolve_download_url('models-3.4.0', 'crossface_simswap.onnx'),
					'path': resolve_relative_path('../.assets/models/crossface_simswap.onnx')
				}
			},
			'type': 'simswap',
			'template': 'arcface_112_v1',
			'size': (512, 512),
			'mean': [ 0.0, 0.0, 0.0 ],
			'standard_deviation': [ 1.0, 1.0, 1.0 ]
		},
		'uniface_256':
		{
			'__metadata__':
			{
				'vendor': 'xc-csc101',
				'license': 'Unknown',
				'year': 2022
			},
			'hashes':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'uniface_256.hash'),
					'path': resolve_relative_path('../.assets/models/uniface_256.hash')
				}
			},
			'sources':
			{
				'face_swapper':
				{
					'url': resolve_download_url('models-3.0.0', 'uniface_256.onnx'),
					'path': resolve_relative_path('../.assets/models/uniface_256.onnx')
				}
			},
			'type': 'uniface',
			'template': 'ffhq_512',
			'size': (256, 256),
			'mean': [ 0.5, 0.5, 0.5 ],
			'standard_deviation': [ 0.5, 0.5, 0.5 ]
		}
	}


def get_inference_pool() -> InferencePool:
	model_names = [ get_model_name() ]
	model_source_set = get_model_options().get('sources')

	return inference_manager.get_inference_pool(__name__, model_names, model_source_set)


def clear_inference_pool() -> None:
	model_names = [ get_model_name() ]
	inference_manager.clear_inference_pool(__name__, model_names)


def get_model_options() -> ModelOptions:
	model_name = get_model_name()
	return create_static_model_set('full').get(model_name)


def get_model_name() -> str:
	model_name = state_manager.get_item('face_swapper_model')

	if is_macos() and has_execution_provider('coreml') and model_name == 'inswapper_128_fp16':
		return 'inswapper_128'
	return model_name


def register_args(program : ArgumentParser) -> None:
	group_processors = find_argument_group(program, 'processors')
	if group_processors:
		group_processors.add_argument('--face-swapper-model', help = translator.get('help.model', __package__), default = config.get_str_value('processors', 'face_swapper_model', 'hyperswap_1a_256'), choices = face_swapper_choices.face_swapper_models)
		known_args, _ = program.parse_known_args()
		face_swapper_pixel_boost_choices = face_swapper_choices.face_swapper_set.get(known_args.face_swapper_model)
		group_processors.add_argument('--face-swapper-pixel-boost', help = translator.get('help.pixel_boost', __package__), default = config.get_str_value('processors', 'face_swapper_pixel_boost', get_first(face_swapper_pixel_boost_choices)), choices = face_swapper_pixel_boost_choices)
		group_processors.add_argument('--face-swapper-weight', help = translator.get('help.weight', __package__), type = float, default = config.get_float_value('processors', 'face_swapper_weight', '0.5'), choices = face_swapper_choices.face_swapper_weight_range)
		facefusion.jobs.job_store.register_step_keys([ 'face_swapper_model', 'face_swapper_pixel_boost', 'face_swapper_weight' ])


def apply_args(args : Args, apply_state_item : ApplyStateItem) -> None:
	apply_state_item('face_swapper_model', args.get('face_swapper_model'))
	apply_state_item('face_swapper_pixel_boost', args.get('face_swapper_pixel_boost'))
	apply_state_item('face_swapper_weight', args.get('face_swapper_weight'))


def pre_check() -> bool:
	model_hash_set = get_model_options().get('hashes')
	model_source_set = get_model_options().get('sources')

	return conditional_download_hashes(model_hash_set) and conditional_download_sources(model_source_set)


def pre_process(mode : ProcessMode) -> bool:
	logger.info('[face_swapper] pre_process called', __name__)
	
	if not has_image(state_manager.get_item('source_paths')):
		logger.error(translator.get('choose_image_source') + translator.get('exclamation_mark'), __name__)
		return False

	source_image_paths = filter_image_paths(state_manager.get_item('source_paths'))
	logger.info(f'[face_swapper] Found {len(source_image_paths)} source images', __name__)
	
	source_frames = read_static_images(source_image_paths)
	source_faces = get_many_faces(source_frames)
	
	logger.info(f'[face_swapper] Detected {len(source_faces)} source faces', __name__)

	if not get_one_face(source_faces):
		logger.error(translator.get('no_source_face_detected') + translator.get('exclamation_mark'), __name__)
		return False
	
	logger.info('[face_swapper] pre_process passed', __name__)

	if mode in [ 'output', 'preview' ] and not is_image(state_manager.get_item('target_path')) and not is_video(state_manager.get_item('target_path')):
		logger.error(translator.get('choose_image_or_video_target') + translator.get('exclamation_mark'), __name__)
		return False

	if mode == 'output' and not in_directory(state_manager.get_item('output_path')):
		logger.error(translator.get('specify_image_or_video_output') + translator.get('exclamation_mark'), __name__)
		return False

	if mode == 'output' and not same_file_extension(state_manager.get_item('target_path'), state_manager.get_item('output_path')):
		logger.error(translator.get('match_target_and_output_extension') + translator.get('exclamation_mark'), __name__)
		return False

	return True


def post_process() -> None:
	read_static_image.cache_clear()
	read_static_video_frame.cache_clear()
	video_manager.clear_video_pool()
	if state_manager.get_item('video_memory_strategy') in [ 'strict', 'moderate' ]:
		get_static_model_initializer.cache_clear()
		clear_inference_pool()
	if state_manager.get_item('video_memory_strategy') == 'strict':
		content_analyser.clear_inference_pool()
		face_classifier.clear_inference_pool()
		face_detector.clear_inference_pool()
		face_landmarker.clear_inference_pool()
		face_masker.clear_inference_pool()
		face_recognizer.clear_inference_pool()


def swap_face(source_face : Face, target_face : Face, temp_vision_frame : VisionFrame) -> VisionFrame:
	model_template = get_model_options().get('template')
	model_size = get_model_options().get('size')
	pixel_boost_size = unpack_resolution(state_manager.get_item('face_swapper_pixel_boost'))
	pixel_boost_total = pixel_boost_size[0] // model_size[0]
	crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(temp_vision_frame, target_face.landmark_set.get('5/68'), model_template, pixel_boost_size)
	temp_vision_frames = []
	crop_masks = []

	if 'box' in state_manager.get_item('face_mask_types'):
		box_mask = create_box_mask(crop_vision_frame, state_manager.get_item('face_mask_blur'), state_manager.get_item('face_mask_padding'))
		crop_masks.append(box_mask)

	if 'occlusion' in state_manager.get_item('face_mask_types'):
		occlusion_mask = create_occlusion_mask(crop_vision_frame)
		crop_masks.append(occlusion_mask)

	pixel_boost_vision_frames = implode_pixel_boost(crop_vision_frame, pixel_boost_total, model_size)
	for pixel_boost_vision_frame in pixel_boost_vision_frames:
		pixel_boost_vision_frame = prepare_crop_frame(pixel_boost_vision_frame)
		pixel_boost_vision_frame = forward_swap_face(source_face, target_face, pixel_boost_vision_frame)
		pixel_boost_vision_frame = normalize_crop_frame(pixel_boost_vision_frame)
		temp_vision_frames.append(pixel_boost_vision_frame)
	crop_vision_frame = explode_pixel_boost(temp_vision_frames, pixel_boost_total, model_size, pixel_boost_size)

	if 'area' in state_manager.get_item('face_mask_types'):
		face_landmark_68 = cv2.transform(target_face.landmark_set.get('68').reshape(1, -1, 2), affine_matrix).reshape(-1, 2)
		area_mask = create_area_mask(crop_vision_frame, face_landmark_68, state_manager.get_item('face_mask_areas'))
		crop_masks.append(area_mask)

	if 'region' in state_manager.get_item('face_mask_types'):
		region_mask = create_region_mask(crop_vision_frame, state_manager.get_item('face_mask_regions'))
		crop_masks.append(region_mask)

	crop_mask = numpy.minimum.reduce(crop_masks).clip(0, 1)
	paste_vision_frame = paste_back(temp_vision_frame, crop_vision_frame, crop_mask, affine_matrix)
	return paste_vision_frame


def forward_swap_face(source_face : Face, target_face : Face, crop_vision_frame : VisionFrame) -> VisionFrame:
	face_swapper = get_inference_pool().get('face_swapper')
	model_type = get_model_options().get('type')
	face_swapper_inputs = {}

	if is_macos() and has_execution_provider('coreml') and model_type in [ 'ghost', 'uniface' ]:
		face_swapper.set_providers([ facefusion.choices.execution_provider_set.get('cpu') ])

	for face_swapper_input in face_swapper.get_inputs():
		if face_swapper_input.name == 'source':
			if model_type in [ 'blendswap', 'uniface' ]:
				face_swapper_inputs[face_swapper_input.name] = prepare_source_frame(source_face)
			else:
				source_embedding = prepare_source_embedding(source_face)
				source_embedding = balance_source_embedding(source_embedding, target_face.embedding)
				face_swapper_inputs[face_swapper_input.name] = source_embedding
		if face_swapper_input.name == 'target':
			face_swapper_inputs[face_swapper_input.name] = crop_vision_frame

	with conditional_thread_semaphore():
		crop_vision_frame = face_swapper.run(None, face_swapper_inputs)[0][0]

	return crop_vision_frame


def forward_convert_embedding(face_embedding : Embedding) -> Embedding:
	embedding_converter = get_inference_pool().get('embedding_converter')

	with conditional_thread_semaphore():
		face_embedding = embedding_converter.run(None,
		{
			'input': face_embedding
		})[0]

	return face_embedding


def prepare_source_frame(source_face : Face) -> VisionFrame:
	model_type = get_model_options().get('type')
	source_vision_frame = read_static_image(get_first(state_manager.get_item('source_paths')))

	if model_type == 'blendswap':
		source_vision_frame, _ = warp_face_by_face_landmark_5(source_vision_frame, source_face.landmark_set.get('5/68'), 'arcface_112_v2', (112, 112))

	if model_type == 'uniface':
		source_vision_frame, _ = warp_face_by_face_landmark_5(source_vision_frame, source_face.landmark_set.get('5/68'), 'ffhq_512', (256, 256))

	source_vision_frame = source_vision_frame[:, :, ::-1] / 255.0
	source_vision_frame = source_vision_frame.transpose(2, 0, 1)
	source_vision_frame = numpy.expand_dims(source_vision_frame, axis = 0).astype(numpy.float32)
	return source_vision_frame


def prepare_source_embedding(source_face : Face) -> Embedding:
	model_type = get_model_options().get('type')

	if model_type == 'ghost':
		source_embedding = source_face.embedding.reshape(-1, 512)
		source_embedding, _ = convert_source_embedding(source_embedding)
		source_embedding = source_embedding.reshape(1, -1)
		return source_embedding

	if model_type == 'hyperswap':
		source_embedding = source_face.embedding_norm.reshape((1, -1))
		return source_embedding

	if model_type == 'inswapper':
		model_path = get_model_options().get('sources').get('face_swapper').get('path')
		model_initializer = get_static_model_initializer(model_path)
		source_embedding = source_face.embedding.reshape((1, -1))
		source_embedding = numpy.dot(source_embedding, model_initializer) / numpy.linalg.norm(source_embedding)
		return source_embedding

	source_embedding = source_face.embedding.reshape(-1, 512)
	_, source_embedding_norm = convert_source_embedding(source_embedding)
	source_embedding = source_embedding_norm.reshape(1, -1)
	return source_embedding


def balance_source_embedding(source_embedding : Embedding, target_embedding : Embedding) -> Embedding:
	model_type = get_model_options().get('type')
	face_swapper_weight = state_manager.get_item('face_swapper_weight')
	face_swapper_weight = numpy.interp(face_swapper_weight, [ 0, 1 ], [ 0.35, -0.35 ]).astype(numpy.float32)

	if model_type in [ 'hififace', 'hyperswap', 'inswapper', 'simswap' ]:
		target_embedding = target_embedding / numpy.linalg.norm(target_embedding)

	source_embedding = source_embedding.reshape(1, -1)
	target_embedding = target_embedding.reshape(1, -1)
	source_embedding = source_embedding * (1 - face_swapper_weight) + target_embedding * face_swapper_weight
	return source_embedding


def convert_source_embedding(source_embedding : Embedding) -> Tuple[Embedding, Embedding]:
	source_embedding = forward_convert_embedding(source_embedding)
	source_embedding = source_embedding.ravel()
	source_embedding_norm = source_embedding / numpy.linalg.norm(source_embedding)
	return source_embedding, source_embedding_norm


def prepare_crop_frame(crop_vision_frame : VisionFrame) -> VisionFrame:
	model_mean = get_model_options().get('mean')
	model_standard_deviation = get_model_options().get('standard_deviation')

	crop_vision_frame = crop_vision_frame[:, :, ::-1] / 255.0
	crop_vision_frame = (crop_vision_frame - model_mean) / model_standard_deviation
	crop_vision_frame = crop_vision_frame.transpose(2, 0, 1)
	crop_vision_frame = numpy.expand_dims(crop_vision_frame, axis = 0).astype(numpy.float32)
	return crop_vision_frame


def normalize_crop_frame(crop_vision_frame : VisionFrame) -> VisionFrame:
	model_type = get_model_options().get('type')
	model_mean = get_model_options().get('mean')
	model_standard_deviation = get_model_options().get('standard_deviation')

	crop_vision_frame = crop_vision_frame.transpose(1, 2, 0)

	if model_type in [ 'ghost', 'hififace', 'hyperswap', 'uniface' ]:
		crop_vision_frame = crop_vision_frame * model_standard_deviation + model_mean

	crop_vision_frame = crop_vision_frame.clip(0, 1)
	crop_vision_frame = crop_vision_frame[:, :, ::-1] * 255
	return crop_vision_frame


def extract_source_face(source_vision_frames : List[VisionFrame]) -> Optional[Face]:
	"""Extract source face - returns averaged face for backward compatibility"""
	source_faces = extract_source_faces(source_vision_frames)
	return get_average_face(source_faces) if source_faces else None


def extract_source_faces(source_vision_frames : List[VisionFrame]) -> List[Face]:
	"""Extract all source faces from source images - returns list of faces, one per source image"""
	source_faces = []

	if source_vision_frames:
		for source_vision_frame in source_vision_frames:
			temp_faces = get_many_faces([source_vision_frame])
			temp_faces = sort_faces_by_order(temp_faces, 'large-small')

			if temp_faces:
				source_faces.append(get_first(temp_faces))

	return source_faces


def process_frame(inputs : FaceSwapperInputs) -> ProcessorOutputs:
	# Log that we're being called (at info level so it always shows)
	if inputs.get('frame_number', 0) == 0:
		logger.info('[face_swapper] process_frame called', __name__)
	reference_vision_frame = inputs.get('reference_vision_frame')
	source_vision_frames = inputs.get('source_vision_frames')
	target_vision_frame = inputs.get('target_vision_frame')
	temp_vision_frame = inputs.get('temp_vision_frame')
	temp_vision_mask = inputs.get('temp_vision_mask')
	frame_number = inputs.get('frame_number', 0)
	
	# Extract all source faces
	source_faces = extract_source_faces(source_vision_frames)
	target_faces = select_faces(reference_vision_frame, target_vision_frame, frame_number)
	
	# Log at info level for visibility (only log first frame to avoid spam)
	if frame_number == 0:
		logger.info(f'[face_swapper] Frame {frame_number}: {len(source_faces)} source faces, {len(target_faces)} target faces', __name__)

	if source_faces and target_faces:
		# Get cluster-to-source mapping from state_manager
		cluster_source_mapping = state_manager.get_item('cluster_source_mapping')
		
		# Log at info level for visibility (only log first frame)
		if frame_number == 0:
			logger.info(f'[face_swapper] Frame {frame_number}: cluster_source_mapping={cluster_source_mapping}, source_count={len(source_faces)}', __name__)
			logger.info(f'[face_swapper] cluster_source_mapping type: {type(cluster_source_mapping)}, is None: {cluster_source_mapping is None}, is empty: {cluster_source_mapping == {}}', __name__)
		
		# If no mapping or only one source, use backward-compatible behavior
		if not cluster_source_mapping or len(source_faces) == 1:
			if frame_number == 0:
				if not cluster_source_mapping:
					logger.warn(f'[face_swapper] Using backward-compatible mode: cluster_source_mapping is {cluster_source_mapping}', __name__)
				if len(source_faces) == 1:
					logger.warn(f'[face_swapper] Using backward-compatible mode: only 1 source face detected', __name__)
				logger.info(f'[face_swapper] Using backward-compatible mode (averaged face)', __name__)
			source_face = get_average_face(source_faces) if source_faces else None
			if source_face:
				for target_face in target_faces:
					target_face = scale_face(target_face, target_vision_frame, temp_vision_frame)
					temp_vision_frame = swap_face(source_face, target_face, temp_vision_frame)
		else:
			# Multi-face mode: map each target face to appropriate source face
			from facefusion.video_face_database import get_cluster_for_face
			
			if frame_number == 0:
				logger.info(f'[face_swapper] Using multi-face mode with {len(source_faces)} sources', __name__)
			
			for target_face_index, target_face in enumerate(target_faces):
				target_face = scale_face(target_face, target_vision_frame, temp_vision_frame)
				
				# Find which source face to use for this target face
				source_face = None
				
				# Check if this is a forced replacement face first
				from facefusion.forced_replacements import get_forced_replacements_for_frame
				forced_replacements = get_forced_replacements_for_frame(frame_number) if frame_number is not None else []
				
				# Match forced replacement by bounding box position
				forced_repl = None
				if forced_replacements:
					target_bbox = target_face.bounding_box
					target_center = ((target_bbox[0] + target_bbox[2]) / 2, (target_bbox[1] + target_bbox[3]) / 2)
					
					for fr in forced_replacements:
						fr_bbox = fr['bounding_box']
						fr_center = ((fr_bbox[0] + fr_bbox[2]) / 2, (fr_bbox[1] + fr_bbox[3]) / 2)
						distance = ((target_center[0] - fr_center[0])**2 + (target_center[1] - fr_center[1])**2)**0.5
						
						# If within 50 pixels, this is the forced replacement
						if distance < 50:
							forced_repl = fr
							break
				
				# If this is a forced replacement, use its source face (takes priority)
				if forced_repl:
					source_index = forced_repl['source_face_index']
					if 0 <= source_index < len(source_faces):
						source_face = source_faces[source_index]
						if frame_number is not None and frame_number < 5:
							logger.info(f'[face_swapper] Frame {frame_number}, face {target_face_index}: Using forced replacement source {source_index}', __name__)
				
				# Otherwise, try to get cluster ID for this target face from database
				if source_face is None:
					# Since select_faces() may sort/filter faces, we need to match by position, not index
					cluster_id = None
					if frame_number is not None:
						# Try to match by bounding box position to find the correct face_index in database
						from facefusion.video_face_database import get_video_face_database
						database = get_video_face_database()
						
						if database:
							# Get all face instances for this frame from all clusters
							frame_instances = []
							for cluster in database['clusters']:
								for inst in cluster['all_instances']:
									if inst['frame_number'] == frame_number:
										frame_instances.append(inst)
							
							# Match target_face to database instance by bounding box position
							target_bbox = target_face.bounding_box
							best_match = None
							best_distance = float('inf')
							
							for inst in frame_instances:
								inst_bbox = inst['bounding_box']
								# Calculate center distance
								target_center = ((target_bbox[0] + target_bbox[2]) / 2, (target_bbox[1] + target_bbox[3]) / 2)
								inst_center = ((inst_bbox[0] + inst_bbox[2]) / 2, (inst_bbox[1] + inst_bbox[3]) / 2)
								distance = ((target_center[0] - inst_center[0])**2 + (target_center[1] - inst_center[1])**2)**0.5
								
								if distance < best_distance:
									best_distance = distance
									best_match = inst
							
							# If we found a good match (within reasonable distance), use its cluster
							if best_match and best_distance < 50:  # 50 pixels threshold
								cluster_id = database['face_to_cluster'].get((frame_number, best_match['face_index']))
								if frame_number < 5:
									logger.info(f'[face_swapper] Frame {frame_number}, face {target_face_index}: matched to DB face_index {best_match["face_index"]}, cluster_id={cluster_id}', __name__)
							else:
								# Fallback: try original index-based lookup
								cluster_id = get_cluster_for_face(frame_number, target_face_index)
								if frame_number < 5:
									logger.warn(f'[face_swapper] Frame {frame_number}, face {target_face_index}: no bbox match (dist={best_distance}), trying index lookup: cluster_id={cluster_id}', __name__)
						else:
							# No database, use index-based lookup
							cluster_id = get_cluster_for_face(frame_number, target_face_index)
						
						if frame_number < 5:  # Log first few frames for debugging
							logger.info(f'[face_swapper] Frame {frame_number}, face {target_face_index}: final cluster_id={cluster_id}, mapping={cluster_source_mapping}', __name__)
					
					# Get source face from mapping if cluster found
					# Handle both string and integer keys in mapping (due to JSON serialization)
					if cluster_id is not None:
						# Try integer key first, then string key
						source_index = None
						if cluster_id in cluster_source_mapping:
							source_index = cluster_source_mapping[cluster_id]
						elif str(cluster_id) in cluster_source_mapping:
							source_index = cluster_source_mapping[str(cluster_id)]
						
						if source_index is not None:
							if frame_number < 5:
								logger.info(f'[face_swapper] Frame {frame_number}, face {target_face_index}: ✓ MAPPED cluster {cluster_id} → source {source_index}', __name__)
							if 0 <= source_index < len(source_faces):
								source_face = source_faces[source_index]
							else:
								if frame_number < 5:
									logger.warn(f'[face_swapper] Frame {frame_number}, face {target_face_index}: source_index {source_index} out of range (max: {len(source_faces)-1})', __name__)
						else:
							# Cluster not mapped - skip replacement for this face
							if frame_number < 5:
								logger.info(f'[face_swapper] Frame {frame_number}, face {target_face_index}: ⊘ SKIP - cluster {cluster_id} not in mapping {list(cluster_source_mapping.keys())}', __name__)
							source_face = None  # Explicitly set to None to skip
					else:
						# No cluster_id found - skip replacement
						if frame_number < 5:
							logger.info(f'[face_swapper] Frame {frame_number}, face {target_face_index}: ⊘ SKIP - no cluster_id found', __name__)
						source_face = None
				
				# Only swap if we have a valid source face from mapping
				# NO FALLBACKS - if not mapped, don't replace
				if source_face:
					temp_vision_frame = swap_face(source_face, target_face, temp_vision_frame)
				elif frame_number < 5:
					logger.info(f'[face_swapper] Frame {frame_number}, face {target_face_index}: ⊘ SKIPPED - no mapping, face not replaced', __name__)
	else:
		if frame_number == 0:
			logger.warn(f'[face_swapper] Frame {frame_number}: no source faces ({len(source_faces)}) or target faces ({len(target_faces)})', __name__)

	return temp_vision_frame, temp_vision_mask
