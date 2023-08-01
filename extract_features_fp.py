import torch
import torch.nn as nn
from math import floor
import os
import random
import numpy as np
import pdb
import time
from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from torch.utils.data import DataLoader
from models.resnet_custom import resnet50_baseline, resnet_50_histo_pretr, resnet18_baseline
import models.models_vit as models_vit
import argparse
from utils.utils import print_network, collate_features
from utils.file_utils import save_hdf5
from PIL import Image
import h5py
import openslide
import timm
from timm.models.layers import trunc_normal_
from timm.models._hub import load_state_dict_from_hf
from utils.pos_embed import interpolate_pos_embed
from HIPT.hipt_model_utils import get_vit256

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



def load_model_weights(model, weights):

    model_dict = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict}
    if weights == {}:
        print('No weight could be loaded..')
    model_dict.update(weights)
    msg = model.load_state_dict(model_dict)
    print(msg)
    model.load_state_dict(model_dict)
    return model


def load_ViT_model_from_pretr(model_name, path_to_model_checkpoint=None):
	"""
	Load ViT model from pretrained checkpoint
	model (str): model name/size [vit_large_patch16, vit_large_patch32]
	
	"""

	print()
	print(f"Loading model {model_name} from checkpoint {path_to_model_checkpoint} (If checkpoint is None, load pretrained weights from hf hub)")
	print()


	model = models_vit.__dict__[model_name](
        num_classes=1000,
        drop_path_rate=0.1,
        global_pool=True,
    )

	if path_to_model_checkpoint is None:
		if model_name == 'vit_large_patch16':
			pretrained_loc_224 = "timm/vit_large_patch16_224.augreg_in21k_ft_in1k"
			checkpoint_model = load_state_dict_from_hf(pretrained_loc_224)

		elif model_name == 'vit_large_patch32':
			pretrained_loc_384 = "timm/vit_large_patch32_384.orig_in21k_ft_in1k"
			checkpoint_model = load_state_dict_from_hf(pretrained_loc_384)

		else:
			raise NotImplementedError
	
	else:
		checkpoint = torch.load(path_to_model_checkpoint)
		checkpoint_model = checkpoint['model']


	state_dict = model.state_dict()

	for k in ['head.weight', 'head.bias']:
		if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
			print(f"Removing key {k} from pretrained checkpoint")
			del checkpoint_model[k]

	# interpolate position embedding
	interpolate_pos_embed(model, checkpoint_model)

	# load pre-trained model
	msg = model.load_state_dict(checkpoint_model, strict=False)
	print(msg)

	# manually initialize fc layer
	trunc_normal_(model.head.weight, std=2e-5)

	return model



def compute_w_loader(file_path, output_path, wsi, model, model_type,
 	batch_size = 8, verbose = 0, print_every=20, pretrained=True, 
	custom_downsample=1, target_patch_size=-1):
	"""
	args:
		file_path: directory of bag (.h5 file)
		output_path: directory to save computed features (.h5 file)
		model: model
		model_type: important on how to extract features
		batch_size: batch_size for computing features in batches
		verbose: level of feedback
		pretrained: use weights pretrained on imagenet
		custom_downsample: custom defined downscale factor of image patches
		target_patch_size: custom defined, rescaled image size before embedding
	"""
	dataset = Whole_Slide_Bag_FP(file_path=file_path, wsi=wsi, pretrained=pretrained, 
		custom_downsample=custom_downsample, target_patch_size=target_patch_size)
	x, y = dataset[0]
	kwargs = {'num_workers': 4, 'pin_memory': True} if device.type == "cuda" else {}
	loader = DataLoader(dataset=dataset, batch_size=batch_size, **kwargs, collate_fn=collate_features)

	if verbose > 0:
		print('processing {}: total of {} batches'.format(file_path,len(loader)))

	mode = 'w'
	for count, (batch, coords) in enumerate(loader):
		with torch.no_grad():	
			if count % print_every == 0:
				print('batch {}/{}, {} files processed'.format(count, len(loader), count * batch_size))
			batch = batch.to(device, non_blocking=True)

			if model_type in ["resnet50", "resnet_50_histo_pretr",
		    'resnet_18', 'resnet_18_no_trunc',
			'resnet_18_sshisto', 'resnet_18_sshisto_no_trunc'
			'vit_small_patch16_384_HIPT_pretr']:
				features = model(batch)
				features = features.cpu().numpy()

			elif model_type in ["vit_large_patch16_224_norm_pretr", "vit_large_patch16_384_norm_pretr", 
		       "vit_large_patch16_224_MAE_pretr", "vit_large_patch16_224_MAE_histopatho_pretr"]  :
				features = model.forward_features(batch)
				features = features.cpu().numpy()

			else:
				raise NotImplementedError

			asset_dict = {'features': features, 'coords': coords}
			save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
			mode = 'a'
	
	return output_path


parser = argparse.ArgumentParser(description='Feature Extraction')
parser.add_argument('--data_h5_dir', type=str, default=None)
parser.add_argument('--data_slide_dir', type=str, default=None)
parser.add_argument('--slide_ext', type=str, default= '.svs')
parser.add_argument('--csv_path', type=str, default=None)
parser.add_argument('--feat_dir', type=str, default=None)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--no_auto_skip', default=False, action='store_true')
parser.add_argument('--custom_downsample', type=int, default=1)
parser.add_argument('--target_patch_size', type=int, default=-1)
parser.add_argument('--model', type=str, help="model to load for feature extraction", 
		    choices=['resnet50', 'resnet_50_histo_pretr',
	        'resnet_18', 'resnet_18_no_trunc',
	        'resnet_18_sshisto', 'resnet_18_sshisto_no_trunc', 
			'vit_large_patch16_224_norm_pretr', 'vit_large_patch16_384_norm_pretr',
	        'vit_large_patch16_224_MAE_pretr', 'vit_large_patch16_224_MAE_histopatho_pretr',
			'vit_small_patch16_384_HIPT_pretr'])
parser.add_argument('--path_to_model_checkpoint', default=None, type=str, help="path to model checkpoint")
args = parser.parse_args()


if __name__ == '__main__':

	print('initializing dataset')
	csv_path = args.csv_path
	if csv_path is None:
		raise NotImplementedError

	bags_dataset = Dataset_All_Bags(csv_path)
	
	os.makedirs(args.feat_dir, exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'pt_files'), exist_ok=True)
	os.makedirs(os.path.join(args.feat_dir, 'h5_files'), exist_ok=True)
	dest_files = os.listdir(os.path.join(args.feat_dir, 'pt_files'))


	if args.model == 'resnet50':
		print('loading model checkpoint for resnet50')
		model = resnet50_baseline(pretrained=True)

	elif args.model == 'resnet_50_histo_pretr':
		print('loading model checkpoint for resnet_50_histo_pretr')
		model = resnet_50_histo_pretr()

	elif args.model == 'resnet_18':
		print('loading model checkpoint for resnet_18')
		model = resnet18_baseline(pretrained=True, truncated=True)

	elif args.model == 'resnet_18_no_trunc':
		print('loading model checkpoint for resnet_18')
		model = resnet18_baseline(pretrained=True, truncated=False)
	
	elif args.model == 'resnet_18_sshisto':
		print('loading model checkpoint for resnet_18_sshisto')
		model = resnet18_baseline(pretrained=False, truncated=True)
		MODEL_PATH = '/data/Maack/PANT/CLAM/experiments/resnet_18_sshisto/tenpercent_resnet18.ckpt'
		state = torch.load(MODEL_PATH)
		state_dict = state['state_dict']
		for key in list(state_dict.keys()):
			state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)
		model = load_model_weights(model, state_dict)

	elif args.model == 'resnet_18_sshisto_no_trunc':
		print('loading model checkpoint for resnet_18_sshisto')
		model = resnet18_baseline(pretrained=False, truncated=False)
		MODEL_PATH = '/data/Maack/PANT/CLAM/experiments/resnet_18_sshisto/tenpercent_resnet18.ckpt'
		state = torch.load(MODEL_PATH)
		state_dict = state['state_dict']
		for key in list(state_dict.keys()):
			state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)
		model = load_model_weights(model, state_dict)

	elif args.model == 'vit_large_patch16_224_norm_pretr':
		print("loading model checkpoint for ViT 224_norm_pretr")
		model = load_ViT_model_from_pretr(model_name='vit_large_patch16', path_to_model_checkpoint=None)
		print("model loaded")

	elif args.model == 'vit_large_patch16_384_norm_pretr':
		print("loading model checkpoint for ViT 384_norm_pretr")
		model = load_ViT_model_from_pretr(model_name='vit_large_patch32', path_to_model_checkpoint=None)
		print("model loaded")
	
	elif args.model == 'vit_large_patch16_224_MAE_pretr':
		print("loading model checkpoint for ViT 224_MAE_pretr")
		model = load_ViT_model_from_pretr(model_name='vit_large_patch16', 
				    path_to_model_checkpoint="/home/Maack/Medulloblastoma/CLAM/mae_pretrain_vit_large_ImageNet.pth")
		print("model loaded")
	
	elif args.model == 'vit_large_patch16_224_MAE_histopatho_pretr':
		if args.path_to_model_checkpoint is not None:
			print("loading model checkpoint for ViT 224_MAE_histopatho_pretr")
			model = load_ViT_model_from_pretr(model_name='vit_large_patch16',
				path_to_model_checkpoint=args.path_to_model_checkpoint)
			print("model loaded")
		else:
			raise NotImplementedError

	elif args.model == 'vit_small_patch16_384_HIPT_pretr':
		print("loading model checkpoint for vit_small_patch16_384_HIPT_pretr")
		model = get_vit256("/data/Maack/PANT/CLAM/experiments/vit_small_patch16_384_HIPT_pretr/vit256_small_dino.pth")
		print("model loaded")
	
	else:
		raise NotImplementedError

	model = model.to(device)
	
	# print_network(model)
	if torch.cuda.device_count() > 1:
		model = nn.DataParallel(model)
		
	model.eval()
	total = len(bags_dataset)

	for bag_candidate_idx in range(total):
		slide_id = bags_dataset[bag_candidate_idx].split(args.slide_ext)[0]
		bag_name = slide_id+'.h5'
		h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
		slide_file_path = os.path.join(args.data_slide_dir, slide_id+args.slide_ext)
		print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
		print(slide_id)

		if not args.no_auto_skip and slide_id+'.pt' in dest_files:
			print('skipped {}'.format(slide_id))
			continue 

		output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
		time_start = time.time()
		wsi = openslide.open_slide(slide_file_path)
		output_file_path = compute_w_loader(h5_file_path, output_path, wsi, 
		model = model, model_type=args.model, batch_size = args.batch_size, verbose = 1, print_every = 20, 
		custom_downsample=args.custom_downsample, target_patch_size=args.target_patch_size)
		time_elapsed = time.time() - time_start
		print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))
		file = h5py.File(output_file_path, "r")

		features = file['features'][:]
		print('features size: ', features.shape)
		print('coordinates size: ', file['coords'].shape)
		features = torch.from_numpy(features)
		bag_base, _ = os.path.splitext(bag_name)
		torch.save(features, os.path.join(args.feat_dir, 'pt_files', bag_base+'.pt'))



