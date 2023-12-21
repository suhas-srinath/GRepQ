from torchvision import transforms
from data_read_utils import *
from skimage.io import imread
import torch.utils.data
from PIL import Image
from piq import fsim
import logging
import numpy

logging.getLogger('PIL').setLevel(logging.WARNING)


# Read training data
def get_train_dataset(base_dataset_path: Path, train_datasets):
    train_data = {'images': None}

    for curr_set in train_datasets:
        curr_list = get_dataset_list(base_dataset_path, curr_set)
        train_data['images'] = curr_list['image_paths']

    return train_data


# Dataloader for the low level model
class FrameLoaderLLModel(torch.utils.data.Dataset):
    def __init__(self, learning_data: dict):

        self.learning_data = learning_data
        self.transform = transforms.ToTensor()
        self.flip = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5), transforms.RandomVerticalFlip(p=0.5)])

    def __len__(self):
        return len(self.learning_data['images'])

    def __getitem__(self, idx):
        annotator_matrix = None  # Initially setting to None
        image_dir = self.learning_data['images'][idx]  # Directory of all distorted versions of the same image
        all_dists = sorted(os.listdir(image_dir))
        all_distortions = [os.path.join(image_dir, dist) for dist in all_dists]  # Full path to each image
        
        images = []  
        torch_permuted_images = []
        aligned_param = len(all_distortions) # number of spatial fragmentation

        for dist in all_distortions:
            curr_image = imread(dist) / 255.0
            curr_image_torch = torch.from_numpy(curr_image)
            images.append(curr_image_torch)
            torch_permuted_images.append(curr_image_torch.permute(2, 0, 1))
        
        annotator_matrix = numpy.zeros((9, 9)) # 9 images in a folder including the synthetically undistorted image

        for i in range(len(torch_permuted_images)):
            for j in range(0, i):
                im1 = torch_permuted_images[i].unsqueeze(0)
                im2 = torch_permuted_images[j].unsqueeze(0)
                annotator_matrix[i][j] = fsim(im1, im2, data_range=1.0)

        annotator_matrix = annotator_matrix + numpy.transpose(annotator_matrix) # fsim is symmetric
        numpy.fill_diagonal(annotator_matrix, float(1))

        video_for_fragments = torch.stack(images, 0)
        video_for_fragments = video_for_fragments.permute(3, 0, 1, 2)  # have to change to [C,T,H,W] from [T,H,W,C] for the below function
        
        fragmented_anchor_video = self.get_spatial_fragments(video_for_fragments, aligned= aligned_param)
        fragmented_anchor_video = fragmented_anchor_video.permute(1, 0, 2, 3)
        
        fragmented_augmented_video = self.get_spatial_fragments(video_for_fragments, aligned= aligned_param)
        fragmented_augmented_video = fragmented_augmented_video.permute(1, 0, 2, 3)
        flipped_frag_aug_video = self.flip(fragmented_augmented_video)
        
        fragmented_images = list(torch.unbind(fragmented_anchor_video))
        fragmented_augmentations = list(torch.unbind(flipped_frag_aug_video))

        return_sample = {
            "images": (torch.stack(fragmented_images, dim=0)).to(torch.float32),
            "augmentations": (torch.stack(fragmented_augmentations, dim=0)).to(torch.float32),
            "annotators": (torch.from_numpy(annotator_matrix)).to(torch.float32)
        }

        return return_sample
    
    # Function is from FastVQA (https://github.com/VQAssessment/FAST-VQA-and-FasterVQA)
    # @staticmethod
    def get_spatial_fragments(
            self,
            video,
            fragments_h=7,
            fragments_w=7,
            fsize_h=32,
            fsize_w=32,
            aligned=9,  # changed to 9 because this indicates the no. of frames fragmented at same locations
            nfrags=1,
            random=False,
            random_upsample=False,
            fallback_type="upsample",
            **kwargs,
    ):
        size_h = fragments_h * fsize_h
        size_w = fragments_w * fsize_w
        ## video: [C,T,H,W]
        ## situation for images
        if video.shape[1] == 1:
            aligned = 1

        dur_t, res_h, res_w = video.shape[-3:]
        ratio = min(res_h / size_h, res_w / size_w)
        if fallback_type == "upsample" and ratio < 1:
            ovideo = video
            video = torch.nn.functional.interpolate(
                video / 255.0, scale_factor=1 / ratio, mode="bilinear"
            )
            video = (video * 255.0).type_as(ovideo)

        if random_upsample:
            randratio = random.random() * 0.5 + 1
            video = torch.nn.functional.interpolate(
                video / 255.0, scale_factor=randratio, mode="bilinear"
            )
            video = (video * 255.0).type_as(ovideo)

        assert dur_t % aligned == 0, "Please provide match vclip and align index"
        size = size_h, size_w

        ## make sure that sampling will not run out of the picture
        hgrids = torch.LongTensor(
            [min(res_h // fragments_h * i, res_h - fsize_h) for i in range(fragments_h)]
        )
        wgrids = torch.LongTensor(
            [min(res_w // fragments_w * i, res_w - fsize_w) for i in range(fragments_w)]
        )
        hlength, wlength = res_h // fragments_h, res_w // fragments_w

        if random:
            print("This part is deprecated. Please remind that.")
            if res_h > fsize_h:
                rnd_h = torch.randint(
                    res_h - fsize_h, (len(hgrids), len(wgrids), dur_t // aligned)
                )
            else:
                rnd_h = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
            if res_w > fsize_w:
                rnd_w = torch.randint(
                    res_w - fsize_w, (len(hgrids), len(wgrids), dur_t // aligned)
                )
            else:
                rnd_w = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
        else:
            if hlength > fsize_h:
                rnd_h = torch.randint(
                    hlength - fsize_h, (len(hgrids), len(wgrids), dur_t // aligned)
                )
            else:
                rnd_h = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()
            if wlength > fsize_w:
                rnd_w = torch.randint(
                    wlength - fsize_w, (len(hgrids), len(wgrids), dur_t // aligned)
                )
            else:
                rnd_w = torch.zeros((len(hgrids), len(wgrids), dur_t // aligned)).int()

        target_video = torch.zeros(video.shape[:-2] + size).to(video.device)
        # target_videos = []

        for i, hs in enumerate(hgrids):
            for j, ws in enumerate(wgrids):
                for t in range(dur_t // aligned):
                    t_s, t_e = t * aligned, (t + 1) * aligned
                    h_s, h_e = i * fsize_h, (i + 1) * fsize_h
                    w_s, w_e = j * fsize_w, (j + 1) * fsize_w
                    if random:
                        h_so, h_eo = rnd_h[i][j][t], rnd_h[i][j][t] + fsize_h
                        w_so, w_eo = rnd_w[i][j][t], rnd_w[i][j][t] + fsize_w
                    else:
                        h_so, h_eo = hs + rnd_h[i][j][t], hs + rnd_h[i][j][t] + fsize_h
                        w_so, w_eo = ws + rnd_w[i][j][t], ws + rnd_w[i][j][t] + fsize_w
                    target_video[:, t_s:t_e, h_s:h_e, w_s:w_e] = video[
                                                                :, t_s:t_e, h_so:h_eo, w_so:w_eo
                                                                ]
        return target_video

# Dataloader for high-level model
class FrameLoaderHLModel(torch.utils.data.Dataset):
    def __init__(self, learning_data: dict, crop_size=(224, 224)):
        
        self.crop_size = crop_size
        self.learning_data = learning_data

        self.flip = transforms.RandomHorizontalFlip(p=1)
        self.center_crop = transforms.CenterCrop(crop_size)
        self.random_crop = transforms.RandomCrop(crop_size)

        # CLIP preprocessing steps. Same thing applied.
        self.transform = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()
        self.normalizer = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    def __len__(self):
        return len(self.learning_data['images'])

    def __getitem__(self, idx):
        image_dir = self.learning_data['images'][idx]  # Directory of all distorted versions of an image
        all_dists = sorted(os.listdir(image_dir)) 
        all_distortions = [os.path.join(image_dir, dist) for dist in all_dists]  # Full path to each image
        preprocessed_img = None

        # Preprocessing images CenterCrop -> convert_image_to_rgb -> ToTensor -> Normalize
        for dist in all_distortions:
            if 'REF' in dist:
                PIL_image = Image.open(dist)
                preprocessed_img = self.convert_image_to_rgb(PIL_image)
                preprocessed_img = self.transform(preprocessed_img)
                preprocessed_img = self.normalizer(preprocessed_img)
                preprocessed_img = self.center_crop(preprocessed_img)
                
        return_sample = {
            "image": preprocessed_img,
        }

        return return_sample

    def convert_image_to_rgb(self, image):
        return image.convert("RGB")