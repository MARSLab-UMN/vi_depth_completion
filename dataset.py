import torch
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import skimage.io as sio
import pickle
import numpy as np
import os
import cv2
from PIL import Image
import logging
import warnings
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import fnmatch


def load_normal_image_framenet(filename, normalize=True):
    normal_img = Image.open(filename)
    normal_values = 1 - np.asarray(normal_img).astype(np.float32) / 127.5
    # Fix the shape
    normal_tensor = TF.to_tensor(normal_values)
    if normalize:
        normal_tensor = F.normalize(normal_tensor, dim=0)
    return normal_tensor


def load_depth_image(filename):
    depth_image = Image.open(filename)
    depth_values = np.asarray(depth_image).astype(np.float32) / 1000.0
    depth_tensor = torch.from_numpy(depth_values)[None, ...]
    return depth_tensor


def generate_image_homogeneous_coordinates(fc, cc, image_width, image_height):
    homogeneous = np.zeros((image_height, image_width, 3))
    homogeneous[:, :, 2] = 1

    xx, yy = np.meshgrid([i for i in range(0, image_width)], [i for i in range(0, image_height)])
    homogeneous[:, :, 0] = (xx - cc[0]) / fc[0]
    homogeneous[:, :, 1] = (yy - cc[1]) / fc[1]

    return torch.from_numpy(homogeneous.astype(np.float32))


def compute_alignment_tensor(gravity_tensor):
    psi = gravity_tensor[1]*gravity_tensor[1] + gravity_tensor[2]*gravity_tensor[2]
    if psi < 1e-6:
        alignment_tensor = gravity_tensor
    else:
        pitch_angle = torch.atan2(gravity_tensor[2], gravity_tensor[1])
        if torch.cos(pitch_angle) > 0.3:
            alignment_tensor = torch.tensor([0., 1., 0.], dtype=torch.float)
        else:
            alignment_tensor = gravity_tensor
    return alignment_tensor


class ScanNetSmallFramesDataset(Dataset):
    def __init__(self, root='/mars/mnt/dgx/FrameNet/scannet-small-frames/', usage='test',
                 dataset_pickle_file='./data/first10scenes_train_test_split.pkl', feat=0,
                 read_depths_from_pointcloud=True, skip_every_n_image=1, predicted_normal_subdirectory='DORN_acos_bs16_inference',
                 incomplete_depth_subdirectory='incomplete_depth_dorn_acos',
                 generative_incomplete_depth=False,
                 load_predicted_plane_mask_if_error=False,
                 normalize_normals=False,
                 generate_random_sparse_pointcloud=False):
        super(ScanNetSmallFramesDataset, self).__init__()
        # Transforms
        self.root = root
        self.to_tensor = transforms.ToTensor()

        with open(dataset_pickle_file, 'rb') as file:
            self.data_info = pickle.load(file)[usage]
        self.idx = [i for i in range(0, len(self.data_info[0]), skip_every_n_image)]
        self.data_len = len(self.idx)
        logging.info('Number of frames for the usage {0} is {1}.'.format(usage, self.data_len))

        self.intrinsics = [577.591, 318.905, 578.73, 242.684]
        xx, yy = np.meshgrid(np.array([i for i in range(640)]), np.array([i for i in range(480)]))
        self.mesh_x = cv2.resize((xx - self.intrinsics[1]) / self.intrinsics[0], (320, 240), interpolation=cv2.INTER_NEAREST)
        self.mesh_y = cv2.resize((yy - self.intrinsics[3]) / self.intrinsics[2], (320, 240), interpolation=cv2.INTER_NEAREST)
        self.feat = feat
        self.root = root
        self.read_pointcloud = read_depths_from_pointcloud
        self.generative_incomplete_depth = generative_incomplete_depth
        self.predicted_normal_subdirectory = predicted_normal_subdirectory
        self.incomplete_depth_subdirectory = incomplete_depth_subdirectory
        self.load_predicted_plane_mask_if_error = load_predicted_plane_mask_if_error
        self.normalize_normals = normalize_normals
        self.generate_random_sparse_pointcloud = generate_random_sparse_pointcloud
        fc = np.array([577.87061, 580.25851]) / 2
        cc = np.array([319.87654, 239.87603]) / 2
        self.homogeneous_coords = generate_image_homogeneous_coordinates(fc, cc, 320, 240)

    def __getitem__(self, index):
        # Get image name from the pandas df
        # NOTE: This code is highly dependent on the format of the path and filenames.
        color_info = self.data_info[0][self.idx[index]]
        orient_info = self.data_info[1][self.idx[index]]
        #orient_info_X = self.data_info[1][self.idx[index]][:-10] + 'orient-X.png'
        #orient_info_Y = self.data_info[1][self.idx[index]][:-10] + 'orient-Y.png'
        mask_info = self.data_info[2][self.idx[index]]
        gravity_info = self.data_info[1][self.idx[index]][:-10] + 'gravity.txt'
        pose_info = self.data_info[1][self.idx[index]][:-23] + \
                    '/pose/' + self.data_info[1][self.idx[index]][-23:-11] + '.pose.txt'
        scan_id = int(self.data_info[1][self.idx[index]][-31:-27])

        # Determine if this is a train or test scan.
        if scan_id < 707:
            # Train scan
            pose_info = pose_info.replace('FrameNet/scannet-small-frames', 'ScanNet/scans')
        else:
            pose_info = pose_info.replace('FrameNet/scannet-small-frames', 'ScanNet/scans_test/scans')

        predicted_normal_file = self.data_info[1][self.idx[index]]
        predicted_normal_file = predicted_normal_file.replace('scannet-small-frames', self.predicted_normal_subdirectory)
        predicted_normal_file = predicted_normal_file.replace('normal.png', 'normal_pred.png')

        depth_info = color_info.replace('color', 'depth')

        # Open image
        color_img = sio.imread(color_info)
        color_tensor = self.to_tensor(color_img)
        depth_img = sio.imread(depth_info) / 1000.0
        depth_tensor = torch.tensor(depth_img).float()
        depth_tensor = depth_tensor.view(1, depth_tensor.shape[0], depth_tensor.shape[1])

        orient_mask_tensor = sio.imread(mask_info)
        orient_mask_tensor = torch.Tensor(orient_mask_tensor/255.0)
        orient_img = sio.imread(orient_info)
        Z = -self.to_tensor(orient_img) + 0.5
        gravity_tensor = torch.tensor(np.loadtxt(gravity_info, dtype=np.float), dtype=torch.float)
        pose_tensor = torch.tensor(np.loadtxt(pose_info, dtype=np.float), dtype=torch.float)
        input_tensor = np.zeros((3, color_img.shape[0], color_img.shape[1]), dtype='float32')
        input_tensor[0:3, :, :] = color_tensor

        # A sample pose_info: '/mars/mnt/dgx/ScanNet/scans/scene0002_00//pose/frame-000600.pose.txt'
        pose_split = pose_info.split('/')
        scene_name = pose_split[-4]
        frame_id = int(pose_split[-1][6:12])

        # Creates the incomplete depth, instead of reading it.
        if self.generative_incomplete_depth:
            plane_mask_gt_filename = color_info.replace('FrameNet/scannet-small-frames', 'ScanNet/large_scale')
            plane_mask_gt_filename = plane_mask_gt_filename.replace('/frame-', '/planes_segmentation/instance_segmentation/frame-')
            plane_mask_gt_filename = plane_mask_gt_filename.replace('-color.png', '.instance_segmentation.png')

            try:
                #NOTE: The plane masks might be saved with incorrect size. This is a workaround for that.
                mask_image_gt = Image.open(plane_mask_gt_filename)
                if mask_image_gt.width == 640 and mask_image_gt.height == 640:
                    mask_image_gt = mask_image_gt.crop((0, 80, 640, 560))
                plane_mask_gt = torch.from_numpy(np.asarray(mask_image_gt.resize((320, 240), resample=Image.NEAREST)))
            except:
                if self.load_predicted_plane_mask_if_error:
                    # Get the filename for the predicted plane mask
                    plane_mask_path = '/mars/mnt/dgx/FrameNet/plane_inference'
                    plane_mask_path = os.path.join(plane_mask_path, scene_name)
                    plane_mask_filename = os.path.join(plane_mask_path, 'frame-{0:06d}-instance_segmentation.png'.format(frame_id))
                    mask_image = Image.open(plane_mask_filename)
                    if mask_image.width == 640 and mask_image.height == 640:
                        mask_image = mask_image.crop((0, 80, 640, 560))
                    mask_tensor = torch.from_numpy(np.asarray(mask_image.resize((320, 240), resample=Image.NEAREST)))
                    plane_mask_gt = mask_tensor

                    assert plane_mask_gt.shape[0] == 240
                    assert plane_mask_gt.shape[1] == 320
                else:
                    plane_mask_gt = torch.zeros((240, 320), dtype=torch.uint8)
                    logging.error('Could not load plane mask file {0}.'.format(plane_mask_gt_filename))

        klt_tracks_file = pose_info.replace('/pose/', '/klt_tracks/')

        if self.read_pointcloud:
            klt_tracks_file = klt_tracks_file.replace('.pose.txt', '-observed-pointcloud.txt')
        else:
            klt_tracks_file = klt_tracks_file.replace('.pose.txt', '-tracked-klt-features.txt')
        # Read the klt tracks and prepare the depth.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # To get rid of load empty file warning.
                klt_tracks = np.atleast_2d(np.loadtxt(klt_tracks_file, delimiter=' '))
        except Exception as ex:
            logging.error('Failed to read file {0}: {1}. Ignoring corresponding depths.'.format(klt_tracks_file, ex))
            if self.read_pointcloud:
                klt_tracks = np.zeros((0, 4))
            else:
                klt_tracks = np.zeros((0, 3))

        klt_depth_tensor = torch.zeros_like(depth_tensor)
        if self.generate_random_sparse_pointcloud:
            num_points = klt_tracks.shape[0]
            # Generate as many random points from the ground-truth.
            # Here depth tensor is of size (1, H, W)
            rows, cols = torch.nonzero(depth_tensor.squeeze(), as_tuple=True)
            if num_points >= rows.nelement():
                klt_depth_tensor = depth_tensor.clone()
            else:
                indices = np.random.permutation(rows.shape[0])[0:num_points]
                #print(rows.shape, cols.shape, depth_tensor.shape, indices.shape)
                klt_depth_tensor[0, rows[indices], cols[indices]] = depth_tensor[0, rows[indices], cols[indices]]
        else:
            coordinates = (klt_tracks[:, 1:3] / 2).astype(np.int32)
            for i in range(coordinates.shape[0]):
                row = coordinates[i, 1]
                col = coordinates[i, 0]
                if row >= 0 and row < 240 and col >= 0 and col < 320:
                    if self.read_pointcloud:
                        klt_depth_tensor[0, row, col] = klt_tracks[i, 3]
                    else:
                        klt_depth_tensor[0, row, col] = depth_tensor[0, row, col]

        # Also read the predicted normal file.
        # These files are already in the right size
        normal_img = Image.open(predicted_normal_file)
        assert normal_img.width == 320
        assert normal_img.height == 240
        normal_values = 1 - np.asarray(normal_img).astype(np.float32) / 127.5
        # Fix the shape
        normal_tensor = self.to_tensor(normal_values)

        # Load the predicted depth image.
        incomplete_depth_base = os.path.join(pose_info[0:pose_info.find('/pose/')], self.incomplete_depth_subdirectory)
        incomplete_depth_filename = os.path.join(incomplete_depth_base, 'depth-{0:06d}-incomplete.png'.format(frame_id))
        incomplete_normal_filename = os.path.join(incomplete_depth_base, 'plane_normals-{0:06d}-incomplete.png'.format(frame_id))

        try:
            incomplete_depth_tensor = load_depth_image(incomplete_depth_filename)
        except:
            incomplete_depth_tensor = torch.zeros_like(depth_tensor)

        try:
            incomplete_plane_normals = load_normal_image_framenet(incomplete_normal_filename)
        except:
            incomplete_plane_normals = Z

        alignment_tensor = compute_alignment_tensor(gravity_tensor)

        output = {'image': color_tensor, 'mask': orient_mask_tensor, 'depth': depth_tensor, 'gravity': gravity_tensor, 'color_filename': self.idx[index],
                'normal': Z, 'pose': pose_tensor, 'incomplete_depth': incomplete_depth_tensor, 'plane_normals': incomplete_plane_normals,
                'predicted_normal': normal_tensor, 'sparse_depth': klt_depth_tensor, 'aligned_direction': alignment_tensor,
                'homogeneous_coordinates': self.homogeneous_coords}
        if self.generative_incomplete_depth:
            output['plane_mask_gt'] = plane_mask_gt

        return output

    def __len__(self):
        return self.data_len


class KinectAzureDataset(Dataset):
    def __init__(self, dataset_pickle_file, usage='test', use_triangulation=True, skip_every_n_image=1):
        super(KinectAzureDataset, self).__init__()

        self.to_tensor = transforms.ToTensor()

        with open(dataset_pickle_file, 'rb') as file:
            self.data_info = pickle.load(file)[usage]
        self.idx = [i for i in range(0, len(self.data_info[0]), skip_every_n_image)]
        self.data_len = len(self.idx)
        logging.info('Number of frames for the usage {0} is {1}.'.format(usage, self.data_len))

        # This compensates for both cropping and scaling.
        self.fc = np.array([202.9953, 202.9540])
        self.cc = np.array([159.7645, 122.0951])
        self.use_triangulation = use_triangulation
        self.homogeneous_coords = generate_image_homogeneous_coordinates(self.fc, self.cc, 320, 240)

    def __getitem__(self, index):
        color_info = self.data_info[0][self.idx[index]]
        depth_info = self.data_info[1][self.idx[index]]
        gravity_info = self.data_info[2][self.idx[index]]

        # RGB image, converted to 0.-1. by self.to_sensor.
        color_img = Image.open(color_info)
        assert color_img.width == 640 and color_img.height == 480
        color_img = color_img.resize((320, 240), resample=Image.BILINEAR)
        color_tensor = self.to_tensor(color_img)

        depth_img = Image.open(depth_info).convert('F')  # Convert to float32
        depth_img = depth_img.resize((320, 240), resample=Image.NEAREST)
        depth_tensor = torch.Tensor(np.array(depth_img)) / 1000.0
        depth_tensor = depth_tensor[None, ...]

        # Get the path to the klt_tracks
        if self.use_triangulation:
            klt_tracks_directory = color_info.replace('/color/', '/klt_tracks_triangulation/').replace('.png', '')
        else:
            klt_tracks_directory = color_info.replace('/color/', '/klt_tracks/').replace('.png', '')
        index = klt_tracks_directory.rfind('/')
        assert index != -1
        #frame_id = int(klt_tracks_directory[index+1:])
        frame_id = int(klt_tracks_directory[index+7:])
        if frame_id < 0:
            print(klt_tracks_directory[index+7:])
            exit(0)
        klt_tracks_directory = klt_tracks_directory[0:index]

        klt_tracks_file = os.path.join(klt_tracks_directory, 'frame-{0:06d}-observed-pointcloud.txt'.format(frame_id))

        klt_tracks_file = klt_tracks_file.replace('.pose.txt', '-observed-pointcloud.txt')
        # Read the klt tracks and prepare the depth.
        try:
            #with warnings.catch_warnings():
            #    warnings.simplefilter("ignore")
                # To get rid of load empty file warning.
            klt_tracks = np.atleast_2d(np.loadtxt(klt_tracks_file, delimiter=' '))
        except Exception as ex:
            logging.error('Failed to read file {0}: {1}. Ignoring corresponding depths.'.format(klt_tracks_file, ex))
            klt_tracks = np.zeros((0, 5))

        #assert klt_tracks.shape[1] == 5
        klt_depth_tensor = torch.zeros_like(depth_tensor)
        if klt_tracks.shape[0] > 0:
            # Different from the Scannet, here we save for 320x240 images.
            #coordinates = np.atleast_2d(klt_tracks[:, 1:3]).astype(np.int32)
            for i in range(klt_tracks.shape[0]):
                #row = coordinates[i, 1]
                #col = coordinates[i, 0]
                u = klt_tracks[i, 1] / klt_tracks[i, 3]
                v = klt_tracks[i, 2] / klt_tracks[i, 3]
                px = self.fc[0] * u + self.cc[0]
                py = self.fc[1] * v + self.cc[1]
                col = int(px) #int(points[i, 1] / 2)
                row = int(py) #int(points[i, 2] / 2)
                if row >= 0 and row < 240 and col >= 0 and col < 320:
                    klt_depth_tensor[0, row, col] = klt_tracks[i, 3]

        if torch.sum(klt_depth_tensor) == 0:
            logging.error('Empty sparse depth for {0}.'.format(klt_tracks_file))

        gravity_tensor = torch.tensor(np.loadtxt(gravity_info, dtype=np.float), dtype=torch.float)
        gravity_tensor[1] = -gravity_tensor[1]
        gravity_tensor[2] = -gravity_tensor[2]
        psi = gravity_tensor[1] * gravity_tensor[1] + gravity_tensor[2] * gravity_tensor[2]
        if psi < 1e-4:
            alignment_tensor = torch.tensor([0., 1., 0.], dtype=torch.float)
        else:
            pitch_angle = torch.atan2(gravity_tensor[2], gravity_tensor[1])
            if torch.cos(pitch_angle) > 0.707:
                alignment_tensor = torch.tensor([0., 1., 0.], dtype=torch.float)
            else:
                alignment_tensor = torch.tensor([0., torch.cos(pitch_angle), torch.sin(pitch_angle)], dtype=torch.float)

        return {'image': color_tensor, 'depth': depth_tensor, 'gravity': gravity_tensor,
                'sparse_depth': klt_depth_tensor,
                'aligned_direction': alignment_tensor, 'homogeneous_coordinates': self.homogeneous_coords}

    def __len__(self):
        return self.data_len


class NYUDataset(Dataset):
    def __init__(self, root='/mars/mnt/dgx/nyud_full/', usage='test',
                 dataset_pickle_file='./data/nyud_standard_split.pkl',
                 read_depths_from_pointcloud=True, skip_every_n_image=1,
                 incomplete_depth_subdirectory='incomplete_depth',
                 generative_incomplete_depth=False,
                 load_predicted_plane_mask_if_error=False,
                 normalize_normals=False,
                 use_inpainted_depth=False,
                 max_depth=7.0, min_depth=0.1):
        super(NYUDataset, self).__init__()
        # Transforms
        self.root = root
        self.to_tensor = transforms.ToTensor()

        with open(dataset_pickle_file, 'rb') as file:
            self.data_info = pickle.load(file)[usage]
        self.idx = [i for i in range(0, len(self.data_info), skip_every_n_image)]
        self.data_len = len(self.idx)
        logging.info('Number of frames for the usage {0} is {1}.'.format(usage, self.data_len))

        self.root = root
        self.use_inpainted_depth = use_inpainted_depth
        self.max_depth = max_depth
        self.min_depth = min_depth

    def __getitem__(self, index):
        seq, name = self.data_info[self.idx[index]]
        color_info = os.path.join(self.root, seq, 'color', 'color_%s.png' % name)
        depth_info = color_info.replace('color', 'depth')
        sparse_points_path = os.path.join(self.root, seq, 'klt_tracks', 'frame-%s-observed-pointcloud.txt' % name)
        incomplete_depth_filename = os.path.join(self.root, seq, 'incomplete_depth', 'depth-%s-incomplete.png' % name)
        incomplete_normal_filename = os.path.join(self.root, seq, 'incomplete_depth', 'plane_normals-%s-incomplete.png' % name)

        # normal prediction
        predicted_normal_file = color_info.replace('color', 'normal_pred')
        normal_img = Image.open(os.path.join(self.root, seq, 'normal_pred', predicted_normal_file))
        assert normal_img.width == 320
        assert normal_img.height == 240
        normal_values = 1 - np.asarray(normal_img).astype(np.float32) / 127.5
        normal_tensor = -self.to_tensor(normal_values) + 0.5

        # Open image
        color_img = Image.open(color_info)
        color_img = color_img.resize((320, 240), resample=Image.BILINEAR)
        color_tensor = self.to_tensor(color_img)
        if self.use_inpainted_depth:
            depth_info = depth_info.replace('/depth/', '/depth_inpainted/')
        depth_img = Image.open(depth_info).convert('F')  # Convert to float32
        depth_img = depth_img.resize((320, 240), resample=Image.NEAREST)
        depth_tensor = torch.Tensor(np.array(depth_img)) / 5000.0
        depth_tensor = depth_tensor.view(1, depth_tensor.shape[0], depth_tensor.shape[1])
        depth_tensor[depth_tensor < self.min_depth] = 0
        depth_tensor[depth_tensor > self.max_depth] = 0

        # Read point clouds
        klt_depth_tensor = torch.zeros_like(depth_tensor)
        if os.path.isfile(sparse_points_path):
            with open(sparse_points_path, 'r') as point_file:
                for line in point_file:
                    _, x, y, d = line.split(' ')
                    # Note that the tracks have been created for 640x480 image, so the coordinates should be divided by 2.
                    row, col = int(float(y) / 2), int(float(x) / 2)
                    if 0 <= row < 240 and 0 <= col < 320:
                        klt_depth_tensor[0, row, col] = float(d)

        try:
            incomplete_depth_tensor = load_depth_image(incomplete_depth_filename)
        except:
            #logging.error('Could not read {0}.'.format(incomplete_depth_filename))
            incomplete_depth_tensor = klt_depth_tensor
        try:
            incomplete_plane_normals = load_normal_image_framenet(incomplete_normal_filename)
        except:
            #logging.error('Could not read {0}.'.format(incomplete_normal_filename))
            incomplete_plane_normals = normal_tensor

        incomplete_plane_normals = torch.zeros_like(color_tensor)

        output = {'image': color_tensor, 'depth': depth_tensor, 'incomplete_depth': incomplete_depth_tensor,
                  'plane_normals': incomplete_plane_normals,
                  'predicted_normal': normal_tensor, 'sparse_depth': klt_depth_tensor}

        return output

    def __len__(self):
        return self.data_len


class DemoDataset(Dataset):
    def __init__(self, root):
        super(DemoDataset, self).__init__()

        self.to_tensor = transforms.ToTensor()
        self.root = root

        self.color_files = fnmatch.filter(os.listdir(os.path.join(self.root, 'color')), '*.png')
        self.data_len = len(self.color_files)
        logging.info('Number of frames for the usage {0} is {1}.'.format('test', self.data_len))

        # This compensates for both cropping and scaling.
        self.fc = np.array([202.9953, 202.9540])
        self.cc = np.array([159.7645, 122.0951])
        self.homogeneous_coords = generate_image_homogeneous_coordinates(self.fc, self.cc,
                                                                         320, 240)

    def __getitem__(self, index):
        color_info = os.path.join(self.root, 'color', self.color_files[index])
        gravity_info = color_info.replace('color', 'gravity').replace('png', 'txt')
        depth_sparse_info = color_info.replace('color', 'depth_sparse').replace('png', 'txt')
        depth_incomplete_info = color_info.replace('color', 'depth_incomplete')

        # RGB image, converted to 0.-1. by self.to_sensor.
        color_img = Image.open(color_info)
        assert color_img.width == 640 and color_img.height == 480
        color_img = color_img.resize((320, 240), resample=Image.BILINEAR)
        color_tensor = self.to_tensor(color_img)
        gravity_tensor = torch.tensor(np.loadtxt(gravity_info, dtype=np.float), dtype=torch.float)
        gravity_tensor[1] = -gravity_tensor[1]
        gravity_tensor[2] = -gravity_tensor[2]
        psi = gravity_tensor[1]*gravity_tensor[1] + gravity_tensor[2]*gravity_tensor[2]
        if psi < 1e-4:
            alignment_tensor = torch.tensor([0., 1., 0.], dtype=torch.float)
        else:
            pitch_angle = torch.atan2(gravity_tensor[2], gravity_tensor[1])
            if torch.cos(pitch_angle) > 0.707:
                alignment_tensor = torch.tensor([0., 1., 0.], dtype=torch.float)
            else:
                alignment_tensor = torch.tensor([0., torch.cos(pitch_angle), torch.sin(pitch_angle)], dtype=torch.float)

        # Read the klt tracks and prepare the depth.
        try:
            # with warnings.catch_warnings():
            #    warnings.simplefilter("ignore")
            # To get rid of load empty file warning.
            klt_tracks = np.atleast_2d(np.loadtxt(depth_sparse_info, delimiter=' '))
        except Exception as ex:
            logging.error('Failed to read file {0}: {1}. Ignoring corresponding depths.'.format(depth_sparse_info, ex))
            klt_tracks = np.zeros((0, 5))

        # assert klt_tracks.shape[1] == 5
        klt_depth_tensor = torch.zeros_like(color_tensor[0:1, :, :])
        if klt_tracks.shape[0] > 0:
            # Different from the Scannet, here we save for 320x240 images.
            # coordinates = np.atleast_2d(klt_tracks[:, 1:3]).astype(np.int32)
            for i in range(klt_tracks.shape[0]):
                # row = coordinates[i, 1]
                # col = coordinates[i, 0]
                u = klt_tracks[i, 1] / klt_tracks[i, 3]
                v = klt_tracks[i, 2] / klt_tracks[i, 3]
                px = self.fc[0] * u + self.cc[0]
                py = self.fc[1] * v + self.cc[1]
                col = int(px)  # int(points[i, 1] / 2)
                row = int(py)  # int(points[i, 2] / 2)
                if row >= 0 and row < 240 and col >= 0 and col < 320:
                    klt_depth_tensor[0, row, col] = klt_tracks[i, 3]

        if torch.sum(klt_depth_tensor) == 0:
            logging.error('Empty sparse depth for {0}.'.format(depth_sparse_info))

        return {'image': color_tensor,
                'color_filename': self.color_files[index],
                'sparse_depth': klt_depth_tensor,
                'gravity': gravity_tensor,
                'homogeneous_coordinates': self.homogeneous_coords,
                'aligned_direction': alignment_tensor}

    def __len__(self):
        return self.data_len
