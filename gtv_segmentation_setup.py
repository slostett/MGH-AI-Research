# [[file:gtv_segmentation.org::py-startup-0][py-startup-0]]
# %% Standard imports

import os
# os.environ["NVIDIA_VISIBLE_DEVICES"]="2"
# os.environ["CUDA_VISIBLE_DEVICES"]="2"

FLAG_PLOT_ENV = os.getenv('FLAG_PLOT')
FLAG_PLOT = FLAG_PLOT_ENV is None or FLAG_PLOT_ENV == 1

# General
import sys
import re
import matplotlib as mpl
if FLAG_PLOT:
    mpl.use('Qt5Agg')
    import matplotlib.pyplot as plt
    import PyQt5.QtWidgets as _QtWidgets
    qApp = _QtWidgets.QApplication([" "])
PTOOLS_PARENT_DIR = os.path.join(os.getenv('PYTHON_TOOLS'), '..')
for lib_dir in [PTOOLS_PARENT_DIR, ]:
    if not lib_dir in sys.path:
        sys.path.append(lib_dir)
# py-startup-0 ends here

# [[file:gtv_segmentation.org::py-startup-1][py-startup-1]]
# %% Imports

# General
import re
import random
import copy
import glob
import gc
import importlib
import functools
import datetime
import dill as pickle
import inspect
import logging
import argparse
import shutil
import uuid
import json
import tqdm
import textwrap
import warnings
import pprint
import collections as coll
import numpy as np
import scipy.interpolate as spint
import scipy.io as spio
import scipy.ndimage as spnd
import scipy.optimize as spopt
import scipy.signal as spsig
import skimage.morphology as skmorph
import skimage.measure as skmeas
import sklearn.model_selection as sklmod
import pydicom as pdcm
import cv2
import dicom_contour.contour as pdcmc
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
import tensorflow.keras as keras
if tf.__version__ == '1.15.5':
    # GPU setup
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
if FLAG_PLOT:
    import matplotlib.pyplot as plt
    plt.ion()

# %% Tools

# Python tools
import python_tools.analysis as pta
import python_tools.dict as ptdict
import python_tools.elastix as ptelx
import python_tools.export as ptexp
import python_tools.geometry as ptgeom
import python_tools.iotools as ptio
import python_tools.misc as ptm
import python_tools.numerical as ptnum
import python_tools.oper as ptop
import python_tools.deep_learning as ptdl
if FLAG_PLOT:
    import python_tools.idt as idt

# %% Aliases

ddir = lambda m, s, f0=False, f1=False: ptm.search_field(
    m, s, flags=re.I, flag_recursive=f0, flag_recurse_module=f1)

flag_export = False

# %% Folders

data_fold_base = re.sub('code$', 'data', os.path.split(os.getcwd())[0])
data_fold = os.path.join(data_fold_base, os.path.split(os.getcwd())[-1])
res_fold = os.path.join(data_fold, 'results')
fig_fold = os.path.join(data_fold, 'figures')
# if not os.path.isdir(res_fold):
#     os.makedirs(res_fold)

# %% Logging

log_level = logging.INFO

logger_fmt = logging.Formatter(fmt='%(asctime)-15s %(message)s')
logger = logging.getLogger('gtv_segmentation')
logger.setLevel(log_level)

if not FLAG_PLOT:
    # Batch mode, enable logging to file
    fname = ('log_' + datetime.datetime.now().strftime('%y%m%d_%H%M') + '_' +
             str(uuid.uuid4()))
    fh = logging.FileHandler(fname)
    fh.setLevel(log_level)
    fh.setFormatter(logger_fmt)
    # logger.addHandler(fh)

logger.propagate = False
# py-startup-1 ends here

# [[file:gtv_segmentation.org::pydef-helpers][pydef-helpers]]
# %% Data folder

dfold = 'f:/data_all/190823-TIP-sarcoma/TIPsarcoma'
dfold = os.path.join(data_fold, '../190823-TIP-sarcoma/TIPsarcoma_clean')
if os.getenv('HOSTNAME') == 'meson' or (os.getenv('HOSTNAME') and
                                        'node' in os.getenv('HOSTNAME')):
    dfold = os.path.expanduser(
        '~/data/190823-TIP-sarcoma/TIPsarcoma_gcmi_2')
else:
    dfold = os.path.join(data_fold, '../190823-TIP-sarcoma/TIPsarcoma_gcmi_2')
dfold = os.path.join(data_fold, '../190823-TIP-sarcoma/TIPsarcoma_out_230207')
dfold_hnscc = os.path.join(data_fold, '../200515-TCIA-HNSCC/HNSCC')

def dcm_print(dcm):
    return '''FileDataset:
    Modality: {}
    '''.format(dcm.Modality)

pdcm.dataset.FileDataset.__str__ = dcm_print

p2_fold = os.path.join(dfold, 'Patient 2')
# p2_data = PatientData(p2_fold)

# %% Miscellaneous

def get_num_classes_cons_combine(flag_combine):
    if not isinstance(flag_combine, str):
        return None
    res = re.search('cons([0-9]+)c', flag_combine)
    if res:
        return int(res.groups()[0])
    return None

# %% List datasets for single patient

class ImageData:
    def __init__(self, data_path):
        self._data_path = data_path
        self._fname_pat = os.path.join(self._data_path, '{}.dcm')
        self._slice_order = [[float(s[1]), s[0]]
                             for s in pdcmc.slice_order(self._data_path)]
        if not os.path.isfile(os.path.join(self._data_path,
                                           self._slice_order[0][1] + '.dcm')):
            file_list = [re.sub('\.dcm$', '', os.path.split(d)[1])
                         for d in glob.glob(
                                 os.path.join(self._data_path, '*.dcm'))]
            if len(self._slice_order) != len(file_list):
                if len(self._slice_order) == len([d for d in file_list
                                                  if '.' not in d]):
                    file_list = [d for d in file_list if '.' not in d]
                else:
                    raise RuntimeError('Number of dicom files does not ' +
                                       'match dataset.')
            self._sl_file = {}
            for f in file_list:
                dcm_t = pdcm.read_file(
                    os.path.join(self._data_path, f + '.dcm'))
                sidx = [s[1] for s in self._slice_order].index(
                    dcm_t.SOPInstanceUID)
                self._sl_file[self._slice_order[sidx][1]] = re.sub(
                    r'\.dcm', '', os.path.split(f)[1])
        else:
            self._sl_file = {s[1]: s[1] for s in self._slice_order}
        self._dcm_img = pdcm.read_file(os.path.join(
            self._data_path, self._sl_file[self._slice_order[0][1]] + '.dcm'))
        self._data_type = self._dcm_img.Modality
        # if self._data_type.upper() == 'CT' and \
        #    re.search('sf_T[12]', self._dcm_img.SeriesDescription):
        #     self._data_type = 'MR'
        self._shape = [self._dcm_img.Columns, self._dcm_img.Rows,
                       len(self._slice_order)]
        # Note: slice spacing and slice thickness may be different
        slice_spacing = np.diff(sorted([s[0]
                                        for s in self._slice_order])[:2])[0]
        self._spacing = [float(x) for x in self._dcm_img.PixelSpacing] + \
            [slice_spacing,  ]
        self._origin = [float(x) for x in self._dcm_img.ImagePositionPatient]
        self._orientation = [float(x) for x in
                             self._dcm_img.ImageOrientationPatient]
        self._position = self._dcm_img.PatientPosition
        self._pos_mat = np.diag([1, -1, -1])
        X = self._orientation[:3]
        Y = self._orientation[3:]
        d = self._spacing
        s = self._origin
        n = self._shape
        self._center = self._pos_mat @ (
            np.array(s) + np.array(d) * np.array(n) / 2)[:, np.newaxis][:, 0]
        self._loc = np.array(
            [[X[0] * d[0], Y[0] * d[1], 0, s[0]],
             [X[1] * d[0], Y[1] * d[1], 0, s[1]],
             [X[2] * d[0], Y[2] * d[1], 0, s[2]],
             [0, 0, 0, 1]])
    @property
    def data_path(self):
        return self._data_path
    @property
    def data_type(self):
        return self._data_type
    @property
    def modality(self):
        return self._data_type.lower()
    def get_image(self):
        return np.stack([
            ptio.DataFileDicom().load(
                os.path.join(self._data_path, self._sl_file[s[1]] + '.dcm'))
            for s in self._slice_order])
    def has_id(self, id):
        return id in [os.path.split(self._sl_file[s[1]])[1]
                      for s in self._slice_order]
    def get_pos_phys(self, xi, yi, zi):
        p = self._loc @ np.array([[xi], [yi], [0], [1]])
        p[2, 0] = self._origin[2] + zi * self._spacing[2]
        return p[:3, :]

class SegData:
    mod_map = {'CT-MRI': 'CT-MR', 'CT--MR': 'CT-MR',
               'CT-MR-': 'CT-MR', 'CT-MR-PET-': 'CT-MR-PET',
               'CT-': 'CT'}
    reader_map = {'YLC': 1, 'E': 1}
    def set_mod(self, mod_in):
        mod = re.sub('^-+', '',
                     re.sub('-+$', '',
                            re.sub('updated', '', mod_in, re.I)))
        if mod.upper() in SegData.mod_map:
            mod = SegData.mod_map[mod.upper()]
        self._modality = mod.lower()
    def set_reader(self, reader):
        if isinstance(reader, str) and reader.upper() in SegData.reader_map:
            reader = SegData.reader_map[reader.upper()]
        self._reader = reader
    def __init__(self, data_path, contour_idx=0, flag_disable_parsing=False,
                 flag_single_contour=False, force_modality=None,
                 flag_fixed_pattern=False):
        self._data_path = data_path
        self._reader = 0
        self._trial = 0
        self._contour_type = 'gtv'
        flag_found = False
        self._num_threads = 12
        if flag_fixed_pattern:
            fold_pat = r'(.?TV).Reader.([0-9-]+)\.([^.]+)\.Trial\.([0-9]+)'
            def parser_gtv_ctv(self, re_res):
                self._contour_type = re_res.groups()[0].lower()
                self.set_reader(int(re_res.groups()[1]))
                self.set_mod(re_res.groups()[2])
                self._trial = int(re_res.groups()[3])
            pat_parser_list = [[fold_pat, parser_gtv_ctv], ]
        else:
            # Parsers
            fold_pat_ctv_r2 = (r'CTV.[Rr]e[aq]der.([0-9]).from.([^.]+)' +
                               r'.(?:GTV.)?(?:on.)?([^._]+)')
            def parser_ctv_r2(self, re_res):
                r_base = -10 if re_res.groups()[1] == 'consensus' \
                    else -20
                self.set_reader(r_base - int(re_res.groups()[0]))
                self._trial = 1
                mod_str = re_res.groups()[2].lower()
                if mod_str == 'gtv':
                    mod_str = 'ct'
                self.set_mod(re.sub(r'\.', '-', mod_str))
            fold_pat_cons = r'[Cc]on[sc]ensus\.GTV\.on\.([A-Za-z-.]+)'
            def parser_cons(self, re_res):
                self.set_reader(-1)
                self._trial = 1
                self.set_mod(re.sub(r'\.', '-', re_res.groups()[0].lower()))
            fold_pat_cons_1 = r'[Cc]on[sc]ensus\.GTV'
            def parser_cons_1(self, re_res):
                self.set_reader(-1)
                self._trial = 1
                self.set_mod('ct')
            fold_pat_ctv = r'CTV.Reader\.([0-9]+)\.from\.([^.]+)\.(.+)'
            def parser_ctv(self, re_res):
                self.set_reader(int(re_res.groups()[0]))
                self._trial = 1
                self.set_mod('CTV-{}-{}'.format(
                    re_res.groups()[1].lower(),
                    re.sub(r'\.', '-', re_res.groups()[2].lower())))
            fold_pat_r2 = (r'[CG][TV][TV]?.+Re[aqd]?[sd]e[rt]\.+a?(\d)\.?' +
                           r'(.*)\.T.?r[ia]+l\.?(\d)')
            def parser_r2(self, re_res):
                self.set_reader(int(re_res.groups()[0]))
                self._trial = int(re_res.groups()[2])
                self.set_mod(re.sub('\.', '-', re_res.groups()[1].lower()))
            fold_pat_r2a = (r'[CG][TV][TV]?.+Re[aqd]?[sd]+e[rt]\.?(\d)\.?' +
                            r'(.*)\.T.?r[ia]+l\.?(\d)')
            def parser_r2a(self, re_res):
                self.set_reader(int(re_res.groups()[0]))
                self._trial = int(re_res.groups()[2])
                self.set_mod(re.sub(r'\.', '-', re_res.groups()[1].lower()))
            fold_pat_r2b = r'GTV\.+(.*)\.Re[aq][sd]er\.(\d)\.Trial\.(\d)'
            def parser_r2b(self, re_res):
                self.set_reader(int(re_res.groups()[1]))
                self._trial = int(re_res.groups()[2])
                self.set_mod(re.sub(r'\.', '-',
                                        re.sub(r'on\.', '',
                                               re_res.groups()[0].lower())))
            fold_pat_r2c = r'GTV\.+Re[aq]der\.(\d)\.(.*)\.(\d)'
            def parser_r2c(self, re_res):
                self.set_reader(int(re_res.groups()[0]))
                self._trial = int(re_res.groups()[2])
                self.set_mod(re.sub(r'\.', '-',
                                        re.sub(r'on\.', '',
                                               re_res.groups()[1].lower())))
            fold_pat_r1 = r'GTV.?\.(?:on\.)?(.*).Reader.([^_.]+)'
            def parser_r1(self, re_res):
                self.set_mod(re_res.groups()[0].lower())
                try:
                    self.set_reader(int(re_res.groups()[1]))
                except ValueError:
                    self.set_reader(re_res.groups()[1])
                self._trial = 1
            fold_pat_r1b = r'([^_]+)\.Trial.(.)\.Reader.([a-zA-Z0-9]+)'
            def parser_r1b(self, re_res):
                self.set_mod(re_res.groups()[0].lower())
                try:
                    self.set_reader(int(re_res.groups()[2]))
                except ValueError:
                    self.set_reader(re_res.groups()[2])
                self._trial = int(re_res.groups()[1])
            fold_pat_r1c = r'GTV.?\.(?:on\.)?(.*).Reader.(\w+).Trial.(\d)'
            fold_pat_r1c = r'GTV.?\.(?:on\.)?(.*).Reader.?(\w+).Trial\.*(\d)'
            def parser_r1c(self, re_res):
                self.set_mod(re_res.groups()[0].lower())
                try:
                    self.set_reader(int(re_res.groups()[1]))
                except ValueError:
                    self.set_reader(re_res.groups()[1])
                self._trial = int(re_res.groups()[2])
            fold_pat_r1d = (r'([^_.]+)\.Re[aq]der[e]?.([a-zA-Z0-9]+)\.' +
                            r'Tr[ia][ia]l\.+(.)')
            def parser_r1d(self, re_res):
                self.set_mod(re_res.groups()[0].lower())
                try:
                    self.set_reader(int(re_res.groups()[1]))
                except ValueError:
                    self.set_reader(re_res.groups()[1])
                self._trial = int(re_res.groups()[2])
            fold_pat_s = r'(\d)-RTstruct(.*)-'
            def parser_s(self, re_res):
                self.set_reader(int(re_res.groups()[0]))
                self.set_mod(re_res.groups()[1].lower())
                if re.search('T1', self._modality):
                    self._modality = 'mri'
                if re.search('T2', self._modality):
                    self._modality = 'mri-t2'
                self._trial = contour_idx + 1
            pat_parser_list = [[fold_pat_ctv_r2, parser_ctv_r2],
                               [fold_pat_cons, parser_cons],
                               [fold_pat_ctv, parser_ctv],
                               [fold_pat_r2, parser_r2],
                               [fold_pat_r2a, parser_r2a],
                               [fold_pat_r2b, parser_r2b],
                               [fold_pat_r2c, parser_r2c],
                               [fold_pat_r1c, parser_r1c],
                               [fold_pat_r1d, parser_r1d],
                               [fold_pat_r1, parser_r1],
                               [fold_pat_r1b, parser_r1b],
                               [fold_pat_cons_1, parser_cons_1],
                               [fold_pat_s, parser_s]]
        f_fallback = None
        for pat_f, parser_f in pat_parser_list:
            re_path = re.search(pat_f, self._data_path, re.I)
            if re_path and not flag_found:
                if all(re_path.groups()):
                    flag_found = True
                    parser_f(self, re_path)
                    break
                else:
                    f_fallback = [parser_f, re_path]
        if not flag_found and not flag_disable_parsing:
            if f_fallback is not None:
                f_fallback[0](self, f_fallback[1])
            raise RuntimeError('Malformed filename.')
        if force_modality is not None:
            self.set_mod(force_modality)
        self._data_fname = glob.glob(os.path.join(data_path, '*.dcm'))[0]
        # Read DICOM contour file
        self._dcm_data = pdcm.read_file(self._data_fname)
        # Get contours
        if flag_disable_parsing:
            self._reader = pdcmc.get_roi_names(
                self._dcm_data)[contour_idx].lower()
        self._num_contours = len(self._dcm_data.ROIContourSequence)
        if self._num_contours > 1 and flag_single_contour:
            if self._reader == -1:
                # Find consensus by name
                contour_idx = np.argmin([
                    ptdict._Levenshtein.distance('consensus gtv', t.lower())
                    for t in pdcmc.get_roi_names(self._dcm_data)])
            elif isinstance(flag_single_contour, list):
                # Find name in list
                dist_all = np.array([[
                    ptdict._Levenshtein.distance(c_name.lower(), t.lower())
                    for t in pdcmc.get_roi_names(self._dcm_data)]
                                     for c_name in flag_single_contour])
                contour_idx = np.argmin(dist_all.min(axis=0))
            elif contour_idx is None:
                contour_idx = np.argmax([
                    np.sum([np.min(np.var(np.reshape(
                        [float(c) for c in cc.ContourData], [-1, 3])[:, :2],
                                          axis=0))
                            for cc in
                            self._dcm_data.ROIContourSequence[cidx].get(
                                'ContourSequence', [])])
                    for cidx in range(self._num_contours)])
        self._contours = list(
            self._dcm_data.ROIContourSequence[contour_idx].get(
                'ContourSequence', []))
        self._contours = [c for c in self._contours
                          if hasattr(c, 'ContourImageSequence')]
        self._img_ids = [
            contour.ContourImageSequence[0].ReferencedSOPInstanceUID
            for contour in self._contours]
    @property
    def data_path(self):
        return self._data_path
    @property
    def reader(self):
        return self._reader
    @property
    def trial(self):
        return self._trial
    @property
    def modality(self):
        return self._modality
    @property
    def contour_type(self):
        return self._contour_type
    @property
    def image_path(self):
        return self._img_path
    def get_image(self):
        return self._img_data.get_image()
    def get_contours_pix(self, dcm_img):
        output = []
        for contour in self._contours:
            output.append(self.get_contours_pix_single(contour, dcm_img))
        return output
    def get_contours_pix_single(self, contour, dcm_img):
        # Get location
        contour_coord = contour.ContourData
        z = float(contour_coord[2])
        # x, y, z coordinates of the contour in mm
        coord = []
        for i in range(0, len(contour_coord), 3):
            coord.append((float(contour_coord[i]),
                          float(contour_coord[i + 1]),
                          float(contour_coord[i + 2])))
        # physical distance between the center of each pixel
        x_spacing, y_spacing, z_spacing = dcm_img._spacing

        # orientation
        sc_x = dcm_img._orientation[0]
        sc_y = dcm_img._orientation[4]

        # this is the center of the upper left voxel
        origin_x, origin_y, origin_z = dcm_img._origin

        # y, x is how it's mapped
        pixel_coords = [(np.ceil((x - origin_x) / x_spacing * sc_x),
                         np.ceil((y - origin_y) / y_spacing * sc_y))
                        for x, y, z in coord]
        return {'z_phys': z, 'xy_idx': np.stack(pixel_coords)}

    def get_mask(self, dcm_img, flag_single_thread=False):
        slice_list = np.array([s[0] for s in dcm_img._slice_order])
        mask_list = []
        z_list = []
        shape = dcm_img._shape
        if flag_single_thread:
            contours_pix = self.get_contours_pix(dcm_img)
        else:
            contours_pix = ptnum.apply_multithread(
                lambda c: self.get_contours_pix_single(c, dcm_img),
                self._contours, num_threads=self._num_threads)
        mask = np.zeros([len(slice_list), dcm_img._shape[1],
                         dcm_img._shape[0]], dtype=np.uint8)
        if len(contours_pix) == 0:
            return mask
        # Version 1: single-threaded
        # mxy = np.meshgrid(np.arange(dcm_img._shape[0]),
        #                   np.arange(dcm_img._shape[1]), indexing='ij')
        for contour in contours_pix:
            z_list.append(contour['z_phys'])
            pixel_coords = contour['xy_idx']
            mask_list.append(ptgeom.rasterize_contour(
                pixel_coords.T, 'cv2', dcm_img._shape[1], dcm_img._shape[0]))
            # p_path = mpl.path.Path(np.stack(pixel_coords))
            # mask_list.append(p_path.contains_points(
            #     np.stack(mxy).transpose([2, 1, 0]).reshape([-1, 2])).reshape(
            #         [dcm_img._shape[1], dcm_img._shape[0]]))
        # # Version 2: multi-threaded
        # for contour in contours_pix:
        #     z_list.append(contour['z_phys'])
        # def get_mask_s(contour):
        #     pixel_coords = contour['xy_idx']
        #     p_path = mpl.path.Path(np.stack(pixel_coords))
        #     mask_list.append(p_path.contains_points(
        #         np.stack(mxy).transpose([2, 1, 0]).reshape([-1, 2])).reshape(
        #             [dcm_img._shape[1], dcm_img._shape[0]]))
        # mask_list = ptnum.apply_multithread(get_mask_s, contours_pix,
        #                                     num_threads=self._num_threads)

        si_mask_used = []
        slice_thickness = float(dcm_img._dcm_img.SliceThickness)
        for si, sl in enumerate(slice_list):
            sl_dist = np.abs(sl - np.array(z_list))
            sl_dist_min_idx_list = np.nonzero(sl_dist == np.min(sl_dist))[0]
            if sl_dist[sl_dist_min_idx_list[0]] < slice_thickness / 2:
                for sl_dist_min_idx in sl_dist_min_idx_list:
                    si_mask = sl_dist_min_idx
                    mask[si] = np.logical_or(mask[si], mask_list[si_mask])
                    if si_mask not in si_mask_used:
                        si_mask_used.append(si_mask)
        return mask

class RegData:
    def __init__(self, data_path):
        self._data_path = data_path
        # Read DICOM contour file
        self._data_fname = glob.glob(os.path.join(data_path, '*.dcm'))[0]
        self._dcm_data = pdcm.read_file(self._data_fname)
        self._xform_tag = pdcm.tag.TupleTag([0x0013, 0x2050])
        self._reg_seq_list = self._dcm_data[self._xform_tag]
        self._reg_list = []
        for reg_seq in self._reg_seq_list:
            reg_list_seq = []
            for seq in reg_seq.RegistrationSequence:
                img_uid = seq.ReferencedImageSequence[0].ReferencedSOPClassUID
                img_seq = [s.ReferencedSOPInstanceUID
                           for s in seq.ReferencedImageSequence]
                matrix = np.array(
                    seq.MatrixRegistrationSequence[0].MatrixSequence[0].\
                    FrameOfReferenceTransformationMatrix).reshape([4, 4])
                reg_list_seq.append({'img_uid': img_uid, 'img_seq': img_seq,
                                     'matrix': matrix})
            self._reg_list.append(reg_list_seq)

class PatientData:
    def __init__(self, data_path):
        """Constructor

        Parameters
        ----------
        data_path : str
            Base folder.
        """
        self._base_fold = data_path
        self._patient_idx = int(re.search('Patient (\d+)',
                                          self._base_fold).groups()[0])
        self._seg_data = []
        print(self._patient_idx)

        fold_list = glob.glob(os.path.join(self._base_fold, '**', '*/'))
        for fold in fold_list:
            data_type_re = re.search(
                'p{}_TIPsarcoma_([^_]+)'.format(self._patient_idx), fold)
            if data_type_re:
                data_type = data_type_re.groups()[0]
            else:
                raise ValueError('Could not parse data type ({})'.format(fold))
            if data_type == 'RTst':
                self._seg_data.append(SegData(fold))

# %% Load all images

def load_all(pidx, flag_force_load=False, **kwargs):
    p_cache_fold = os.path.join(data_fold, 'p_cache')
    if not os.path.isdir(p_cache_fold):
        os.makedirs(p_cache_fold)
    p_cache_fname = os.path.join(p_cache_fold, pidx + '_{}')
    load_params = ptdict.merge(ptm.extract_defaults(do_load_all), kwargs)
    res_cache = ptio.ResultsCache(p_cache_fname, load_params)
    if flag_force_load:
        res_cache.clear()
    return res_cache.run('data_p', do_load_all, os.path.join(dfold, pidx),
                         **kwargs)


def do_load_all(p_fold, flag_compute_masks=True, modality_list=None,
                seg_data_args=None, contour_idx_name=None,
                flag_single_contour=True, flag_flat_fold=False,
                reader_list=None, contour_type=None,
                modality_mask_list=None, flag_stack_mr=False,
                flag_single_thread=False):
    fold_args = [p_fold]
    if not flag_flat_fold:
        fold_args.append('**')
    fold_args.append('*')
    dir_list = [os.path.normpath(d) for d in
                glob.glob(os.path.join(*fold_args))]
    img_map = {}
    seg_map = {}
    for fold in dir_list:
        flist = glob.glob(os.path.join(fold, '*.dcm'))
        if len(flist):
            dcm_data = pdcm.read_file(flist[0])
            modality = dcm_data.Modality.lower()
            if modality == 'rtstruct':
                contour_names = pdcmc.get_roi_names(dcm_data)
                if contour_idx_name is None:
                    contour_list = range(len(dcm_data.ROIContourSequence))
                else:
                    contour_list = [
                        idx for idx, s in enumerate(pdcmc.get_roi_names(
                            dcm_data))
                        if re.search(contour_idx_name, s, re.I)]
                    if not contour_list:
                        continue
                for ci in contour_list:
                    seg_data_c = SegData(
                        fold, contour_idx=ci,
                        flag_single_contour=flag_single_contour,
                        **(seg_data_args or {}))
                    if not seg_data_c._contours:
                        continue
                    if seg_data_c.contour_type == 'ctv' and \
                       'ctv' not in contour_names[ci].lower():
                        continue
                    ptdict.set_value(
                        seg_map, [seg_data_c.contour_type, seg_data_c.modality,
                                  seg_data_c.reader, seg_data_c.trial],
                        seg_data_c)
            elif modality_list is None or modality.lower() in modality_list:
                image_data = ImageData(fold)
                image_img = image_data.get_image()
                img_mod = image_data.modality
                if image_data.modality in img_map:
                    img_mod += '-1'  # Assume at most two images (T1, T2 MR)
                ptdict.set_value(img_map, [img_mod], [image_data, image_img])
    # Fallback search for images in parent folder
    if 'ct' not in img_map:
        p_fold_p = os.path.abspath(os.path.join(p_fold, '..'))
        flist_p = glob.glob(os.path.join(p_fold_p, '**', '*.dcm'),
                            recursive=True)
        dlist_p = [pdcm.read_file(f) for f in flist_p]
        # Not updated to support GTV/CTV structure
        img_ids_all = [ids for s in seg_map.values()
                       for sm in s.values() for smt in sm.values()
                       for ids in smt._img_ids]
        for fp, dp in zip(flist_p, dlist_p):
            if dp.SOPInstanceUID in img_ids_all:
                image_data = ImageData(os.path.split(fp)[0])
                image_img = image_data.get_image()
                ptdict.set_value(img_map, [image_data.modality],
                                 [image_data, image_img])
                continue
    if flag_stack_mr and 'mr' in img_map and 'mr-1' in img_map:
        # Combine images
        img_dat = np.zeros(img_map['mr'][1].shape)
        for k in ['mr', 'mr-1']:
            idx_z = [i for i in range(img_map['mr'][1].shape[0])
                     if np.any(img_map[k][1][i])]
            img_dat[idx_z] = img_map[k][1][idx_z]
        # Combine ImageData objects
        img_obj = copy.deepcopy(img_map['mr'][0])
        # Replace
        del img_map['mr-1']
        del img_map['mr']
        img_map['mr'] = [img_obj, img_dat]
    # Limit to desired readers/modalities
    if reader_list is not None or modality_mask_list is not None:
        seg_map_out = {}
        for ctype, c_data in seg_map.items():
            if contour_type is None or ctype == contour_type.lower():
                if ctype not in seg_map_out:
                    seg_map_out[ctype] = {}
                for mod, m_data in c_data.items():
                    if modality_mask_list is None or \
                       mod.lower() in modality_mask_list:
                        if mod not in seg_map_out:
                            seg_map_out[ctype][mod] = {}
                        for reader, r_data in m_data.items():
                            if reader_list is None or reader in reader_list:
                                if reader not in seg_map_out[ctype][mod]:
                                    seg_map_out[ctype][mod][reader] = {}
                                for trial, t_data in r_data.items():
                                    ptdict.set_value(
                                        seg_map_out, [ctype, mod, reader,
                                                      trial],
                                        seg_map[ctype][mod][reader][trial])
        seg_map = seg_map_out
    # Output
    output = {'fold': p_fold, 'img_map': img_map, 'seg_map': seg_map}
    if flag_compute_masks:
        mask_map = {}
        for ctype, c_data in seg_map.items():
            for mod, m_data in c_data.items():
                for reader, r_data in m_data.items():
                    for trial, t_data in r_data.items():
                        ptdict.set_value(
                            mask_map, [ctype, mod, reader, trial],
                            t_data.get_mask(
                                img_map['ct'][0],
                                flag_single_thread=flag_single_thread))
        output['mask_map'] = mask_map
    return output

# %% List segmentation modalities manually

seg_list_p2 = [
    {'index': 1, 'reader': 2, 'trial': 1, 'modality': 'ct-mr-pet'},
    {'index': 2, 'reader': 2, 'trial': 2, 'modality': 'ct-mr-pet'},
    {'index': 3, 'reader': 2, 'trial': 3, 'modality': 'ct-mr-pet'},
    {'index': 4, 'reader': 2, 'trial': 1, 'modality': 'ct-mr'},
    {'index': 5, 'reader': 2, 'trial': 2, 'modality': 'ct-mr'},
    {'index': 6, 'reader': 2, 'trial': 3, 'modality': 'ct-mr'},
    {'index': 7, 'reader': 2, 'trial': 1, 'modality': 'ct'},
    {'index': 8, 'reader': 2, 'trial': 2, 'modality': 'ct'},
    {'index': 9, 'reader': 2, 'trial': 3, 'modality': 'ct'},
    {'index': 0, 'reader': 1, 'trial': 1, 'modality': 'ct'},
    {'index': 10, 'reader': 1, 'trial': 1, 'modality': 'mri'}]
seg_list_p3 = [
    {'index': 0, 'reader': 2, 'trial': 1, 'modality': 'ct-mr-pet'},
    {'index': 1, 'reader': 2, 'trial': 2, 'modality': 'ct-mr-pet'},
    {'index': 2, 'reader': 2, 'trial': 3, 'modality': 'ct-mr-pet'},
    {'index': 3, 'reader': 2, 'trial': 1, 'modality': 'ct-mr'},
    {'index': 4, 'reader': 2, 'trial': 2, 'modality': 'ct-mr'},
    {'index': 5, 'reader': 2, 'trial': 3, 'modality': 'ct-mr'},
    {'index': 6, 'reader': 2, 'trial': 1, 'modality': 'ct'},
    {'index': 7, 'reader': 2, 'trial': 2, 'modality': 'ct'},
    {'index': 8, 'reader': 2, 'trial': 3, 'modality': 'ct'},
    {'index': 9, 'reader': 1, 'trial': 1, 'modality': 'ct'}]
seg_list_p4 = [
    {'index': 0, 'reader': 2, 'trial': 1, 'modality': 'ct-mr-pet'},
    {'index': 1, 'reader': 2, 'trial': 2, 'modality': 'ct-mr-pet'},
    {'index': 2, 'reader': 2, 'trial': 3, 'modality': 'ct-mr-pet'},
    {'index': 3, 'reader': 2, 'trial': 1, 'modality': 'ct-mr'},
    {'index': 4, 'reader': 2, 'trial': 2, 'modality': 'ct-mr'},
    {'index': 5, 'reader': 2, 'trial': 3, 'modality': 'ct-mr'},
    {'index': 6, 'reader': 2, 'trial': 1, 'modality': 'ct'},
    {'index': 7, 'reader': 2, 'trial': 2, 'modality': 'ct'},
    {'index': 8, 'reader': 2, 'trial': 3, 'modality': 'ct'},
    {'index': 9, 'reader': 1, 'trial': 1, 'modality': 'ct'}]
seg_list_p5 = [
    {'index': 0, 'reader': 2, 'trial': 1, 'modality': 'ct'},
    {'index': 1, 'reader': 2, 'trial': 2, 'modality': 'ct'},
    {'index': 2, 'reader': 2, 'trial': 3, 'modality': 'ct'}]
seg_list_all = {'p2': seg_list_p2, 'p3': seg_list_p3, 'p4': seg_list_p4,
                'p5': seg_list_p5}

# %% Network tools

def get_tfr_data(tfr_fname, num_images):
    sess = tf.Session()
    batch = read_fn(tfr_fname, perform_shuffle=False, repeat_count=1,
                    batch_size=num_images, label_dtype=np.uint8,
                    out_shape=img_shape, input_name=input_name)
    with sess.as_default():
        d = sess.run(batch)
        x = d[0][input_name]
        y = d[1]
    return x, y

# %% TFRecord interface

def write_tfrecord_v2(x, y, writer):
    """Write pairs of training sample and label to disk"""
    f_wrap = lambda x: tf.train.Feature(bytes_list=tf.train.BytesList(
        value=[x.tobytes()]))
    data = {'image': f_wrap(x),
            'label': f_wrap(y)}
    # Wrap the data as TensorFlow Features.
    features = tf.train.Features(feature=data)
    # Wrap again as a TensorFlow Example.
    example = tf.train.Example(features=features)
    # Serialize the data.
    serialized = example.SerializeToString()
    # Write the serialized data to the TFRecords file.
    writer.write(serialized)

def write_tfrecord(x, y, writer, label_dtype=np.float32, slice_window=None):
    """Write pairs of training sample and label to disk"""
    if x.ndim == 2 or slice_window is None:
        xx = x
        yy = y
    else:
        xx = np.moveaxis(x, 0, -1)
        yy = y[slice_window]
    data = {
        'image': tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[xx.astype(np.float32).tobytes()])),
        'label': tf.train.Feature(
            bytes_list=tf.train.BytesList(
                value=[yy.astype(label_dtype).tobytes()]))}
    # Wrap the data as TensorFlow Features.
    feature = tf.train.Features(feature=data)
    # Wrap again as a TensorFlow Example.
    example = tf.train.Example(features=feature)
    # Serialize the data.
    serialized = example.SerializeToString()
    # Write the serialized data to the TFRecords file.
    writer.write(serialized)

def read_fn_v2(filenames, perform_shuffle=False, repeat_count=1, batch_size=1,
               label_dtype=np.float32, out_shape_img=None, out_shape_lbl=None,
               data_augment=False):
    """TFRecord reader"""
    def _parse_function(serialized):
        features = {'image': tf.io.FixedLenFeature([], tf.string),
                    'label': tf.io.FixedLenFeature([], tf.string)}
        # Parse the serialized data so we get a dict with our data.
        parsed_example = tf.io.parse_single_example(serialized=serialized,
                                                    features=features)
        # Get the image as raw bytes.
        image = tf.reshape(tf.io.decode_raw(
            parsed_example['image'], tf.float32), out_shape_img)
        label = tf.reshape(tf.io.decode_raw(
            parsed_example['label'], label_dtype), out_shape_lbl)
        return image, label

    dataset = tf.data.TFRecordDataset(filenames=filenames)
    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the image and labels.
    # dataset = dataset.map(_parse_function,
    #                       num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(_parse_function)
    # Data augmentation
    if data_augment:
        raise NotImplementedError('Data augmentation not implemented')
        # dataset = dataset.map(lambda x, l: augment_data(
        #     x, l, out_shape, batch_size, flag_use_zoom=len(out_shape) == 2))
    if perform_shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(repeat_count)  # Repeats dataset this # times
    # dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(10)
    dataset = dataset.batch(batch_size)  # Batch size to use
    return dataset

def read_fn(filenames, perform_shuffle=False, repeat_count=1, batch_size=1,
            label_dtype=np.float32, out_shape=None, input_name=None,
            data_augment=False, flag_return_dataset=False):
    """TFRecord reader"""
    def _parse_function(serialized):
        features = {'image': tf.FixedLenFeature([], tf.string),
                    'label': tf.FixedLenFeature([], tf.string)}
        # Parse the serialized data so we get a dict with our data.
        parsed_example = tf.parse_single_example(serialized=serialized,
                                                 features=features)
        # Get the image as raw bytes.
        if len(out_shape) == 2:
            out_shape_img = list(out_shape) + [1, ]
            out_shape_lbl = out_shape_img
        elif len(out_shape) == 3:
            out_shape_img = out_shape
            out_shape_lbl = list(out_shape[:-1]) + [1, ]
        image = tf.reshape(tf.decode_raw(parsed_example['image'], tf.float32),
                           out_shape_img)
        label = tf.reshape(tf.decode_raw(parsed_example['label'], label_dtype),
                           out_shape_lbl)
        #d = {input_name: image}, label
        d = image, label
        return d

    dataset = tf.data.TFRecordDataset(filenames=filenames)
    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the image and labels.
    dataset = dataset.map(_parse_function)
    if perform_shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(repeat_count)  # Repeats dataset this # times
    dataset = dataset.batch(batch_size)  # Batch size to use
    # dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=4)
    # Data augmentation
    if data_augment:
        dataset = dataset.map(lambda x, l: augment_data(
            x, l, out_shape, batch_size, flag_use_zoom=len(out_shape) == 2))
    if flag_return_dataset:
        return dataset
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels

# %% TFRecord generation

def add_tfr_img_v2(tfr_writer_list, img, lbl, patch_params=None):
    sl_mask_lbl = np.array([np.any(s) for s in lbl])
    sl_mask_img = np.all(np.array([[np.any(s) for s in img[..., mi]]
                                   for mi in range(img.shape[-1])]), axis=0)
    sl_mask = sl_mask_lbl & sl_mask_img
    if patch_params is not None:
        sl_mask = skmorph.binary_dilation(
            sl_mask, np.ones(patch_params['patch_size'][0]))
    img_sl, lbl_sl = img[sl_mask], lbl[sl_mask]
    if patch_params is None:
        img_list = [img_sl]
        lbl_list = [lbl_sl]
        patch_idx_list = None
    else:
        patch_params_img = copy.deepcopy(patch_params)
        patch_params_lbl = copy.deepcopy(patch_params)
        for pp, x in zip([patch_params_img, patch_params_lbl],
                         [img, lbl]):
            pp['patch_size'] += [x.shape[-1], ]
            pp['patch_overlap'] += [0, ]
            pp['edge_mode'] = 'clip'
        op_patch_img = ptop.OperPatchPartitioner(img_sl.shape,
                                                 **patch_params_img)
        op_patch_lbl = ptop.OperPatchPartitioner(lbl_sl.shape,
                                                 **patch_params_lbl)
        patch_mask_list = [np.any(lbl_t[..., 1])
                           for lbl_t in op_patch_lbl.img_gen(lbl_sl)]
        patch_idx_list = [i for i in np.arange(len(patch_mask_list))
                          if patch_mask_list[i]]
        img_list, lbl_list = [op_patch.img_gen(z)
                              for op_patch, z in zip(
                                      [op_patch_img, op_patch_lbl],
                                      [img_sl, lbl_sl])]
    for pi, (img_t, lbl_t) in enumerate(zip(img_list, lbl_list)):
        if patch_idx_list is None or pi in patch_idx_list:
            tfr_writer = np.random.choice(tfr_writer_list)
            write_tfrecord_v2(img_t, lbl_t, tfr_writer)


def add_tfr_img(tfr_writer_list, img, lbl, downsample_factor=1,
                flag_mask_only=False, slice_window=None, patch_params=None,
                sl_active=None, **kwargs):
    if sl_active is None:
        if flag_mask_only:
            # Find slices with mask
            sl_active = np.nonzero(np.any(
                lbl.reshape([img.shape[0], -1]), axis=1))[0]
            if flag_mask_only == 'balance':
                # Order other slices by distance to center
                sl_empty = sorted([p for p in range(img.shape[0])
                                   if p not in sl_active],
                                  key=lambda x: np.abs(x - img.shape[0] / 2))
                # Select same amount of empty slices
                sl_add = sl_empty[:np.min([len(sl_empty), len(sl_active)])]
                if sl_add:
                    sl_active = np.hstack([sl_active, sl_add])
        else:
            sl_active = np.arange(img.shape[0])
    for sl in sl_active:
        if slice_window is None:
            sl_c = sl
        else:
            sl_c = np.clip(np.arange(sl - slice_window,
                                     sl + slice_window + 1),
                           0, img.shape[0] - 1)
        sl_img = img[sl_c].squeeze()
        sl_lbl = lbl[sl_c].squeeze()
        if img.ndim == 3:
            sl_img_d = sl_img[..., ::downsample_factor, ::downsample_factor]
        elif img.ndim == 4:
            sl_img_d = sl_img[..., ::downsample_factor, ::downsample_factor, :]
        sl_lbl_d = sl_lbl[..., ::downsample_factor, ::downsample_factor]
        if patch_params is None:
            img_list = [sl_img_d]
            lbl_list = [sl_lbl_d]
            patch_idx_list = None
        else:
            op_patch = ptop.OperPatchPartitioner(sl_img_d.shape,
                                                 **patch_params)
            patch_mask_list = [np.any(lbl_t)
                               for lbl_t in op_patch.img_gen(sl_lbl_d)]
            if flag_mask_only:
                patch_idx_list = [i for i in np.arange(len(patch_mask_list))
                                  if patch_mask_list[i]]
                if flag_mask_only == 'balance':
                    num_patches = op_patch._num_patches_all
                    patch_idx_2 = np.stack(np.unravel_index(
                        np.arange(len(patch_mask_list)),
                        op_patch._num_patches_all))
                    patch_idx_s = sorted(
                        np.arange(len(patch_mask_list)),
                        key=lambda x: np.sum(
                            (patch_idx_2[:, x] -
                             np.array(num_patches) / 2)**2))
                    patch_idx_list.extend(patch_idx_s[:len(patch_idx_list)])
                elif flag_mask_only == 'fully_contained_patch':
                    p_idx = []
                    patch_gen = list(op_patch.img_gen(sl_lbl_d))
                    sum_mask = np.sum(sl_lbl_d > 0)
                    for pi in patch_idx_list:
                        sum_patch = np.sum(patch_gen[pi] > 0)
                        if sum_mask == sum_patch:
                            p_idx.append(pi)
                    patch_idx_list = p_idx
            else:
                patch_idx_list = np.arange(len(patch_mask_list))

            img_list, lbl_list = [op_patch.img_gen(z)
                                  for z in [sl_img_d, sl_lbl_d]]
        for pi, (img_t, lbl_t) in enumerate(zip(img_list, lbl_list)):
            if patch_idx_list is None or pi in patch_idx_list:
                tfr_writer = np.random.choice(tfr_writer_list)
                write_tfrecord(img_t, lbl_t, tfr_writer,
                               slice_window=slice_window, **kwargs)

def dnn0_add_patient(tfr_writer_list, pidx, fold=None, flag_combine=False,
                     mask_idx=None, modality='ct', modality_mask=None,
                     load_modality_list=None, reader_list=None,
                     load_params=None, write_mode='tfr', **kwargs):

    if write_mode == 'tfr':
        def write(data_p=None, *args, **kwargs):
            add_tfr_img(*args, **kwargs)
    elif write_mode == 'pik':
        def write(flist, img, lbl, data_p=None, **kwargs):
            aux = {'pixel_size': data_p['img_map']['ct'][0]._spacing[::-1],
                   'dims': list(img.shape)}
            json.dump(aux, open(flist[0] + '.json', 'wt'))
            pickle.dump(
                {'img': img,
                 'slices_mask': np.nonzero([np.any(x) for x in lbl])[0],
                 'lbl': lbl}, open(flist[0], 'wb'), protocol=4)
    else:
        raise RuntimeError('write_mode not supported')

    if fold is None:
        fold = os.path.join(dfold, pidx)
    data_p = do_load_all(fold, modality_list=load_modality_list,
                         **(load_params or {}))
    if modality_mask is None:
        modality_mask = modality
    if isinstance(modality, str):
        img_in = data_p['img_map'][modality][1]
    else:
        # Multimodality
        img_in = np.zeros(list(data_p['img_map'][modality[0]][1].shape) +
                          [len(modality), ], dtype=np.float32)
        if 'mr' in modality:
            m_list = [d for d in data_p['img_map'].keys() if 'mr' in d]
            if len(m_list) > 1:
                # Test of halved MR
                sl_list = np.stack([np.array(
                    [np.any(s) for s in data_p['img_map'][m][1]])
                                    for m in m_list])
                sl_list_s = np.sum(sl_list, axis=0)
                if sl_list_s.max() == 1 and \
                   np.sum(np.diff(sl_list_s) == 1) == 1 and \
                   np.sum(np.diff(sl_list_s) == -1) == 1:
                    img_mr = np.zeros(data_p['img_map'][modality[0]][1].shape,
                                      dtype=np.float32)
                    for s, m in zip(sl_list, m_list):
                        img_mr[s] = data_p['img_map'][m][1][s]
                else:
                    img_mr = data_p['img_map']['mr'][1]
            else:
                img_mr = data_p['img_map']['mr'][1]
        for mi, mod in enumerate(modality):
            if mod == 'mr':
                img_t = img_mr
            else:
                img_t = data_p['img_map'][mod][1]
            img_in[..., mi] = img_t
    if modality_mask not in data_p['mask_map']:
        warnings.warn('No contour found for dataset, skipping {}'.format(fold))
        return
    mask_all_mod = [data_p['mask_map'][modality_mask][kr][kt].astype(np.uint8)
                   for kr in data_p['mask_map'][modality_mask].keys()
                   for kt in data_p['mask_map'][modality_mask][kr].keys()
                   if reader_list is None or kr in reader_list]
    if mask_idx is not None:
        mask_all_mod = [mask_all_mod[mask_idx], ]

    if flag_combine is True or flag_combine == 'sum-bool':
        mask_comb_mod = np.zeros(mask_all_mod[0].shape, dtype=np.float32)
        for mask in mask_all_mod:
            mask_comb_mod += mask.astype(np.float32)
        if flag_combine is True:
            mask_comb_mod /= len(mask_all_mod)
        elif flag_combine == 'sum-bool':
            mask_comb_mod = mask_comb_mod > 0
        dtype = np.uint8 if flag_combine == 'sum-bool' else np.float32
        write(tfr_writer_list, img_in, mask_comb_mod,
              label_dtype=dtype, data_p=data_p, **kwargs)
    elif get_num_classes_cons_combine(flag_combine) is not None:
        nbins = get_num_classes_cons_combine(flag_combine)
        dig_bins = np.linspace(0, len(mask_all_mod) - 1, nbins - 1)
        mask_comb_mod = np.digitize(np.sum(mask_all_mod, axis=0),
                                    bins=dig_bins, right=True)
        # mask_comb_mod = np.any(np.stack(mask_all_mod),
        #                       axis=0).astype(np.uint8)
        # mask_comb_mod[np.all(np.stack(mask_all_mod), axis=0)] = 2
        write(tfr_writer_list, img_in, mask_comb_mod,
              label_dtype=np.uint8, data_p=data_p, **kwargs)
    else:
        for mask in mask_all_mod:
            write(tfr_writer_list, img_in, mask,
                  label_dtype=np.uint8, data_p=data_p, **kwargs)

def dnn0_load_patient_from_tfr(tfr_fold_in, pidx, label_dtype, img_shape,
                               fname=None):
    sess = tf.Session()
    if fname is None:
        fname = os.path.join(tfr_fold_in, pidx + '.tfr')
    with sess.as_default():
        num_rec = sum(1 for i in tf.python_io.tf_record_iterator(fname))
        X = sess.run(read_fn(fname, perform_shuffle=False, repeat_count=None,
                             batch_size=num_rec, label_dtype=label_dtype,
                             out_shape=img_shape, input_name='layer1'))
    return X

def dnn0_add_patient_from_tfr(tfr_writer_list, tfr_fold_in, pidx,
                              label_dtype, img_shape, **kwargs):
    X = dnn0_load_patient_from_tfr(tfr_fold_in, pidx, label_dtype, img_shape)
    add_tfr_img(tfr_writer_list, X[0], X[1], label_dtype=X[1].dtype, **kwargs)

def make_fname_info(exp_name, fname_lbl, num_tfr=100, plist_tst=None):
    fold_base = os.path.join(res_fold, exp_name)
    output = {
        'fold': fold_base,
        'tfr_fname_list_tr': [os.path.join(fold_base, 'data_{}_{}.tfr'.format(
            fname_lbl, i)) for i in range(num_tfr)]}
    if plist_tst is not None:
        output['tfr_fname_list_tst'] = {
            pidx: os.path.join(fold_base, 'data_tst_{}.tfr'.format(
                pidx)) for pidx in plist_tst}
    return output

def dnn1_add_patient(tfr_writer_list, pidx, fold=None, comb_mode='mean',
                     modality_img=None, modality_mask='ct-mr-pet',
                     contour_type_list=None, load_params=None,
                     patch_sel_type=None, write_mode='tfr', **kwargs):

    if write_mode == 'tfr':
        def write(data_out=None, *args, **kwargs):
            ptdl.write_tfr_patches(data_out, mask=mask_all['gtv'] > 0,
                                  tfr_list=tfr_writer_list, **kwargs)
    elif write_mode == 'pik':
        def write(data_out, mask, *args, **kwargs):
            with open(tfr_writer_list[0], 'wb') as fid:
                pickle.dump({**data_out, 'patch_sel_mask': mask}, fid,
                            protocol=4)
    else:
        raise RuntimeError('write_mode not supported')

    if modality_img is None:
        modality_img = ['ct', 'mr', 'pt']
    if contour_type_list is None:
        contour_type_list = ['gtv', 'ctv']
    if fold is None:
        fold = os.path.join(dfold, pidx)
    data_p = do_load_all(fold, **ptdict.merge(
        {'modality_list': modality_img,
         'seg_data_args': {'flag_fixed_pattern': True}},
        load_params))
    # Multimodality
    img_all = {}
    if 'mr' in modality_img:
        m_list = [d for d in data_p['img_map'].keys() if 'mr' in d]
        if len(m_list) > 1:
            # Test of halved MR
            sl_list = np.stack([np.array(
                [np.any(s) for s in data_p['img_map'][m][1]])
                                for m in m_list])
            sl_list_s = np.sum(sl_list, axis=0)
            if sl_list_s.max() == 1 and \
               np.sum(np.diff(sl_list_s) == 1) == 1 and \
               np.sum(np.diff(sl_list_s) == -1) == 1:
                img_all['mr'] = np.zeros(
                    data_p['img_map'][modality_img[0]][1].shape,
                    dtype=np.float32)
                for s, m in zip(sl_list, m_list):
                    img_all['mr'][s] = data_p['img_map'][m][1][s]
            else:
                img_all['mr'] = data_p['img_map']['mr'][1]
        else:
            img_all['mr'] = data_p['img_map']['mr'][1]
        img_all['mr'] = (img_all['mr'] - img_all['mr'].min()) / \
            (img_all['mr'].max() - img_all['mr'].min())
    if 'ct' in modality_img:
        img_all['ct'] = data_p['img_map']['ct'][1] / 2000
    if 'pt' in modality_img:
        img_all['pt'] = ptnum.compute_suv(data_p['img_map']['pt'][1],
                                          data_p['img_map']['pt'][0]._dcm_img)

    mask_all = {}
    for contour_type in contour_type_list:
        mask_map = data_p['mask_map'][contour_type][modality_mask]
        mask_all_mod = [mask_map[kr][kt].astype(np.uint8)
                        for kr in mask_map.keys()
                        for kt in mask_map[kr].keys()
                        if reader_list is None or kr in reader_list]
        mask_comb_mod = np.zeros(mask_all_mod[0].shape, dtype=np.float32)
        for mask in mask_all_mod:
            mask_comb_mod += mask.astype(np.float32)
        mask_comb_mod /= len(mask_all_mod)
        mask_all[contour_type] = mask_comb_mod

    data_out = ptdict.merge(
        {k: {'data': v, 'type': 'float32'} for k, v in img_all.items()},
        {'mask_comb_{}'.format(k): {'data': v, 'type': 'float32'}
         for k, v in mask_all.items()})
    if patch_sel_type is None:
        patch_sel_mask = None
    else:
        patch_sel_mask = mask_all[patch_sel_type] > 0

    return write(data_out, mask=patch_sel_mask,
                 tfr_list=tfr_writer_list, **kwargs)

# %% Data augmentation

def augment_data_flip(x, l):
    return tf.cond(tf.random_uniform(shape=[], minval=0, maxval=1,
                                     dtype=tf.float32) < 0.5,
                   # lambda: (tf.reverse(x, [-2]),
                   #          tf.reverse(l, [-2])),
                   lambda: (tf.image.flip_left_right(x),
                            tf.image.flip_left_right(l)),
                   lambda: (x, l))

def augment_data_zoom(x, l, img_shape, batch_size):
    # Generate 20 crop settings, ranging from a 1% to 20% crop.

    # The following code assumes that the batch size is always smaller than the
    # number of scales (100 here)
    scales = list(np.linspace(0.8, 1.0, 100))
    boxes = np.zeros((len(scales), 4), dtype=np.float32)

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    # Create different crops for an image
    boxes_tf = tf.random_shuffle(boxes)[:batch_size]
    crops_x = tf.image.crop_and_resize(
        x, boxes=boxes_tf, box_ind=tf.range(batch_size),
        method='bilinear', crop_size=img_shape)
    crops_l = tf.cast(tf.image.crop_and_resize(
        l, boxes=boxes_tf, box_ind=tf.range(batch_size),
        method='bilinear', crop_size=img_shape), l.dtype)

    # Return a random crop
    return crops_x, crops_l

def augment_data(x, l, img_shape, batch_size, flag_use_zoom=True):
    choice = tf.random_uniform(shape=[], minval=0, maxval=3, dtype=tf.int32)
    if flag_use_zoom:
        return tf.case([(tf.equal(choice, 0), lambda: (x, l)),
                        (tf.equal(choice, 1), lambda: augment_data_flip(x, l)),
                        (tf.equal(choice, 2), lambda: augment_data_zoom(
                            x, l, img_shape, batch_size))])
    return tf.case([(tf.equal(choice, 0), lambda: (x, l)),
                    (tf.equal(choice, 1), lambda: augment_data_flip(x, l)),
                    (tf.equal(choice, 2), lambda: (x, l))])

# %% Test

# fname = os.path.join(tfr_fold, patient_idx_tr[-1] + '.tfr')
# fname = os.path.join(tfr_fold, patient_idx_tr[0] + '.tfr')
# with sess.as_default():
#     num_rec = sum(1 for i in tf.python_io.tf_record_iterator(fname))
#     X0 = sess.run(read_fn(fname, perform_shuffle=False, repeat_count=None,
#                           batch_size=num_rec, label_dtype=label_dtype,
#                           data_augment=False,
#                           out_shape=img_shape, input_name='layer1'))
#     X1 = sess.run(read_fn(fname, perform_shuffle=False, repeat_count=None,
#                           batch_size=num_rec, label_dtype=label_dtype,
#                           data_augment=True,
#                           out_shape=img_shape, input_name='layer1'))

# with sess.as_default():
#     num_rec = sum(1 for i in tf.python_io.tf_record_iterator(fname))
#     Z0 = list(dnn0_est.predict(lambda : read_fn(
#         fname, perform_shuffle=False, repeat_count=None,
#         batch_size=16, label_dtype=label_dtype, data_augment=False,
#         out_shape=img_shape, input_name='layer1')))

# %% Patch-based inference function

def predict_image_patches(img, est, op_patch, mask=None, slice_window=None,
                          flag_ordinal_multiclass=False):
    if mask is None:
        mask = img
    patch_size_base = list(op_patch._element_size)
    if len(patch_size_base) == 2:
        patch_size_base += [1, ]

    if op_patch._invert_weights is None:
        op_patch.compute_invert_weights()
    if slice_window is None:
        inv_weights = op_patch._invert_weights.squeeze()
    else:
        inv_weights = op_patch._invert_weights.squeeze()[..., :1]

    # Prediction function
    def pred_input_fn(img_s, mask_s, op_patch):
        def img_gen():
            gen_img = op_patch.img_gen(img_s)
            gen_mask = op_patch.img_gen(mask_s)
            for img_t, mask_t in zip(gen_img, gen_mask):
                yield (img_t.reshape([1, ] + patch_size_base),
                       mask_t.reshape([1, ] + patch_size_base))

        return tf.data.Dataset.from_generator(
            img_gen, output_types=(np.float32, np.float32),
            output_shapes=([None, ] + patch_size_base,
                           [None, ] + patch_size_base))

    flag_multi_class = False
    img_out = None
    inv_weights_slice = None
    # Process slices
    for sl in range(img.shape[0]):
        if slice_window is None:
            sl_c = sl
        else:
            sl_c = np.clip(np.arange(sl - slice_window, sl + slice_window + 1),
                           0, img.shape[0] - 1)
        # Prepare prediction iterator
        img_s = img[sl_c]
        mask_s = mask[sl_c]
        if slice_window is not None:
            img_s = np.moveaxis(img_s, 0, -1)
            mask_s = np.moveaxis(mask_s, 0, -1)
        y_pred = est.predict(input_fn=lambda: pred_input_fn(
            img_s, mask_s, op_patch))

        op_patch.reset_generator()
        for y_p, idx in zip(y_pred, op_patch.idx_gen):
            y_c = y_p['prob'].squeeze()
            if y_c.ndim != 2:
                if y_c.shape[-1] == 2:
                    y_c = y_c[..., 1]
                else:
                    flag_multi_class = True
            if img_out is None:
                if flag_multi_class:
                    img_out = np.zeros(list(img.shape) + [y_c.shape[-1]])
                else:
                    img_out = np.zeros_like(img)
            if slice_window is None:
                idx_in = idx
                idx_out = idx
            else:
                idx_out = idx[:-1]
                idx_in = idx[:-1]# + [np.arange(y_c.shape[-1])]
            img_in = y_c[np.ix_(*[np.arange(len(el)) for el in idx_in])]
            img_out[sl][np.ix_(*idx_out)] += img_in
        if inv_weights_slice is None:
            if flag_multi_class:
                inv_weights_slice = inv_weights.squeeze()[..., None]
            else:
                inv_weights_slice = inv_weights.squeeze()
        img_out[sl] *= inv_weights_slice

    if flag_multi_class:
        if flag_ordinal_multiclass:
            img_out = np.maximum(1, (np.sum(img_out >= 0.5, axis=-1))) - 1
        else:
            img_out = np.argmax(img_out, axis=-1)
    return img_out

# %% Helper functions

def plot_metrics(model_info_list, metric_info_list, plot_nxy=None,
                 flag_export=False, fname_export='metrics', lgd_fontsize=12,
                 figsize=[8, 8]):
    if plot_nxy is None:
        plot_nxy = ptnum.split_factors(len(metric_info_list))
    elif plot_nxy == 'col':
        plot_nxy = [len(metric_info_list), 1]
    fig_name_metrics = ''
    fig_metrics = plt.figure(fig_name_metrics)
    fig_metrics.clf()
    fig_metrics, ax_metrics = plt.subplots(
        num=fig_name_metrics, nrows=plot_nxy[0], ncols=plot_nxy[1],
        squeeze=False)
    metric_names = [metric_info['name'] for metric_info in metric_info_list]
    for metric_info, ax in zip(metric_info_list, ax_metrics.flatten()):
        metric = metric_info['name']
        for model_info in model_info_list:
            fname_metric = model_info['fname_fmt'] + \
                '-tag-{}.json'.format(metric)
            if os.path.isfile(fname_metric):
                data_loss = np.array(json.load(open(fname_metric)))
                label = '{}'.format(model_info['name'])
                ax.plot(data_loss[:, 1], data_loss[:, 2],
                        label=re.sub('_', ' ', label),
                        linewidth=0.6)
            else:
                next(ax._get_lines.prop_cycler)
        if metric_info.get('flag_ylog', False):
            ax.set_yscale('log')
        ax.set_xlabel('Step')
        lgd = ax.legend(borderpad=0, fontsize=lgd_fontsize)
        lgd.get_frame().set_linewidth(0.0)
        ax.set_title(re.sub('_', ' ', metric))

    if flag_export:
        ptexp.export_fig(fig_metrics, size=figsize, dpi=600,
                         output_format='png',
                         output_folder=os.path.split(fname_export)[0],
                         output_fname=os.path.split(fname_export)[1])

    return fig_metrics, ax_metrics


def plot_results(pidx, sl, cbox, model_info_list, flag_export=False,
                 fname_export='images', lbl_fontsize=12, fig_width=8,
                 contour_exc_list=None, title_len_max=40, flag_comb=None,
                 mask_idx=None, flag_no_base=False, flag_no_suff=False):
    tr_map = {'tr': 'train', 'tst': 'valid'}
    vmin, vmax = -100, 300
    p_lbl_list = ['CT', 'Reference map'] + \
        [re.sub('_', ' ', m['name']) for m in model_info_list]

    num_img_base = 1 if flag_no_base else 2
    if num_img_base == 1:
        p_lbl_list = p_lbl_list[1:]
    fig_name = '{}'.format(pidx)
    fig_p = plt.figure(fig_name)
    fig_p.clf()
    nrows, ncols = ptnum.split_factors(num_img_base + len(model_info_list))
    fig_p, ax_p = plt.subplots(num=fig_name, nrows=nrows, ncols=ncols,
                               sharex=True, sharey=True, squeeze=False)

    p_data_list = []
    for model_info in model_info_list:
        model_fold = model_info['fname_fmt'] + '-inf'
        model_name = model_info['name']
        model_out_sc = model_info.get('scale', None)
        fname_wc = os.path.join(model_fold, 'data_*_{}_img.dat'.format(pidx))
        fname_c = glob.glob(fname_wc)[0]
        tr_res = re.search(re.sub(r'\*', '([trs]+)', fname_wc), fname_c)
        tr_mode = tr_res.groups()[0]
        tr_lbl = tr_map[tr_mode]
        data_load = load_inf(os.path.join(
            model_fold, re.sub('_img.dat', '', fname_c)))
        if contour_exc_list is not None:
            data_load_1 = np.stack([data_load[1][i]
                                    for i in range(data_load[1].shape[0])
                                    if i not in contour_exc_list])
        else:
            data_load_1 = data_load[1]
        data_load_out = [data_load[0], data_load_1,
                         data_load[2], data_load[3]]
        p_data_list.append([data_load_out, ' ({})'.format(tr_lbl)])

    rp = skmeas.regionprops(skmeas.label(np.any(p_data_list[0][0][1], axis=0)))
    if 'auto' in sl:
        sl_off = 0
        if sl != 'auto':
            sl_off = int(sl.split(':')[1])
        sl_c = int(np.round(rp[0].centroid[0])) + sl_off
    else:
        sl_c = sl
    csz, csy0, csx0, cez, cey0, cex0 = rp[0].bbox
    W = 0.05
    if 'auto' in cbox and cbox != 'auto':
        W = float(cbox.split('-')[1])
    csy = int(np.round(np.max([csy0 - W * (cey0 - csy0), 0])))
    cey = int(np.round(np.min([cey0 + W * (cey0 - csy0),
                               p_data_list[0][0][1].shape[2] - 1])))
    csx = int(np.round(np.max([csx0 - W * (cex0 - csx0), 0])))
    cex = int(np.round(np.min([cex0 + W * (cex0 - csx0),
                               p_data_list[0][0][1].shape[3] - 1])))
    if 'auto' in cbox:
        cbox_c = [csy, cey, csx, cex]
    else:
        cbox_c = cbox

    def f_img(x):
        if x.ndim == 4:
            if x.shape[-1] == 2:
                # Pixel-wise probability, use class 1
                x = x[..., 1]
            else:
                # Multiple mask, average
                x = np.mean(x, axis=0)
        # return x[sl_c][csy:cey, csx:cex]
        return x[sl_c][cbox_c[0]:cbox_c[1], cbox_c[2]:cbox_c[3]]

    img_base = None
    p_img_in_ref = None
    for pi in range(num_img_base + len(model_info_list)):
        model_out_sc_c = model_out_sc
        t_suff = ''
        if pi < num_img_base:
            img_tb = p_data_list[0][0][:2]
        else:
            img_tb = p_data_list[pi - num_img_base][0][2:]
            if not flag_no_suff:
                t_suff = p_data_list[pi - num_img_base][1]

        p_lbl = p_lbl_list[pi]
        ax = ax_p[pi // ncols, pi % ncols]
        ax.matshow(f_img(p_data_list[0][0][0]), cmap=plt.cm.gray,
                   vmin=vmin, vmax=vmax)
        if img_base is None:
            img_base = f_img(p_data_list[0][0][0])
        if pi >= num_img_base - 1:
            p_img_in = img_tb[1 if pi == num_img_base - 1 else 0]
            if p_img_in.ndim == 4:
                if flag_comb == 'cons3c':
                    p_img_in = np.any(p_img_in, axis=0).astype(np.float32) + \
                        np.all(p_img_in, axis=0).astype(np.uint8)
                    p_img_in /= 2
                elif mask_idx is not None:
                    p_img_in = p_img_in[mask_idx]
                elif 2 <= p_img_in.shape[-1] <= 3:
                    p_img_in = p_img_in[..., -1]
                    model_out_sc_c = None
                else:
                    p_img_in = np.mean(p_img_in, axis=0)
            p_img_in = p_img_in.astype(np.float32) / np.max(p_img_in)
            if pi >= num_img_base and model_out_sc_c is not None:
                p_img_in = model_out_sc_c * p_img_in.astype(np.float32)
            # if p_img_in_ref is None:
            #     p_img_in_ref = copy.deepcopy(p_img_in)
            # else:
            #     x1, y1 = p_img_in == 1, p_img_in_ref == 1
            #     x2, y2 = p_img_in >= 0.5, p_img_in_ref >= 0.5
            #     sl_m = np.array([np.any(m)
            #                      for m in p_img_in_ref])[:, None, None]
            #     for pi, (m0_t, m1_t) in enumerate([[x1, y1], [x2, y2]]):
            #         m0 = m0_t[:, csy:cey, csx:cex] * sl_m
            #         m1 = m1_t[:, csy:cey, csx:cex] * sl_m
            #         dice = 2 * np.sum(m0 * m1) / (np.sum(m0) + np.sum(m1))
            #         cdice = 2 * np.sum(p_img_in * p_img_in_ref) / \
            #             (np.sum(p_img_in + p_img_in_ref))
            #         print(pi, m0.shape, dice, cdice)
            #     # print(np.sqrt(np.mean(p_img_in - p_img_in_ref)**2) /
            #     #       (np.max(p_img_in_ref) - np.min(p_img_in_ref)))
            #     # metrics = {'dice1':
            #     #            2 * np.sum((p_img_in >= 0.75) *
            #     #                       (p_img_in_ref >= 0.75)) /
            #     #            (np.sum(p_img_in >= 0.75) +
            #     #             np.sum(p_img_in_ref >= 0.75)),
            #     #            'dice0':
            #     #            2 * np.sum((p_img_in >= 0.25) *
            #     #                       (p_img_in_ref >= 0.25)) /
            #     #            (np.sum(p_img_in >= 0.25) +
            #     #             np.sum(p_img_in_ref >= 0.25))}
            #     # p_lbl = p_lbl.format(**metrics)
            p_img = f_img(p_img_in)
            seg_ma = np.ma.masked_where(p_img < 0.1, p_img)
            ax.matshow(seg_ma, cmap=plt.cm.autumn, alpha=0.4,
                       vmin=0, vmax=1)
        ax.text(5, 5, '\n'.join(textwrap.wrap(p_lbl + t_suff, title_len_max)),
                horizontalalignment='left', verticalalignment='top',
                color='white', fontsize=lbl_fontsize)
    for ax in ax_p.flatten():
        ax.set_axis_off()

    fig_p.subplots_adjust(bottom=0, top=1, left=0, right=1,
                          hspace=0.01, wspace=0.01)

    fig_height = fig_width / (ncols * img_base.shape[1]) * \
        (nrows * img_base.shape[0])
    if flag_export:
        ptexp.export_fig(fig_p, size=[fig_width, fig_height], dpi=600,
                         axis_tight=False, tight_layout=False,
                         flag_skip_save=not flag_export,
                         output_format='png',
                         output_folder=os.path.split(fname_export)[0],
                         output_fname=os.path.split(fname_export)[1])

    return fig_p, ax_p

# %% Read metrics (saved during training)

def get_model_metrics_tr(model_dir, metric_list=None):
    fname_tr = glob.glob(os.path.join(model_dir, '*tfevents*'))[0]
    fname_ev = glob.glob(os.path.join(model_dir, 'eval', '*tfevents*'))
    fname_list = {'training': fname_tr}
    if fname_ev:
        fname_list['eval'] = fname_ev[0]
    output = {}
    for k, f in fname_list.items():
        output_t = {m: [] for m in metric_list or []}
        for e in tf.train.summary_iterator(f):
            for v in e.summary.value:
                if (metric_list is not None and v.tag in metric_list) or \
                   metric_list is None:
                    if v.tag not in output_t:
                        output_t[v.tag] = []
                    output_t[v.tag].append([e.step, v.simple_value])
        output[k] = output_t
    return output

def plot_metrics_tr(metric_list, model_list, output_fname, flag_export=False,
                    res_fold_base_list=None):
    """Plot metrics read from event files (using get_model_metrics_tr)"""
    nrows, ncols = ptnum.split_factors(len(metric_list))

    fig_name_metrics = 'metrics'
    fig_metrics = plt.figure(fig_name_metrics)
    fig_metrics.clf()
    ax_metrics = fig_metrics.subplots(nrows=nrows, ncols=ncols, squeeze=False)

    for mi, model in enumerate(model_list):
        if res_fold_base_list is None:
            res_fold_c = res_fold_base  # Global variable
        else:
            res_fold_c = res_fold_base_list[mi]
        model_dir = os.path.join(res_fold_c, 'model_' + model[0])
        metrics_out = get_model_metrics_tr(model_dir)
        if 'accuracy_1' in metrics_out['training']:
            metrics_out['training']['accuracy'] = \
                metrics_out['training']['accuracy_1']
        for ax, met in zip(ax_metrics.flatten(), metric_list):
            if not met in metrics_out['training']:
                next(ax._get_lines.prop_cycler)
                continue
            h_p = ax.plot([m[0] for m in metrics_out['training'][met]],
                          [m[1] for m in metrics_out['training'][met]],
                          label=model[1])
            if 'eval' in metrics_out:
                if not met in metrics_out['eval']:
                    continue
                ax.plot([m[0] for m in metrics_out['eval'][met]],
                        [m[1] for m in metrics_out['eval'][met]],
                        color=h_p[0].get_color(), linestyle='--')
    for ax, met in zip(ax_metrics.flatten(), metric_list):
        ax.set_title(met)
        if ax in ax_metrics[-1, :]:
            ax.set_xlabel('step')
        ax.legend(fontsize=7)
    for ax in ax_metrics.flatten()[len(metric_list):]:
        ax.set_axis_off()
    if flag_export:
        out_fold, out_fname_full = os.path.split(output_fname)
        out_fname, out_ext = os.path.splitext(out_fname_full)
        ptexp.export_fig(fig_metrics,
                         size=[10, 8],
                         dpi=300,
                         output_folder=out_fold,
                         output_fname=out_fname,
                         output_format=out_ext[1:])
    return fig_metrics, ax_metrics

# %% Compute segmentation metrics from precompute inference

def get_metrics_all(model_list, patient_idx_tst, roi_ext_frac=0.2,
                    contour_exc_map=None, res_fold_base_list=None,
                    metrics='all'):
    pixel_spacing = [0.9765625, 0.9765625, 5.0]

    metrics_all = []

    for mi, model in enumerate(model_list):
        if res_fold_base_list is None:
            res_fold_c = res_fold_base  # Global variable
        else:
            res_fold_c = res_fold_base_list[mi]
        inf_dir = os.path.join(res_fold_c, 'model_' + model[0] + '-inf')
        metrics_model = {}
        # for pidx in patient_idx_tr:
        #     pred_data = load_inf(os.path.join(inf_dir, 'data_tr_' + pidx))
        for pidx in patient_idx_tst:
            if os.path.isfile(os.path.join(
                    inf_dir, 'data_tst_{}_img.dat'.format(pidx))):
                dset_mode = 'tst'
            elif os.path.isfile(os.path.join(
                    inf_dir, 'data_tr_{}_img.dat'.format(pidx))):
                dset_mode = 'tr'
            else:
                raise FileNotFoundError('File not found')
            pred_data = load_inf(os.path.join(
                inf_dir, 'data_{}_{}'.format(dset_mode, pidx)))
            img, mask, pred, pred_p = pred_data
            if contour_exc_map is not None and pidx in contour_exc_map:
                mask = np.stack([mask[i] for i in range(mask.shape[0])
                                 if i not in contour_exc_map[pidx]])

            if pred_p.ndim == 3:
                mask_ref = np.mean(mask.astype(np.float32), axis=0)
                x = pred_p.astype(np.float32) / np.max(pred_p)
            elif pred_p.shape[3] == 3: # cons3c
                mask_ref = 0.5 * (np.any(mask, axis=0).astype(np.float32) +
                                  np.all(mask, axis=0).astype(np.float32))
                x = (np.maximum(np.sum(pred_p >= 0.5, axis=-1), 1) - 1) / 2
                # Calculate metrics for partial agreement region
                # mask_ref = np.any(mask, axis=0).astype(np.float32)
                # x = (np.maximum(np.sum(pred_p >= 0.5, axis=-1), 1) - 1) >= 1
            elif pred_p.shape[3] == 2:
                mask_ref = np.mean(mask.astype(np.float32), axis=0)
                x = pred_p[..., 1]
            else:
                raise NotImplementedError(
                    'Number of output classes not supported')

            mask_locs = np.nonzero(mask_ref)
            cx_min, cx_max = [f(mask_locs[2]) for f in [np.min, np.max]]
            cy_min, cy_max = [f(mask_locs[1]) for f in [np.min, np.max]]
            cz_min, cz_max = [f(mask_locs[0]) for f in [np.min, np.max]]
            if roi_ext_frac > 0.0:
                cx1_min = np.max([0,
                                  int(np.floor(cx_min - 0.5 * roi_ext_frac *
                                               (cx_max - cx_min)))])
                cx1_max = np.min([mask_ref.shape[-1] - 1,
                                  int(np.ceil(cx_max + 0.5 * roi_ext_frac *
                                              (cx_max - cx_min)))])
                cy1_min = np.max([0,
                                  int(np.floor(cy_min - 0.5 * roi_ext_frac *
                                               (cy_max - cy_min)))])
                cy1_max = np.min([mask_ref.shape[-1] - 1,
                                  int(np.ceil(cy_max + 0.5 * roi_ext_frac *
                                              (cy_max - cy_min)))])
                cz1_min = np.max([0,
                                  int(np.floor(cz_min - 0.5 * roi_ext_frac *
                                               (cz_max - cz_min)))])
                cz1_max = np.min([mask_ref.shape[-1] - 1,
                                  int(np.ceil(cz_max + 0.5 * roi_ext_frac *
                                              (cz_max - cz_min)))])
                cx_min, cx_max = cx1_min, cx1_max
                cy_min, cy_max = cy1_min, cy1_max
                cz_min, cz_max = cz1_min, cz1_max
                img_r = img[cz_min:cz_max, cy_min:cy_max, cx_min:cx_max]
                mask_ref_r = mask_ref[cz_min:cz_max, cy_min:cy_max,
                                      cx_min:cx_max]
                x_r = x[cz_min:cz_max, cy_min:cy_max, cx_min:cx_max]

            # Metrics
            metrics = pta.get_mask_metrics(mask_ref_r, x_r, metrics=metrics,
                                           pixel_spacing=pixel_spacing)
            # print(pidx)
            # pprint.pprint(metrics)
            metrics_model[pidx] = metrics
            metrics_model[pidx]['mode'] = dset_mode
        metrics_all.append(metrics_model)
    return metrics_all

def print_metrics_all(metrics_all, flag_percent=False, flag_met_col=False):
    """Print metrics"""
    try:
        import tabulate
    except ModuleNotFoundError:
        tabulate = None

    pat_percent_list = ['dice', 'jaccard', '_rate', 'accuracy', 'sensitivity',
                        'specificity', 'precision']
    f_pc = lambda c: flag_percent and any([re.search(pat, c)
                                           for pat in pat_percent_list])

    # Format table
    if len(metrics_all) > 1:
        f_list = [np.median, ]
        col_list = list(metrics_all[0].values())[0].keys()
        header = ['metric' if flag_met_col else 'model', ] + \
            [c + (' %' if f_pc(c) else '') for c in col_list
             for i in range(len(f_list))]
        # header_sub = ['', ] + [f.__name__ for c in col_list for f in f_list]
        table_met = [header, ]  # header_sub]
        for model, metrics in zip(model_list, metrics_all):
            row = [model[1], ]
            for met in col_list:
                m_data = [m[met] for m in metrics.values()]
                if f_pc(met):
                    sc = 100
                else:
                    sc = 1
                for f in f_list:
                    row.append('{:.2f}'.format(sc * f(m_data)))
            table_met.append(row)
        if flag_met_col:
            table_met = list(zip(*table_met))
    else:
        f_list = [np.mean, np.median, np.min, np.max, np.std]
        col_list = [f.__name__ for f in f_list]
        header = ['metric', ] + col_list
        for met in col_list:
            if met == list(col_list)[0]:
                table_met = [header]
                #table_met = [header]
                #print('******* ' + met + '\n')
            for model, metrics in zip(model_list, metrics_all):
                #row = [model[1], ]
                row = [met, ]
                m_data = [m[met] for m in metrics.values()]
                for f in f_list:
                    row.append('{:.2f}'.format(f(m_data)))
                table_met.append(row)
    if tabulate is not None:
        print(tabulate.tabulate(table_met, tablefmt='orgtbl',
                                floatfmt='.1f'))
    else:
        print('\n'.join(['|' + '|'.join([str(c) for c in r]) + '|'
                         for r in table_met]))
    print('')
    return table_met

def sort_pidx_list(pidx_list):
    return sorted(np.unique(pidx_list),
                   key=lambda x: (re.sub('[0-9]+', '', x.lower()),
                                  int(re.search('([0-9]+)', x).groups()[0])))

def dataframe_intersection(dataframes, by):
    set_index = [d.set_index(by) for d in dataframes]
    index_intersection = functools.reduce(pd.Index.intersection,
                                          [d.index for d in set_index])
    intersected = [df.loc[index_intersection].reset_index()
                   for df in set_index]
    return intersected

# %% Mask helper functions

def get_largest_region(img):
    img_l = spnd.measurements.label(img)[0]
    rp = skmeas.regionprops(img_l)
    if len(rp):
        mask_idx = np.argmax([r.area for r in rp])
        return img_l == mask_idx + 1
    return np.zeros(img.shape, dtype=np.bool)
# pydef-helpers ends here

def run_exp_prep_data(dfold_base, downsample_factor=1, mask_idx=None,
                      flag_comb=True, flag_mask_only=True, patch_params=None,
                      slice_window=None,
                      patient_idx_tr=None, patient_idx_tst=None,
                      fold_pre='', modality_list=['ct'], reader_list=[2],
                      flag_skip_write=False, fname_fmt=None, tfr_fold_fmt=None,
                      **kwargs):
    # kwargs used for unused parameters (e.g. kfold_idx to ensure different
    # .info files for different kfold sets)

    img_shape_load = [512 // downsample_factor, ] * 2
    img_shape_orig = copy.deepcopy(img_shape_load)
    if slice_window is not None:
        img_shape_orig.append(2 * slice_window + 1)
    img_shape = copy.deepcopy(img_shape_orig)

    # String representation
    if flag_mask_only is True:
        mask_only_str = ''
    elif flag_mask_only == 'balance':
        mask_only_str = '_mb'
    elif flag_mask_only == 'fully_contained_patch':
        mask_only_str = '_mfp'
    else:
        mask_only_str = '_ma'
    if mask_idx is None:
        if not flag_comb:
            comb_str = 'm'
        elif flag_comb is True:
            comb_str = 'c'
        else:
            comb_str = 'c{}'.format(flag_comb)
    else:
        comb_str = 'M{}'.format(mask_idx)
    if patch_params is not None:
        patch_str = re.sub(
            '[[], ]', '',
            '_patchN{patch_size}O{patch_overlap}'.format(**patch_params))
        img_shape = ptop.OperPatchPartitioner(
            img_shape_orig, **patch_params).element_size
        if slice_window is not None:
            img_shape = [img_shape[i] for i in [1, 2, 0]]
    else:
        patch_str = ''
    fname_fmt_params = {'mask_only_str': mask_only_str,
                        'comb_str': comb_str,
                        'flag_comb': flag_comb,
                        'mask_idx': mask_idx,
                        'patch_str': patch_str,
                        'modality_list_str': '_'.join(modality_list),
                        'slice_window': slice_window,
                        'patient_idx_tr': patient_idx_tr,
                        'patient_idx_tst': patient_idx_tst}
    if fname_fmt is None:
        fname_fmt = 'dnn0-{comb_str}{mask_only_str}_sl{patch_str}'
    if tfr_fold_fmt is None:
        tfr_fold_fmt = os.path.join(
            'AAPM', 'data_tfr_{modality_list_str}_comb{flag_comb}')

    # Prepare filenames
    fname_info = make_fname_info(
        os.path.join(dfold_base, fold_pre + '_' +
                     fname_fmt.format(**fname_fmt_params)),
        'tr', num_tfr=50, plist_tst=patient_idx_tst)

    tfr_fold = os.path.join(data_fold, tfr_fold_fmt.format(**fname_fmt_params))

    # Network output type
    if flag_comb is True:
        label_dtype = np.float32
    else:
        label_dtype = np.uint8

    # Create training dataset

    # Training data
    if reader_list is None:
        pidx_suff = ''
    else:
        pidx_suff = '_R' + ''.join([str(r) for r in reader_list])
    if not flag_skip_write:
        # Output folder
        for f in fname_info['tfr_fname_list_tr']:
            if os.path.isfile(f):
                os.unlink(f)
        if not os.path.isdir(fname_info['fold']):
            os.makedirs(fname_info['fold'])
        tfr_writer_list = [tf.python_io.TFRecordWriter(tfr_fname)
                           for tfr_fname in fname_info['tfr_fname_list_tr']]
        for pidx in patient_idx_tr:
            print('adding', pidx, '(training)')
            dnn0_add_patient_from_tfr(
                tfr_writer_list, tfr_fold, pidx + pidx_suff, label_dtype,
                img_shape=img_shape_load, downsample_factor=downsample_factor,
                flag_mask_only=flag_mask_only, patch_params=patch_params,
                slice_window=slice_window)
        for tfr_writer in tfr_writer_list:
            tfr_writer.close()
        # Write evaluation data
        for pidx in patient_idx_tst:
            print('adding', pidx, '(testing)')
            tfr_fname = fname_info['tfr_fname_list_tst'][pidx]
            tfr_writer = tf.python_io.TFRecordWriter(tfr_fname)
            dnn0_add_patient_from_tfr(
                [tfr_writer], tfr_fold, pidx + pidx_suff, label_dtype,
                img_shape=img_shape_load, downsample_factor=downsample_factor,
                flag_mask_only=flag_mask_only, patch_params=patch_params,
                slice_window=slice_window)
            tfr_writer.close()

    return {'label_dtype': label_dtype,
            'img_shape_load': img_shape_load,
            'img_shape': img_shape,
            'img_shape_orig': img_shape_orig,
            'fname_info': fname_info,
            'tfr_fold': tfr_fold,
            'pidx_suff': pidx_suff}
# pydef-helper-eval-func ends here

# [[file:gtv_segmentation.org::pydef-helper-eval-func_v2][pydef-helper-eval-func_v2]]
# %% Evaluation function

def run_exp_v2(name_fmt, dfold_base, train_params_list, eval_params,
               patient_idx_tr, patient_idx_tst, train_params_fname=None,
               modality_list=['ct', 'mr', 'pt'], data_scaling_params=None,
               tfr_tr_extra=None, fold_pre='ct-sts', flag_multi_gpu=False,
               flag_reset=False, flag_skip_train=False, flag_return_model=False):
    # Input parameters
    input_params = copy.deepcopy(locals())

    if not os.path.isdir(dfold_base):
        os.makedirs(dfold_base)

    # Default parameters
    data_params_defaults = {'patient_idx_tr': patient_idx_tr,
                            'patient_idx_tst': patient_idx_tst,
                            'fold_pre': fold_pre,
                            'data_scaling_params': data_scaling_params,
                            'modality_list': modality_list}
    train_params_defaults = {'n_start': 16,
                             'final_activation': 'sigmoid',
                             'flag_attention_gate': False,
                             'data_augment': True,
                             'lr': 1e-3,
                             'batch_size': 16,
                             'loss': 'mse',
                             'l2_reg': 0.1,
                             'dropout': None,
                             'num_steps': 100000}

    # Parameters of last training segment to form folder name
    run_prep_params_defaults = ptm.extract_defaults(run_exp_prep_data_v2,
                                                    ['flag_skip_write'])
    if train_params_fname is None:
        train_params_fname = train_params_list[-1]

    model_params_last = ptdict.merge(
        {'data_params': run_prep_params_defaults},
        {'data_params': data_params_defaults},
        train_params_defaults, train_params_fname)

    def conv_to_str(k, v):
        if isinstance(v, (dict, coll.OrderedDict)):
            return coll.namedtuple(k, v.keys())(**{kk: conv_to_str(kk, vv)
                                                   for kk, vv in v.items()})
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, str):
            return re.sub('\s+', '_', v)
        if isinstance(v, (list, tuple)):
            return '_'.join([str(vv) for vv in v])
        return v

    # Model folder
    model_fold = os.path.join(
        res_fold, dfold_base,
        'model_' + name_fmt.format(**{k: conv_to_str(k, v)
                                      for k, v in model_params_last.items()}))
    if len(train_params_list) > 1:
        model_fold += '_pre'

    if flag_reset:
        if os.path.isdir(model_fold):
            tf.summary.FileWriterCache.clear()
            shutil.rmtree(model_fold)
        if os.path.isfile(model_fold + '.info'):
            os.unlink(model_fold + '.info')
    if not os.path.isdir(model_fold):
        os.makedirs(model_fold)

    output = {'aux_info': {'model_fold': model_fold,
                           'input_params': input_params}}

    # Auxiliary model file
    if os.path.isfile(model_fold + '.info'):
        run_info = json.load(open(model_fold + '.info', 'rb'))
        run_info['train_params_list'].extend(train_params_list)
    else:
        run_info = {'input_params': input_params,
                    'train_params_list': train_params_list}
    if not flag_skip_train:
        json.dump(run_info, open(model_fold + '.info', 'wt'),
                  indent=2, sort_keys=True, default=ptdict.get_dict_default)

    for train_params_t in train_params_list:
        train_params = ptdict.merge(copy.deepcopy(train_params_defaults),
                                    train_params_t)

        # Prepare data
        data_params = ptdict.merge(run_prep_params_defaults,
                                   data_params_defaults,
                                   {'flag_skip_write': flag_skip_train,
                                    'loss': train_params['loss']},
                                   train_params['data_params'])
        data_fname = os.path.join(res_fold, dfold_base,
                                  'data_info_' + fold_pre + '{}')
        data_params_cache = copy.deepcopy(data_params)
        for k in ['flag_skip_write', ]:
            del data_params_cache[k]
        res_cache = ptio.ResultsCache(data_fname, data_params_cache)
        data_info = res_cache.run(
            'data_info', run_exp_prep_data_v2, dfold_base, **data_params)

        # Train

        # Parameters
        n_start = train_params['n_start']
        final_activation = getattr(tf.nn, train_params['final_activation'])
        data_augment = train_params['data_augment']
        flag_attention_gate = train_params.get('flag_attention_gate', False)
        flag_comb = data_params['flag_comb']
        l2_reg = train_params.get('l2_reg', 0.1)
        dropout = train_params.get('dropout', None)
        if flag_attention_gate or dropout is not None or flag_multi_gpu:
            raise RuntimeError('Options not supported')

        if n_start == 8:
            n_start_str = ''
        else:
            n_start_str = '_N{}'.format(n_start)
        if final_activation is None:
            fin_act_str = ''
        else:
            fin_act_str = '_act' + final_activation.__name__
        if flag_attention_gate:
            attg_str = '_attg'
        else:
            attg_str = ''
        spec_args_base = {'repeat_count': None,
                          'out_shape_img': data_info['img_shape'],
                          'out_shape_lbl': data_info['lbl_shape'],
                          'data_augment': data_augment,
                          'label_dtype': data_info['label_dtype']}

        lr = train_params['lr']
        batch_size = train_params['batch_size']
        loss = train_params['loss']
        num_steps = train_params['num_steps']

        spec_args = ptdict.merge(spec_args_base,
                                 {'batch_size': batch_size})
        if flag_multi_gpu:
            spec_args['flag_return_dataset'] = True

        model_params = {'learning_rate': lr, 'loss': loss,
                        'flag_comb': flag_comb,
                        'n_start': n_start,
                        'l2_reg': l2_reg,
                        'final_activation': final_activation,
                        'flag_attention_gate': flag_attention_gate,
                        'dropout': dropout}

        output.update(create_model_v2(data_info=data_info, **model_params))

        save_suff = datetime.datetime.strftime(datetime.datetime.now(),
                                               '%Y%m%d_%H%M%S')
        output['model_params'] = model_params
        output['data_info'] = data_info
        output['f_get_train_fname'] = lambda s=save_suff: {
            'train_info_fname': os.path.join(
                model_fold, 'train_info_{}.pik'.format(s)),
            'model_weights_fname': os.path.join(
                model_fold, 'model_weights_{}.h5'.format(s))}
        output['save_suff'] = save_suff
        for k in ['train_info_fname', 'model_weights_fname']:
            output[k] = output['f_get_train_fname'](save_suff)[k]

        if flag_return_model:
            return output
        if not flag_skip_train:
            dataset_tst = read_fn_v2(
                list(data_info['fname_info']['tfr_fname_list_tst'].values()),
                perform_shuffle=False, **spec_args)

            dataset_tr = read_fn_v2(
                data_info['fname_info']['tfr_fname_list_tr'],
                perform_shuffle=True, **spec_args)
            history = output['model'].fit(
                dataset_tr, batch_size=batch_size,
                epochs=num_steps,
                steps_per_epoch=int(np.ceil(data_info['num_tr_samples'] /
                                            batch_size)),
                verbose=1, validation_data=dataset_tst,
                validation_steps=int(np.ceil(data_info['num_tst_samples'] /
                                             batch_size)))
            pickle.dump(
                {'history': history.history,
                 'history_params': history.params,
                 'data_info': data_info,
                 'model_params': model_params,
                 'spec_args': spec_args},
                open(output['train_info_fname'],
                     'wb'), protocol=4)
            output['model'].save_weights(output['model_weights_fname'])

    # Last model parameters
    flag_comb = data_params['flag_comb']
    reader_list_eval = eval_params.get('reader_list_eval',
                                       data_params['reader_list'])
    loss = train_params['loss']

    # Evaluate
    patch_params_inf = eval_params.get('patch_params_inf', None)
    pidx_eval_list = eval_params.get(
        'pidx_eval_list', ['p2', 'p3', 'p4', 'p14', 'p16', 'STS_001',
                           'STS_010'])
    dfold_eval = eval_params.get('dfold', dfold)
    load_params = eval_params.get('load_params', {})

    for pidx in pidx_eval_list:
        print(pidx)
        X = pickle.load(open(
            os.path.join(data_info['tfr_fold'],
                         pidx + data_info['pidx_suff'] + '.pik'), 'rb'))
        img, lbl = data_info['f_preproc'](X['img'], X['lbl'])

        if data_params['patch_params'] is not None or \
           patch_params_inf is not None:
            if patch_params_inf is None:
                patch_params_inf = copy.deepcopy(data_params['patch_params'])
            patch_params_inf_img = copy.deepcopy(patch_params_inf)
            patch_params_inf_lbl = copy.deepcopy(patch_params_inf)
            for pp, x in zip([patch_params_inf_img, patch_params_inf_lbl],
                             [img, lbl]):
                pp['patch_size'] += [x.shape[-1], ]
                pp['patch_overlap'] += [0, ]
                pp['edge_mode'] = 'clip'
        else:
            raise RuntimeError('Inference without patches not implemented')
        op_patch_img = ptop.OperPatchPartitioner(
            img.shape, **patch_params_inf_img)
        op_patch_lbl = ptop.OperPatchPartitioner(
            lbl.shape, flag_precompute_invert_weights=True,
            **patch_params_inf_lbl)

        img_gen = functools.partial(op_patch_img, img)

        dataset_eval = tf.data.Dataset.from_generator(
            img_gen, output_signature=(tf.TensorSpec(
                shape=patch_params_inf_img['patch_size'])))
        dataset_eval = dataset_eval.batch(1)
        def generator_predict(d):
            for dd in d:
                yield output['model'].predict(dd)
        pred_gen = generator_predict(dataset_eval)

        img_pred = np.zeros(lbl.shape, dtype=np.float32)
        for m, idx in zip(pred_gen, op_patch_lbl.idx_gen):
            img_pred[np.ix_(*idx)] += m[0][np.ix_(*[np.arange(len(el))
                                                    for el in idx])]
        img_pred *= op_patch_lbl._invert_weights
        img_pred_p = img_pred
        out_fold = model_fold + '-inf'
        if not os.path.isdir(out_fold):
            os.makedirs(out_fold)
        out_fname = os.path.join(out_fold, 'data_{}_{}'.format(
            'tr' if pidx in patient_idx_tr else 'tst', pidx))
        save_inf(img, lbl, img_pred, img_pred_p, out_fname)

    return output

# %% Write TFrecords for training

def run_exp_prep_data_v2(dfold_base, mask_idx=None, flag_comb=True,
                         flag_mask_only=True, patch_params=None,
                         data_scaling_params=None, loss='MeanSquaredError',
                         patient_idx_tr=None, patient_idx_tst=None,
                         fold_pre='', modality_list=['ct'], reader_list=[2],
                         flag_skip_write=False, fname_fmt=None, tfr_fold_fmt=None,
                         **kwargs):
    # kwargs used for unused parameters (e.g. kfold_idx to ensure different
    # .info files for different kfold sets)

    img_shape_load = [1, 512, 512]
    img_shape_orig = copy.deepcopy(img_shape_load)
    img_shape = copy.deepcopy(img_shape_orig)

    # String representation
    if flag_mask_only is True:
        mask_only_str = ''
    elif flag_mask_only == 'balance':
        mask_only_str = '_mb'
    elif flag_mask_only == 'fully_contained_patch':
        mask_only_str = '_mfp'
    else:
        mask_only_str = '_ma'
    if mask_idx is None:
        if not flag_comb:
            comb_str = 'm'
        elif flag_comb is True:
            comb_str = 'c'
        else:
            comb_str = 'c{}'.format(flag_comb)
    else:
        comb_str = 'M{}'.format(mask_idx)
    if patch_params is not None:
        patch_str = re.sub(
            '[[], ]', '',
            '_patchN{patch_size}O{patch_overlap}'.format(**patch_params))
        img_shape = list(ptop.OperPatchPartitioner(
            img_shape_orig, **patch_params).element_size) + \
            [len(modality_list), ]
        lbl_shape = list(ptop.OperPatchPartitioner(
            img_shape_orig, **patch_params).element_size)
        if flag_comb is True:
            if loss in ['KLDivergence']:
                lbl_shape += [2, ]
            elif loss in ['MeanSquaredError']:
                lbl_shape += [1, ]
    else:
        patch_str = ''
    fname_fmt_params = {'mask_only_str': mask_only_str,
                        'comb_str': comb_str,
                        'flag_comb': flag_comb,
                        'mask_idx': mask_idx,
                        'patch_str': patch_str,
                        'modality_list_str': '_'.join(modality_list),
                        'patient_idx_tr': patient_idx_tr,
                        'patient_idx_tst': patient_idx_tst}
    if fname_fmt is None:
        fname_fmt = 'dnn1-{comb_str}{mask_only_str}_sl{patch_str}'
    if tfr_fold_fmt is None:
        tfr_fold_fmt = os.path.join(
            'AAPM', 'data_tfr_{modality_list_str}_comb{flag_comb}')

    # Prepare filenames
    fname_info = make_fname_info(
        os.path.join(dfold_base, fold_pre + '_' +
                     fname_fmt.format(**fname_fmt_params)),
        'tr', num_tfr=50, plist_tst=patient_idx_tst)

    tfr_fold = os.path.join(data_fold, tfr_fold_fmt.format(**fname_fmt_params))

    # Network output type
    if flag_comb is True:
        label_dtype = np.float32
    else:
        label_dtype = np.uint8

    # Data scaling
    def f_preproc(img, lbl):
        if data_scaling_params is not None:
            if data_scaling_params['method'] == 'sample-max':
                img /= np.amax(img, axis=(0, 1, 2))
            else:
                raise RuntimeError('Scaling method not supported')
        if flag_comb is True and loss in ['KLDivergence']:
            lbl = np.stack([1 - lbl, lbl], axis=-1)
        return img, lbl

    # Create training dataset

    # Training data
    if reader_list is None:
        pidx_suff = ''
    else:
        pidx_suff = '_R' + ''.join([str(r) for r in reader_list])
    if not flag_skip_write:
        # Output folder
        for f in fname_info['tfr_fname_list_tr']:
            if os.path.isfile(f):
                os.unlink(f)
        if not os.path.isdir(fname_info['fold']):
            os.makedirs(fname_info['fold'])
        tfr_writer_list = [tf.io.TFRecordWriter(tfr_fname)
                           for tfr_fname in fname_info['tfr_fname_list_tr']]
        for pidx in patient_idx_tr:
            print('adding', pidx, '(training)')
            X = pickle.load(open(
                os.path.join(tfr_fold, pidx + pidx_suff + '.pik'), 'rb'))
            img, lbl = f_preproc(X['img'], X['lbl'])
            add_tfr_img_v2(tfr_writer_list, img, lbl,
                           patch_params=patch_params)
        for tfr_writer in tfr_writer_list:
            tfr_writer.close()

        # Write evaluation data
        for pidx in patient_idx_tst:
            print('adding', pidx, '(testing)')
            tfr_fname = fname_info['tfr_fname_list_tst'][pidx]
            X = pickle.load(open(
                os.path.join(tfr_fold, pidx + pidx_suff + '.pik'), 'rb'))
            img, lbl = f_preproc(X['img'], X['lbl'])
            tfr_writer = tf.io.TFRecordWriter(tfr_fname)
            add_tfr_img_v2([tfr_writer], img, lbl,
                           patch_params=patch_params)
            tfr_writer.close()

    # Get total number of training samples
    num_tr_samples = sum(1 for i in tf.data.TFRecordDataset(
        fname_info['tfr_fname_list_tr']))

    # Get total number of testing samples
    tfr_list_tst = list(fname_info['tfr_fname_list_tst'].values())
    num_tst_samples = sum(
        1 for i in tf.data.TFRecordDataset(tfr_list_tst))

    return {'label_dtype': label_dtype,
            'img_shape_load': img_shape_load,
            'img_shape': img_shape,
            'lbl_shape': lbl_shape,
            'img_shape_orig': img_shape_orig,
            'fname_info': fname_info,
            'num_tr_samples': num_tr_samples,
            'num_tst_samples': num_tst_samples,
            'tfr_fold': tfr_fold,
            'pidx_suff': pidx_suff,
            'f_preproc': f_preproc}
# pydef-helper-eval-func_v2 ends here

# [[file:gtv_segmentation.org::pydef-helper-eval-func_v3][pydef-helper-eval-func_v3]]
# %% Folder names

LNET_PATH = '/home/code/220919-Lymphoma_TEP_CENIR'
if LNET_PATH not in sys.path:
    sys.path.append(LNET_PATH)
import sklearn.model_selection as skmod
import dill as dill
import network_setup as lnet
import python_tools.misc as ptmisc

# %% Data preparation

def prep_data(fold_info, data_params=None, model_params=None,
              training_params=None):

    # List input datasets
    pidx_list = [os.path.splitext(os.path.split(d.rstrip('/'))[-1])[0]
                 for d in glob.glob(
                         os.path.join(data_params['input_fold'], '*.pik'))]

    # Dataset list
    idx_all = np.arange(len(pidx_list))

    # Split into training, validation, testing
    if data_params['tst_frac'] > 0:
        idx_list_tr_val, idx_list_tst = skmod.train_test_split(
            idx_all, train_size=1 - data_params['tst_frac'],
            random_state=data_params['data_split_rseed'], shuffle=True)
    else:
        idx_list_tr_val = idx_all.copy()
        idx_list_tst = []
    idx_list_tr, idx_list_val = skmod.train_test_split(
        idx_all[idx_list_tr_val], train_size=1 - data_params['val_frac'],
        random_state=data_params['data_split_rseed'] + 1, shuffle=True)

    # Helper functions to load individual dataset
    def f_get_data(pidx):
        with open(os.path.join(data_params['input_fold'],
                               pidx + '.pik'), 'rb') as fid:
            data = pickle.load(fid)
        data_info = {k: {
            'data': scale_vol(data[k]['data'],
                              data_params['scale_params'].get(k, None)),
            'type': data[k]['type']}
                     for k in data_params['input_mod_list']}
        lbl_key = 'mask_comb_{}'.format(data_params['output_lbl'])
        data_info['lbl'] = {'data': data[lbl_key]['data'],
                            'type': data[lbl_key]['type']}
        mask = data_info['lbl']['data'] > 0
        return data_info, mask

    # Prepare TFRecords (use cache)
    write_tfr_params = {'patch_params': data_params['patch_params'],
                        'min_mask_frac': data_params['min_mask_frac'],
                        'flag_3d': data_params['flag_3d']}
    write_tfr_cache = ptio.ResultsCache(fold_info['data_fold'] + '_{}.pik',
                                        data_params)
    data_info = {'pidx_list': pidx_list}
    for idx, lbl in zip([idx_list_tr, idx_list_tr,
                         idx_list_val, idx_list_tst],
                        ['tr', 'tr_eval', 'val', 'tst']):
        if len(idx) > 0:
            data_info[lbl] = write_tfr_cache.run(
                'data_info_{}'.format(lbl), write_tfr_all,
                '{}_{}'.format(fold_info['data_fold'], write_tfr_cache.hash),
                idx, pidx_list, lbl, f_get_data,
                num_tfr_tr=data_params['num_tfr_tr'], flag_shuffle=lbl=='tr',
                write_tfr_params=write_tfr_params)
    data_info['img_shape'] = data_info['tr']['img_shape']
    data_info['feature_info'] = data_info['tr']['feature_info']

    # Datasets
    if model_params['loss'] in ['BinaryCrossentropy', ]:
        flag_one_hot = False
    elif model_params['loss'] in ['CategoricalCrossentropy', ]:
        flag_one_hot = True
    else:
        flag_one_hot = False
    data_info['flag_one_hot'] = flag_one_hot

    for idx, lbl in zip([idx_list_tr, idx_list_tr,
                         idx_list_val, idx_list_tst],
                        ['tr', 'tr_eval', 'val', 'tst']):
        if len(idx) > 0:
            data_info['dataset_{}'.format(lbl)] = (
                read_tfr(data_info[lbl]['tfr_fname_list'],
                         feature_info=data_info['feature_info'],
                         mod_list=data_params['input_mod_list'],
                         flag_shuffle=lbl=='tr',
                         flag_augment=lbl=='tr',
                         flag_one_hot=flag_one_hot)
                .batch(training_params['batch_size'])
                .prefetch(tf.data.AUTOTUNE))

    return data_info

# %% TFRecord writer

def write_tfr_all(data_fold, idx_list, pidx_list, lbl, f_get_data,
                  num_tfr_tr=10, flag_shuffle=False, write_tfr_params=None):
    if not os.path.isdir(data_fold):
        os.makedirs(data_fold)
    if flag_shuffle:
        tfr_fname_list = [
            os.path.join(data_fold, 'data_{}_{}.tfr'.format(lbl, i))
            for i in range(num_tfr_tr)]
    else:
        tfr_fname_list = [
            os.path.join(data_fold,
                         'data_{}_{}.tfr'.format(lbl, pidx_list[idx]))
            for idx in idx_list]

    # Training data
    tfr_list = [tf.io.TFRecordWriter(fname) for fname in tfr_fname_list]
    tfr_info_list = []
    for ii, idx in tqdm.tqdm(enumerate(idx_list), desc=lbl,
                             total=len(idx_list)):
        pidx = pidx_list[idx]
        data_info, mask = f_get_data(pidx)
        if flag_shuffle:
            tfr_list_in = tfr_list
        else:
            tfr_list_in = [tfr_list[ii], ]
        tfr_info_list.append(ptdl.write_tfr_patches(
            data_info, mask, tfr_list_in, **(write_tfr_params or {})))
    # Close files
    for tfr in tfr_list:
        tfr.close()

    num_channels = len([k for k in data_info.keys() if k not in ['lbl', ]])
    if not write_tfr_params.get('flag_3d', False):
        num_channels *= (write_tfr_params['patch_params']
                         or {}).get('patch_size', [1])[0]

    img_shape_orig = list(next(iter(data_info.values()))['data'].shape)
    img_shape = list(next(iter(
        tfr_info_list[0]['feature_info'].values()))['size'])[:-1] + \
        [num_channels, ]

    return {'idx_list': idx_list,
            'lbl': lbl,
            'tfr_fname_list': tfr_fname_list,
            'tfr_info_list': tfr_info_list,
            'img_shape_orig': img_shape_orig[:-1],
            'img_shape': img_shape,
            'feature_info': tfr_info_list[0]['feature_info']}


def scale_vol(image, scale_params):
    if scale_params is not None:
        if scale_params.startswith('scale'):
            scale_raw = scale_params.split(';')[1:]
            if len(scale_raw) == 1:
                scale_m = float(scale_raw[0])
            elif len(scale_raw) == 2:
                for si, s in enumerate(scale_raw):
                    try:
                        v = float(s)
                    except ValueError:
                        v = getattr(np, s)(image)
                    if si == 0:
                        scale_m = v
                    else:
                        scale_s = v
            image = (image - scale_m) / scale_s
    return image

def read_tfr(filenames, feature_info=None, mod_list=None, flag_shuffle=False,
             flag_augment=False, flag_one_hot=False):
    dataset = ptdl.read_tfr(filenames, feature_info)
    dataset = dataset.map(
        lambda d: (tf.concat([d[k] for k in mod_list], axis=-1), d['lbl']))

    if flag_one_hot:
        dataset = dataset.map(lambda x, y: (x, tf.concat([1 - y, y], axis=-1)))

    if flag_shuffle:
        dataset = dataset.shuffle(100)

    if flag_augment:
        dataset = dataset.map(Augment(),
                              num_parallel_calls=tf.data.AUTOTUNE)

    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

class Augment(tf.keras.layers.Layer):
    def __init__(self, seed=42):
        super().__init__()
        self.augment_inputs = tf.keras.Sequential([
            tf.keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=seed),
            tf.keras.layers.RandomRotation(factor=0.4, fill_mode="constant",
                                           fill_value=0, seed=seed),
            tf.keras.layers.RandomZoom(height_factor=(-0.0,-0.2),
                                       fill_mode='constant', fill_value=0,
                                       seed=seed)])
        self.augment_labels = tf.keras.Sequential([
            tf.keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=seed),
            tf.keras.layers.RandomRotation(factor=0.4, fill_mode="constant",
                                           fill_value=0, seed=seed),
            tf.keras.layers.RandomZoom(height_factor=(-0.0,-0.2),
                                       fill_mode='constant', fill_value=0,
                                       seed=seed)])
    def call(self, inputs, labels):
        # inputs = self.augment_inputs(inputs)
        # labels = self.augment_labels(labels)
        if tf.random.uniform(()) > 0.5:
            inputs = tf.image.flip_up_down(inputs)
            labels = tf.image.flip_up_down(labels)
        if tf.random.uniform(()) > 0.5:
            inputs = tf.image.flip_left_right(inputs)
            labels = tf.image.flip_left_right(labels)
        return inputs, labels

def dice_loss(y_true, y_pred, smooth=100):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    dice = (2.0 * intersection + smooth) / \
        (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return 1 - dice

def get_model(img_shape_slice, data_params=None, model_params=None,
              job_params=None):
    # Loss
    final_activation = None
    loss_args = {}
    training_metrics = []
    if model_params['loss'] in ['BinaryCrossentropy', 'MeanSquaredError']:
        if model_params['loss'] == 'Binarycrossentropy':
            loss_args['from_logits'] = True
            training_metrics.append('accuracy')
        num_classes_out = 1
    elif model_params['loss'] in ['CategoricalCrossentropy', 'KLDivergence']:
        if model_params['loss'] == 'CategoricalCrossentropy':
            loss_args['from_logits'] = True
        training_metrics.append('accuracy')
        num_classes_out = 2
    else:
        num_classes_out = 1
    loss = model_params['loss']
    if loss == 'dice_loss':
        training_loss = dice_loss
        final_activation = 'sigmoid'
    else:
        training_loss = getattr(keras.losses, loss)(**loss_args)

    # Input
    num_channels_in = len(data_params['input_mod_list'])
    if not data_params.get('flag_3d', False):
        num_channels_in *= (data_params['patch_params']
                            or {}).get('patch_size', [1])[0]

    def get_num_channels_out(loss):
        if model_params['loss'] in ['MeanSquaredError', 'BinaryCrossentropy',
                                    'dice_loss']:
            return 1
        if model_params['loss'] in ['KLDivergence', 'CategoricalCrossentropy']:
            return 2

    num_channels_out = get_num_channels_out(model_params['loss'])
    img_shape_in = list(img_shape_slice) + [num_channels_in, ]
    output_extra = {}
    if model_params['model_type'].startswith('unet:'):
        unet_params = ptdict.merge(
            {'num_filt_start': 8, 'kernel_size': (3, 3),
             'pool_size': [2, 2], 'depth': 5,
             'batch_norm': 'post', 'activation': 'relu'},
            ptdict.get_json_param(
                model_params['model_type'].split('unet:')[1]))

        inputs = keras.Input(shape=img_shape_in)
        outputs = ptdl.unet(inputs=inputs, num_channels_out=num_channels_out,
                            final_activation=final_activation, **unet_params)
        model = keras.Model(inputs=[inputs], outputs=[outputs])
        output_extra['inputs'] = inputs
        output_extra['outputs'] = outputs
    elif model_params['model_type'].startswith('dcl:'):
        unet_params = ptdict.merge(
            {'num_filt_start': 8, 'kernel_size': (3, 3),
             'pool_size': [2, 2], 'depth': 5,
             'batch_norm': 'post', 'activation': 'relu'},
            ptdict.get_json_param(
                model_params['model_type'].split('dcl:')[1]))

        inputs = keras.Input(shape=img_shape_in)
        outputs = ptdl.unet_deformable(inputs=inputs, num_channels_out=num_channels_out,
                            final_activation=final_activation, **unet_params)
        model = keras.Model(inputs=[inputs], outputs=[outputs])
        output_extra['inputs'] = inputs
        output_extra['outputs'] = outputs
    elif model_params['model_type'].startswith('swin:'):
        swin_params = ptdict.merge(
            {'filter_num_begin': 128, 'depth': 4,
             'stack_num_down': 2, 'stack_num_up': 2,
             'patch_size': (4, 4), 'num_heads': [4, 8, 8, 8],
             'window_size': [4, 2, 2, 2],
             'num_mlp': 512, 'shift_window': True},
            ptdict.get_json_param(
                model_params['model_type'].split('swin:')[1]))
        inputs = keras.Input(shape=img_shape_in)
        outputs = swin_unet_2d_base(inputs, **swin_params)
        outputs = tf.keras.layers.Conv2D(num_channels_out, kernel_size=1,
                                         use_bias=False)(outputs)
        if final_activation is not None:
            outputs = getattr(tf.nn, final_activation)(outputs)
        model = keras.Model(inputs=[inputs], outputs=[outputs])
        output_extra['inputs'] = inputs
        output_extra['outputs'] = outputs
    elif model_params['model_type'].startswith('transunet:'):
        tun_params = ptdict.merge(
            {'filter_num_begin': 128, 'depth': 4,
             'stack_num_down': 2, 'stack_num_up': 2,
             'patch_size': (4, 4), 'num_heads': [4, 8, 8, 8],
             'window_size': [4, 2, 2, 2],
             'num_mlp': 512, 'shift_window': True},
            ptdict.get_json_param(
                model_params['model_type'].split('transunet:')[1]))
        print(tun_params)
        inputs = keras.Input(shape=img_shape_in)
        model = ptdl.tun(inputs=inputs,
                         image_size=img_shape_in[0],
                         patch_size=tun_params['patch_size'],
                         depth=tun_params['depth'],
                         mlp_dim=tun_params['num_mlp'])

        output_extra['inputs'] = inputs
        #output_extra['outputs'] = outputs
    elif model_params['model_type'].startswith('sm:'):
        m_sm, m_type, m_params = model_params['model_type'].split(':', 2)
        params_defaults = {'backbone_name': 'vgg16',
                           'input_shape': img_shape_in,
                           'classes': num_channels_out,
                           'activation': None}
        net_params = ptdict.merge(
            params_defaults,
            {'Unet': {'weights': None,
                      'encoder_weights': 'imagenet',
                      'encoder_freeze': False,
                      'encoder_features': 'default',
                      'decoder_block_type': 'upsampling',
                      'decoder_filters': (128, 64, 32, 16, 8),
                      'decoder_use_batchnorm': True},
             'FPN': {},
             'Linknet': {'decoder_filters': (128, 64, 32, 16, 8)},
             'PSPNet': {}}[m_type],
            ptdict.get_json_param(m_params))
        model = getattr(sm, m_type)(**net_params)
        output_extra['inputs'] = model.inputs
        output_extra['outputs'] = model.outputs
    # Optimizer
    optimizer = getattr(keras.optimizers, model_params['optimizer'])(
        **(model_params['optimizer_params'] or {}))

    # Compile model
    model.compile(loss=training_loss, optimizer=optimizer,
                  metrics=training_metrics)

    return {'model': model,
            'training_loss': training_loss,
            'optimizer': optimizer,
            **output_extra}
# pydef-helper-eval-func_v3 ends here

# [[file:gtv_segmentation.org::pydef-helper-models_v3][pydef-helper-models_v3]]
# %% SwinUNET
# https://github.com/yingkaisha/keras-vision-transformer

KVT_PARENT_PATH = os.path.join('/home/software/src',
                               'python_packages/keras-vision-transformer')
if KVT_PARENT_PATH not in sys.path:
    sys.path.append(KVT_PARENT_PATH)

from keras_vision_transformer import swin_layers
from keras_vision_transformer import transformer_layers
from keras_vision_transformer import utils

def swin_transformer_stack(X, stack_num, embed_dim, num_patch, num_heads,
                           window_size, num_mlp, shift_window=True, name=''):
    '''
    Stacked Swin Transformers that share the same token size.

    Alternated Window-MSA and Swin-MSA will be configured if
    `shift_window=True`, Window-MSA only otherwise.

    *Dropout is turned off.
    '''
    # Turn-off dropouts
    # Droupout after each MLP layer
    mlp_drop_rate = 0
    # Dropout after Swin-Attention
    attn_drop_rate = 0
    # Dropout at the end of each Swin-Attention block, i.e., after linear
    # projections
    proj_drop_rate = 0
    # Drop-path within skip-connections
    drop_path_rate = 0

    # Convert embedded patches to query, key, and values with a learnable
    # additive value
    qkv_bias = True
    # None: Re-scale query based on embed dimensions per attention head # Float
    # for user specified scaling factor
    qk_scale = None

    if shift_window:
        shift_size = window_size // 2
    else:
        shift_size = 0

    for i in range(stack_num):

        if i % 2 == 0:
            shift_size_temp = 0
        else:
            shift_size_temp = shift_size

        X = swin_layers.SwinTransformerBlock(dim=embed_dim,
                                             num_patch=num_patch,
                                             num_heads=num_heads,
                                             window_size=window_size,
                                             shift_size=shift_size_temp,
                                             num_mlp=num_mlp,
                                             qkv_bias=qkv_bias,
                                             qk_scale=qk_scale,
                                             mlp_drop=mlp_drop_rate,
                                             attn_drop=attn_drop_rate,
                                             proj_drop=proj_drop_rate,
                                             drop_path_prob=drop_path_rate,
                                             name='name{}'.format(i))(X)
    return X


def swin_unet_2d_base(input_tensor, filter_num_begin, depth, stack_num_down,
                      stack_num_up, patch_size, num_heads, window_size,
                      num_mlp, shift_window=True, name='swin_unet'):
    '''
    The base of Swin-UNET.

    The general structure:

    1. Input image --> a sequence of patches --> tokenize these patches
    2. Downsampling: swin-transformer --> patch merging (pooling)
    3. Upsampling: concatenate --> swin-transfprmer --> patch expanding
       (unpooling)
    4. Model head

    '''
    # Compute number be patches to be embeded
    input_size = input_tensor.shape.as_list()[1:]
    num_patch_x = input_size[0]//patch_size[0]
    num_patch_y = input_size[1]//patch_size[1]

    # Number of Embedded dimensions
    embed_dim = filter_num_begin

    depth_ = depth

    X_skip = []

    X = input_tensor

    # Patch extraction
    X = transformer_layers.patch_extract(patch_size)(X)

    # Embed patches to tokens
    X = transformer_layers.patch_embedding(num_patch_x * num_patch_y,
                                           embed_dim)(X)

    # The first Swin Transformer stack
    X = swin_transformer_stack(X,
                               stack_num=stack_num_down,
                               embed_dim=embed_dim,
                               num_patch=(num_patch_x, num_patch_y),
                               num_heads=num_heads[0],
                               window_size=window_size[0],
                               num_mlp=num_mlp,
                               shift_window=shift_window,
                               name='{}_swin_down0'.format(name))
    X_skip.append(X)

    # Downsampling blocks
    for i in range(depth_-1):

        # Patch merging
        X = transformer_layers.patch_merging((num_patch_x, num_patch_y),
                                             embed_dim=embed_dim,
                                             name='down{}'.format(i))(X)

        # update token shape info
        embed_dim = embed_dim*2
        num_patch_x = num_patch_x//2
        num_patch_y = num_patch_y//2

        # Swin Transformer stacks
        X = swin_transformer_stack(X,
                                   stack_num=stack_num_down,
                                   embed_dim=embed_dim,
                                   num_patch=(num_patch_x, num_patch_y),
                                   num_heads=num_heads[i+1],
                                   window_size=window_size[i+1],
                                   num_mlp=num_mlp,
                                   shift_window=shift_window,
                                   name='{}_swin_down{}'.format(name, i+1))

        # Store tensors for concat
        X_skip.append(X)

    # reverse indexing encoded tensors and hyperparams
    X_skip = X_skip[::-1]
    num_heads = num_heads[::-1]
    window_size = window_size[::-1]

    # upsampling begins at the deepest available tensor
    X = X_skip[0]

    # other tensors are preserved for concatenation
    X_decode = X_skip[1:]

    depth_decode = len(X_decode)

    for i in range(depth_decode):

        # Patch expanding
        X = transformer_layers.patch_expanding(
            num_patch=(num_patch_x, num_patch_y), embed_dim=embed_dim,
            upsample_rate=2, return_vector=True)(X)

        # update token shape info
        embed_dim = embed_dim//2
        num_patch_x = num_patch_x*2
        num_patch_y = num_patch_y*2

        # Concatenation and linear projection
        X = tf.keras.layers.concatenate(
            [X, X_decode[i]], axis=-1,
            name='{}_concat_{}'.format(name, i))
        X = tf.keras.layers.Dense(
            embed_dim, use_bias=False,
            name='{}_concat_linear_proj_{}'.format(name, i))(X)

        # Swin Transformer stacks
        X = swin_transformer_stack(X,
                                   stack_num=stack_num_up,
                                   embed_dim=embed_dim,
                                   num_patch=(num_patch_x, num_patch_y),
                                   num_heads=num_heads[i],
                                   window_size=window_size[i],
                                   num_mlp=num_mlp,
                                   shift_window=shift_window,
                                   name='{}_swin_up{}'.format(name, i))

    # The last expanding layer; it produces full-size feature maps based on the
    # patch size
    # !!! <--- "patch_size[0]" is used; it assumes patch_size = (size, size)

    X = transformer_layers.patch_expanding(num_patch=(num_patch_x,
                                                      num_patch_y),
                                           embed_dim=embed_dim,
                                           upsample_rate=patch_size[0],
                                           return_vector=False)(X)

    return X

# %% segmentation_models

SM_PARENT_PATH = os.path.join('/home/software/src',
                              'python_packages/segmentation_models')
if SM_PARENT_PATH not in sys.path:
    sys.path.append(SM_PARENT_PATH)
import segmentation_models as sm
sm.set_framework('tf.keras')
sm.framework()
# pydef-helper-models_v3 ends here
