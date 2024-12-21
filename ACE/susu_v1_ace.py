"""susu_pth_ACE library.

Library for discovering and testing concept activation vectors. It contains
ConceptDiscovery class that is able to discover the concepts belonging to one
of the possible classification labels of the classification task of a network
and calculate each concept's TCAV score..
"""
import pickle
from multiprocessing import dummy as multiprocessing
import sys
import os
import numpy as np
from PIL import Image
import scipy.stats as stats
import skimage.segmentation as segmentation
import sklearn.cluster as cluster
import sklearn.metrics.pairwise as metrics
import tensorflow as tf
from tcav import cav
from susu_v1_ace_helpers import *
import pandas


class ConceptDiscovery(object):
  """Discovering and testing concepts of a class.

  For a trained network, it first discovers the concepts as areas of the images
  in the class and then calculates the TCAV score of each concept. It is also
  able to transform images from pixel space into concept space.
  """

  def __init__(self,
               model,
               target_class,
               random_concept,
               bottlenecks,
               sess,
               source_dir,
               activation_dir,
               cav_dir,
               cav_dir_new,
               num_random_exp=2,
               channel_mean=True,
               max_imgs=40,
               min_imgs=20,
               num_discovery_imgs=40,
               num_workers=20,
               average_image_value=117):
    """Runs concept discovery for a given class in a trained model.

    For a trained classification model, the ConceptDiscovery class first
    performs unsupervised concept discovery using examples of one of the classes
    in the network.对于经过训练的分类模型，ConceptDiscovery 类首先使用网络中某个类的示例执行无监督的概念发现。

    Args:
      model: A trained classification model on which we run the concept
             discovery algorithm
      target_class: Name of the one of the classes of the network
      random_concept: A concept made of random images (used for statistical
                      test) e.g. "random500_199"
      bottlenecks: a list of bottleneck layers of the model for which the concept
                   discovery stage is performed 执行概念发现阶段的模型瓶颈层列表
      sess: Model's tensorflow session
      source_dir: This directory that contains folders with images of network's
                  classes. 此目录包含网络类图像的文件夹
      activation_dir: directory to save computed activations
      cav_dir: directory to save CAVs of discovered and random concepts
      num_random_exp: Number of random counterparts used for calculating several
                      CAVs and TCAVs for each concept (to make statistical
                        testing possible.)用于为每个概念计算多个CAV和TCAV的随机对应项的数量（以使统计测试成为可能。
      channel_mean: If true, for the unsupervised concept discovery the
                    bottleneck activations are averaged over channels instead
                    of using the whole activation vector (reducing
                    dimensionality)如果为 true，则对于无监督概念发现，瓶颈激活在通道上平均，而不是使用整个激活向量（降低维数)
      max_imgs: maximum number of images in a discovered concept
      min_imgs : minimum number of images in a discovered concept for the
                 concept to be accepted
      num_discovery_imgs: Number of images used for concept discovery. If None,
                          will use max_imgs instead.
      num_workers: if greater than zero, runs methods in parallel with
        num_workers parallel threads. If 0, no method is run in parallel
        threads.
      average_image_value: The average value used for mean subtraction in the
                           nework's preprocessing stage. 在 nework 的预处理阶段用于平均减法的平均值
    """
    self.model = model
    self.sess = sess
    self.target_class = target_class
    self.num_random_exp = num_random_exp  # 用于统计检验的随机实验数
    if isinstance(bottlenecks, str):
      bottlenecks = [bottlenecks]
    self.bottlenecks = bottlenecks
    self.source_dir = source_dir
    self.activation_dir = activation_dir
    self.cav_dir = cav_dir
    self.cav_dir_new = cav_dir_new
    self.channel_mean = channel_mean
    self.random_concept = random_concept
    self.image_shape = model.get_image_shape()[:2]   # image_shape？？？？
    self.max_imgs = max_imgs
    self.min_imgs = min_imgs
    if num_discovery_imgs is None:
      num_discovery_imgs = max_imgs
    self.num_discovery_imgs = num_discovery_imgs
    self.num_workers = num_workers
    self.average_image_value = average_image_value

  def load_concept_imgs(self, concept, max_imgs=1000):
    """Loads all colored images of a concept.

    Args:
      concept: The name of the concept to be loaded
      max_imgs: maximum number of images to be loaded

    Returns:
      Images of the desired concept or class.
    """
    concept_dir = os.path.join(self.source_dir, concept)
    img_paths = [
        os.path.join(concept_dir, d)
        for d in tf.gfile.ListDirectory(concept_dir)
    ]
    # print("concept_image_shape:", self.image_shape)
    return load_images_from_files(
        img_paths,
        max_imgs=max_imgs,
        return_filenames=False,
        do_shuffle=False,
        run_parallel=(self.num_workers > 0),
        shape=(self.image_shape),
        num_workers=self.num_workers)

  def create_patches(self, method='slic', discovery_images=None,
                     param_dict=None):
    """Creates a set of image patches using superpixel methods.

    This method takes in the concept discovery images and transforms it to a
    dataset made of the patches of those images.

    Args:
      method: The superpixel method used for creating image patches. One of
        'slic', 'watershed', 'quickshift', 'felzenszwalb',超像素分割方法，可选slic、watershed、quickshift、felzenszwalb四种方法.
      discovery_images: Images used for creating patches. If None, the images in
        the target class folder are used,用于创建图像块的图像。如果为None，则使用目标类文件夹中的图像.

      param_dict: Contains parameters of the superpixel method used in the form
                of {'param1':[a,b,...], 'param2':[z,y,x,...], ...}. For instance
                {'n_segments':[15,50,80], 'compactness':[10,10,10]} for slic
                method,包含超像素方法的参数，格式为{'param1':[a,b,...], 'param2':[z,y,x,...], ...}。
                例如，对于slic方法，可以设置{'n_segments':[15,50,80], 'compactness':[10,10,10]}.
      返回值：
      dataset：超像素分割后的数据集
      image_numbers：每个图像块所属的原始图像编号
      patches：每个图像块的图像数据
      解读：
      根据传入的参数，初始化函数中的一些变量，如param_dict，dataset，image_numbers，patches等。
      如果没有指定用于创建图像块的图像，则从目标类文件夹中加载指定数量的图像作为原始图像。
      根据是否开启了多进程，对原始图像进行超像素分割。
            如果开启了多进程，则使用multiprocessing.Pool创建进程池，调用_return_superpixels函数对每个原始图像进行超像素分割。
            否则，直接对每个原始图像进行超像素分割。
      将超像素分割后的图像块以及相关的信息记录到dataset，image_numbers，patches中。
      最后，将记录好的超像素分割后的数据集、每个图像块所属的原始图像编号、每个图像块的图像数据作为函数的返回值
    """
    if param_dict is None:
      param_dict = {}
    dataset, image_numbers, patches = [], [], []
    if discovery_images is None:
      raw_imgs = self.load_concept_imgs(
          self.target_class, self.num_discovery_imgs)
      self.discovery_images = raw_imgs
    else:
      self.discovery_images = discovery_images
    if self.num_workers:
      pool = multiprocessing.Pool(self.num_workers)
      outputs = pool.map(
          lambda img: self._return_superpixels(img, method, param_dict),
          self.discovery_images)
      for fn, sp_outputs in enumerate(outputs):
        image_superpixels, image_patches = sp_outputs
        for superpixel, patch in zip(image_superpixels, image_patches):
          dataset.append(superpixel)
          patches.append(patch)
          image_numbers.append(fn)
    else:
      for fn, img in enumerate(self.discovery_images):
        image_superpixels, image_patches = self._return_superpixels(
            img, method, param_dict)
        for superpixel, patch in zip(image_superpixels, image_patches):
          dataset.append(superpixel)
          patches.append(patch)
          image_numbers.append(fn)
    print("dataset:",len(dataset))
    self.dataset, self.image_numbers, self.patches =\
    np.array(dataset), np.array(image_numbers), np.array(patches)

  def _return_superpixels(self, img, method='slic',
                          param_dict=None):
    """Returns all patches for one image.

    Given an image, calculates superpixels for each of the parameter lists in
    param_dict and returns a set of unique superpixels by
    removing duplicates. If two patches have Jaccard similarity more than 0.5,
    they are concidered duplicates. 给定一个图像，计算param_dict中的每个参数列表的超级像素，
    并通过删除重复项返回一组唯一的超级像素。如果两个补丁的Jaccard相似度超过0.5，则它们是重复的。

    Args:
      img: The input image
      method: superpixel method, one of slic, watershed, quichsift, or
        felzenszwalb
      param_dict: Contains parameters of the superpixel method used in the form
                of {'param1':[a,b,...], 'param2':[z,y,x,...], ...}. For instance
                {'n_segments':[15,50,80], 'compactness':[10,10,10]} for slic
                method.
    Raises:
      ValueError: if the segementation method is invaled.
    """
    if param_dict is None:
      param_dict = {}
    if method == 'slic':
      n_segmentss = param_dict.pop('n_segments', [15, 50, 80])   # n_segments 分割块的个数。可能最后分割出的块数与实际设置并不一样，可能是slic算法做了后续处理，将小的超像素合并到大的超像素中。
      n_params = len(n_segmentss)
      compactnesses = param_dict.pop('compactness', [20] * n_params)  # compactness 割块的边界是否压缩，压缩会使分割快的边沿更光滑。
      sigmas = param_dict.pop('sigma', [1.] * n_params)     # sigma 图像每个维度进行预处理时的高斯平滑核宽。若给定为标量值，则同一个值运用到各个维度。0意味着不平滑。如果“sigma”是标量的，并且提供了手动体素间距，则自动缩放它（参见注释部分）。
    elif method == 'watershed':
      markerss = param_dict.pop('marker', [15, 50, 80])
      n_params = len(markerss)
      compactnesses = param_dict.pop('compactness', [0.] * n_params)
    elif method == 'quickshift':
      max_dists = param_dict.pop('max_dist', [20, 15, 10])
      n_params = len(max_dists)
      ratios = param_dict.pop('ratio', [1.0] * n_params)
      kernel_sizes = param_dict.pop('kernel_size', [10] * n_params)
    elif method == 'felzenszwalb':
      scales = param_dict.pop('scale', [1200, 500, 250])
      n_params = len(scales)
      sigmas = param_dict.pop('sigma', [0.8] * n_params)
      min_sizes = param_dict.pop('min_size', [20] * n_params)
    else:
      raise ValueError('Invalid superpixel method!')
    unique_masks = []
    for i in range(n_params):
      param_masks = []
      if method == 'slic':
        segments = segmentation.slic(
            img, n_segments=n_segmentss[i], compactness=compactnesses[i],
            sigma=sigmas[i])
      elif method == 'watershed':
        segments = segmentation.watershed(
            img, markers=markerss[i], compactness=compactnesses[i])
      elif method == 'quickshift':
        segments = segmentation.quickshift(
            img, kernel_size=kernel_sizes[i], max_dist=max_dists[i],
            ratio=ratios[i])
      elif method == 'felzenszwalb':
        segments = segmentation.felzenszwalb(
            img, scale=scales[i], sigma=sigmas[i], min_size=min_sizes[i])
      for s in range(segments.max()):
        mask = (segments == s).astype(float)
        if np.mean(mask) > 0.001:
          unique = True
          for seen_mask in unique_masks:
            jaccard = np.sum(seen_mask * mask) / np.sum((seen_mask + mask) > 0)
            if jaccard > 0.5:
              unique = False
              break
          if unique:
            param_masks.append(mask)
      unique_masks.extend(param_masks)
    superpixels, patches = [], []
    while unique_masks:
      superpixel, patch = self._extract_patch(img, unique_masks.pop())
      # superpixel, patch, superpixel_1 = self._extract_patch(img, unique_masks.pop())
      # print(superpixel.shape)
      # dd = 0
      # if np.all(superpixel == 0):
      #   continue
      # for x in range(superpixel.shape[0]):
      #   for y in range(superpixel.shape[1]):
      #     for z in range(superpixel.shape[2]):
      #       if superpixel[x][y][z] == 0:
      #         dd += 1
      # if dd / (superpixel.shape[0]*superpixel.shape[1]*superpixel.shape[2]) >= 0.8:
      #   continue
      superpixels.append(superpixel)
      patches.append(patch)
    return superpixels, patches

  def _extract_patch(self, image, mask):
    """Extracts a patch out of an image.

    Args:
      image: The original image
      mask: The binary mask of the patch area

    Returns:
      image_resized: The resized patch such that its boundaries touches the
        image boundaries
      patch: The original patch. Rest of the image is padded with average value
    """
    mask_expanded = np.expand_dims(mask, -1)
    patch = (mask_expanded * image + (
        1 - mask_expanded) * float(self.average_image_value) / 255)
    ones = np.where(mask == 1)
    h1, h2, w1, w2 = ones[0].min(), ones[0].max(), ones[1].min(), ones[1].max()
    image = Image.fromarray((patch[h1:h2, w1:w2] * 255).astype(np.uint8))
    image_resized = np.array(image.resize(self.image_shape,
                                          Image.BICUBIC)).astype(float) / 255
    # print(patch.shape)
    return image_resized, patch

  def _patch_activations(self, imgs, bottleneck, bs=100, channel_mean=None):
    """Returns activations of a list of imgs.

    Args:
      imgs: List/array of images to calculate the activations of
      bottleneck: Name of the bottleneck layer of the model where activations
        are calculated
      bs: The batch size for calculating activations. (To control computational
        cost)
      channel_mean: If true, the activations are averaged across channel.

    Returns:
      The array of activations
    """
    if channel_mean is None:
      channel_mean = self.channel_mean
    if self.num_workers:
      pool = multiprocessing.Pool(self.num_workers)
      output = pool.map(
          lambda i: self.model.run_examples(imgs[i * bs:(i + 1) * bs], bottleneck),
          np.arange(int(imgs.shape[0] / bs) + 1))
    else:
      output = []
      for i in range(int(imgs.shape[0] / bs) + 1):
        output.append(
            self.model.run_examples(imgs[i * bs:(i + 1) * bs], bottleneck))
    output = np.concatenate(output, 0)
    if channel_mean and len(output.shape) > 3:
      output = np.mean(output, (1, 2))
    else:
      output = np.reshape(output, [output.shape[0], -1])
    return output

  def _cluster(self, acts, method='KM', param_dict=None):
    """Runs unsupervised clustering algorithm on concept actiavtations.

    Args:
      acts: activation vectors of datapoints points in the bottleneck layer.
        E.g. (number of clusters,) for Kmeans
      method: clustering method. We have:
        'KM': Kmeans Clustering
        'AP': Affinity Propagation 近邻传播聚类
        'SC': Spectral Clustering  谱聚类
        'MS': Mean Shift clustering  均值漂移算法
        'DB': DBSCAN clustering method
      param_dict: Contains superpixl method's parameters. If an empty dict is
                 given, default parameters are used.

    Returns:
      asg: The cluster assignment label of each data points
      cost: The clustering cost of each data point
      centers: The cluster centers. For methods like Affinity Propagetion
      where they do not return a cluster center or a clustering cost, it
      calculates the medoid as the center  and returns distance to center as
      each data points clustering cost.

    Raises:
      ValueError: if the clustering method is invalid.
    """
    if param_dict is None:
      param_dict = {}
    centers = None
    if method == 'KM':
      n_clusters = param_dict.pop('n_clusters', 25)
      km = cluster.KMeans(n_clusters)
      d = km.fit(acts)
      centers = km.cluster_centers_
      d = np.linalg.norm(
          np.expand_dims(acts, 1) - np.expand_dims(centers, 0), ord=2, axis=-1)
      asg, cost = np.argmin(d, -1), np.min(d, -1)
    elif method == 'AP':
      damping = param_dict.pop('damping', 0.5)
      ca = cluster.AffinityPropagation(damping)
      ca.fit(acts)
      centers = ca.cluster_centers_
      d = np.linalg.norm(
          np.expand_dims(acts, 1) - np.expand_dims(centers, 0), ord=2, axis=-1)
      asg, cost = np.argmin(d, -1), np.min(d, -1)
    elif method == 'MS':
      ms = cluster.MeanShift(n_jobs=self.num_workers)
      asg = ms.fit_predict(acts)
    elif method == 'SC':
      n_clusters = param_dict.pop('n_clusters', 25)
      sc = cluster.SpectralClustering(
          n_clusters=n_clusters, n_jobs=self.num_workers)
      asg = sc.fit_predict(acts)
    elif method == 'DB':
      eps = param_dict.pop('eps', 0.5)
      min_samples = param_dict.pop('min_samples', 20)
      sc = cluster.DBSCAN(eps, min_samples, n_jobs=self.num_workers)
      asg = sc.fit_predict(acts)
    else:
      raise ValueError('Invalid Clustering Method!')
    if centers is None:  ## If clustering returned cluster centers, use medoids
      centers = np.zeros((asg.max() + 1, acts.shape[1]))
      cost = np.zeros(len(acts))
      for cluster_label in range(asg.max() + 1):
        cluster_idxs = np.where(asg == cluster_label)[0]
        cluster_points = acts[cluster_idxs]
        pw_distances = metrics.euclidean_distances(cluster_points)
        centers[cluster_label] = cluster_points[np.argmin(
            np.sum(pw_distances, -1))]
        cost[cluster_idxs] = np.linalg.norm(
            acts[cluster_idxs] - np.expand_dims(centers[cluster_label], 0),
            ord=2,
            axis=-1)
    return asg, cost, centers

  def discover_concepts(self,
                        method='KM',
                        activations=None,
                        param_dicts=None):
    """Discovers the frequent occurring concepts in the target class，根据目标类别发现频繁出现的概念. 重点！！！

      Calculates self.dic, a dictionary containing all the informations of the
      discovered concepts in the form of {'bottleneck layer name: bn_dic} where
      bn_dic itself is in the form of {'concepts:list of concepts,
      'concept name': concept_dic} where the concept_dic is in the form of
      {'images': resized patches of concept, 'patches': original patches of the
      concepts, 'image_numbers': image id of each patch}

    Args:
      method: Clustering method.
      activations: If activations are already calculated. If not calculates
                   them. Must be a dictionary in the form of {'bn':array, ...}
                   如果已经计算了激活值，则输入激活值。必须是一个字典，形式为{'bn':array, ...}
      param_dicts: A dictionary in the format of {'bottleneck':param_dict,...}
                   where param_dict contains the clustering method's parametrs
                   in the form of {'param1':value, ...}. For instance for Kmeans
                   {'n_clusters':25}. param_dicts can also be in the format
                   of param_dict where same parameters are used for all
                   bottlenecks.
                   一个字典，格式为{'bottleneck': param_dict, ...}，其中param_dict包含聚类方法的参数，格式为{'param1':value, ...}。
                   例如，对于Kmeans {'n_clusters': 25}。param_dicts也可以是param_dict的形式，其中相同的参数用于所有瓶颈。
      输出：无返回值，但计算了self.dic，一个包含所有发现的概念信息的字典，格式为{'bottleneck layer name: bn_dic}，
           其中bn_dic本身的格式为{'concepts:list of concepts, 'concept name': concept_dic}，
           其中concept_dic的格式为{'images': resized patches of concept, 'patches': original patches of the concepts,
                            'image_numbers': image id of each patch}
    """
    if param_dicts is None:
      param_dicts = {}
    if set(param_dicts.keys()) != set(self.bottlenecks):
      param_dicts = {bn: param_dicts for bn in self.bottlenecks}
    self.dic = {}  ## The main dictionary of the ConceptDiscovery class.
    i = 0
    for bn in self.bottlenecks:
      i = i + 1
      bn_dic = {}
      if activations is None or bn not in activations.keys():
        bn_activations = self._patch_activations(self.dataset, bn)
      else:
        bn_activations = activations[bn]
      print(f"bn_activation_{i}:{bn_activations}")
      print(f"bn_activation_{i}.size:{bn_activations.shape}")
      bn_dic['label'], bn_dic['cost'], centers = self._cluster(
          bn_activations, method, param_dicts[bn])
      concept_number, bn_dic['concepts'] = 0, []
      for i in range(bn_dic['label'].max() + 1):
        label_idxs = np.where(bn_dic['label'] == i)[0]
        if len(label_idxs) > self.min_imgs:
          concept_costs = bn_dic['cost'][label_idxs]
          concept_idxs = label_idxs[np.argsort(concept_costs)[:self.max_imgs]]
          concept_image_numbers = set(self.image_numbers[label_idxs])
          discovery_size = len(self.discovery_images)
          highly_common_concept = len(
              concept_image_numbers) > 0.5 * len(label_idxs)
          mildly_common_concept = len(
              concept_image_numbers) > 0.25 * len(label_idxs)
          mildly_populated_concept = len(
              concept_image_numbers) > 0.25 * discovery_size
          cond2 = mildly_populated_concept and mildly_common_concept
          non_common_concept = len(
              concept_image_numbers) > 0.1 * len(label_idxs)
          highly_populated_concept = len(
              concept_image_numbers) > 0.5 * discovery_size
          cond3 = non_common_concept and highly_populated_concept
          if highly_common_concept or cond2 or cond3:
            concept_number += 1
            concept = '{}_concept{}'.format(self.target_class, concept_number)
            bn_dic['concepts'].append(concept)
            bn_dic[concept] = {
                'images': self.dataset[concept_idxs],
                'patches': self.patches[concept_idxs],
                'image_numbers': self.image_numbers[concept_idxs]
            }
            bn_dic[concept + '_center'] = centers[i]
      bn_dic.pop('label', None)
      bn_dic.pop('cost', None)
      self.dic[bn] = bn_dic

  def _random_concept_activations(self, bottleneck, random_concept):
    """Wrapper for computing or loading activations of random concepts.

    Takes care of making, caching (if desired) and loading activations.

    Args:
      bottleneck: The bottleneck layer name
      random_concept: Name of the random concept e.g. "random500_0"

    Returns:
      A nested dict in the form of {concept:{bottleneck:activation}}
    """
    rnd_acts_path = os.path.join(self.activation_dir, 'acts_{}_{}'.format(
        random_concept, bottleneck))
    if not tf.gfile.Exists(rnd_acts_path):
      rnd_imgs = self.load_concept_imgs(random_concept, self.max_imgs)
      acts = get_acts_from_images(rnd_imgs, self.model, bottleneck)
      with tf.gfile.Open(rnd_acts_path, 'w') as f:
        np.save(f, acts, allow_pickle=False)
      del acts
      del rnd_imgs
    return np.load(rnd_acts_path).squeeze()

  def _calculate_cav(self, c, r, bn, act_c, ow, i, j,directory=None):
    """Calculates a single cav for a concept and a one random counterpart.

    Args:
      c: concept name
      r: random concept name
      bn: the layer name
      act_c: activation matrix of the concept in the 'bn' layer   “BN”层中概念的激活矩阵
      ow: overwrite if CAV already exists
      directory: to save the generated CAV

    Returns:
      The accuracy of the CAV
    """
    if directory is None:
      directory = self.cav_dir
    act_r = self._random_concept_activations(bn, r)
    cav_instance = cav.get_or_train_cav([c, r],
                                        bn, {
                                            c: {
                                                bn: act_c
                                            },
                                            r: {
                                                bn: act_r
                                            }
                                        },
                                        i=i,
                                        j=j,
                                        cav_dir_new=self.cav_dir_new,
                                        cav_dir=directory,
                                        overwrite=ow,
                                        )
    return cav_instance.accuracies['overall']

  def _concept_cavs(self, bn, concept, activations, randoms=None, ow=True, i=0):
    """Calculates CAVs of a concept versus all the random counterparts.

    Args:
      bn: bottleneck layer name
      concept: the concept name
      activations: activations of the concept in the bottleneck layer
      randoms: None if the class random concepts are going to be used
      ow: If true, overwrites the existing CAVs

    Returns:
      A dict of cav accuracies in the form of {'bottleneck layer':
      {'concept name':[list of accuracies], ...}, ...}
    """
    i = i
    j = 0
    if randoms is None:
      randoms = [
          'random500_{}'.format(i) for i in np.arange(self.num_random_exp)
      ]
    if self.num_workers:
      # i += 1
      pool = multiprocessing.Pool(20)
      accs = pool.map(
          lambda rnd: self._calculate_cav(concept, rnd, bn, activations, ow, i=i, j=j),
          randoms)
    else:
      accs = []
      for rnd in randoms:
        j += 1
        accs.append(self._calculate_cav(concept, rnd, bn, activations, ow, i=i, j=j))
    return accs

  def cavs(self, min_acc=0., ow=True):
    """Calculates cavs for all discovered concepts.

    This method calculates and saves CAVs for all the discovered concepts
    versus all random concepts in all the bottleneck layers

    Args:
      min_acc: Delete discovered concept if the average classification accuracy
        of the CAV is less than min_acc
      ow: If True, overwrites an already calcualted cav.

    Returns:
      A dicationary of classification accuracy of linear boundaries orthogonal
      to cav vectors
      该函数是ConceptDiscovery类的一个方法，用于计算所有已发现概念的CAV（concept activation vector）向量。
      参数min_acc表示CVA向量分类精度的下限，ow表示是否覆盖已经计算好的CAV向量。
      该函数首先对于每个瓶颈层和每个已发现的概念，它获取该概念的图像并计算对应的激活值，
      然后调用内部的_concept_cavs方法计算该概念与随机概念之间的CAV向量，并保存到文件中。
      同时，如果该概念的平均分类精度小于min_acc，则从概念列表中删除该概念。
      然后，对于每个瓶颈层，它获取目标类别和随机类别的激活值，并计算出对应的CAV向量，并保存到文件中。
      最后，该方法返回一个字典，包含所有CAV向量的分类精度
    """
    acc = {bn: {} for bn in self.bottlenecks}
    concepts_to_delete = []
    i = 0
    print("self.bootlenecks_number:", len(self.bottlenecks))
    for bn in self.bottlenecks:
      for concept in self.dic[bn]['concepts']:
        i = i + 1
        concept_imgs = self.dic[bn][concept]['images']
        concept_acts = get_acts_from_images(
            concept_imgs, self.model, bn)
        acc[bn][concept] = self._concept_cavs(bn, concept, concept_acts, ow=ow, i=i)


        # path = os.path.join('D:\PaperCode\ACE-1\susu_new\ImageNet', 'concept.csv')
        # if not os.path.exists(path):
        #   data1 = pandas.DataFrame(acc[bn][concept])
        #   data1.to_csv(path)
        # else:
        #   data2 = pandas.read_csv(path)
        #   new_data = pandas.DataFrame(acc[bn][concept])
        #   result = pandas.concat([data2, new_data], ignore_index=True)
        #   result.to_csv(path)
        # np.savez(path, *acc[bn][concept][:])
        # for i in range(len(acc[bn][concept])):
          # path = os.path.join('D:\PaperCode\ACE-1\susu_new\ImageNet', 'concept.npz')
          # np.savez(path, arri=acc[bn][concept][i])
        if np.mean(acc[bn][concept]) < min_acc:
          concepts_to_delete.append((bn, concept))
      target_class_acts = get_acts_from_images(
          self.discovery_images, self.model, bn)
      acc[bn][self.target_class] = self._concept_cavs(
          bn, self.target_class, target_class_acts, ow=ow)
      rnd_acts = self._random_concept_activations(bn, self.random_concept)
      acc[bn][self.random_concept] = self._concept_cavs(
          bn, self.random_concept, rnd_acts, ow=ow)
    for bn, concept in concepts_to_delete:
      self.delete_concept(bn, concept)

    return acc

  def load_cav_direction(self, c, r, bn, directory=None):
    """Loads an already computed cav.

    Args:
      c: concept name
      r: random concept name
      bn: bottleneck layer
      directory: where CAV is saved

    Returns:
      The cav instance
    """
    if directory is None:
      directory = self.cav_dir
    params = tf.contrib.training.HParams(model_type='linear', alpha=.01)
    cav_key = cav.CAV.cav_key([c, r], bn, params.model_type, params.alpha)
    cav_path = os.path.join(self.cav_dir, cav_key.replace('/', '.') + '.pkl')
    vector = cav.CAV.load_cav(cav_path).cavs[0]
    # print("vector_end:",vector)
    # print("vector_end.size:", vector.shape)
    return np.expand_dims(vector, 0) / np.linalg.norm(vector, ord=2)

  def _sort_concepts(self, scores):
    for bn in self.bottlenecks:
      tcavs = []
      for concept in self.dic[bn]['concepts']:
        tcavs.append(np.mean(scores[bn][concept]))
      concepts = []
      for idx in np.argsort(tcavs)[::-1]:
        concepts.append(self.dic[bn]['concepts'][idx])
      self.dic[bn]['concepts'] = concepts

  def _return_gradients(self, images):
    """For the given images calculates the gradient tensors.

    Args:
      images: Images for which we want to calculate gradients.

    Returns:
      A dictionary of images gradients in all bottleneck layers.
    """

    gradients = {}
    class_id = self.model.label_to_id(self.target_class.replace('_', ' '))
    for bn in self.bottlenecks:
      acts = get_acts_from_images(images, self.model, bn)
      bn_grads = np.zeros((acts.shape[0], np.prod(acts.shape[1:])))
      for i in range(len(acts)):
        bn_grads[i] = self.model.get_gradient(
            acts[i:i+1], [class_id], bn, example=None).reshape(-1)
      gradients[bn] = bn_grads
    return gradients

  def _tcav_score(self, bn, concept, rnd, gradients):
    """Calculates and returns the TCAV score of a concept.

    Args:
      bn: bottleneck layer
      concept: concept name
      rnd: random counterpart
      gradients: Dict of gradients of tcav_score_images

    Returns:
      TCAV score of the concept with respect to the given random counterpart
    """
    vector = self.load_cav_direction(concept, rnd, bn)
    print("vector_end:",vector)
    print("vector_end.size:", vector.shape)
    prod = np.sum(gradients[bn] * vector, -1)
    return np.mean(prod < 0)

  def tcavs(self, test=False, sort=True, tcav_score_images=None):
    """Calculates TCAV scores for all discovered concepts and sorts concepts.

    This method calculates TCAV scores of all the discovered concepts for
    the target class using all the calculated cavs. It later sorts concepts
    based on their TCAV scores.

    Args:
      test: If true, perform statistical testing and removes concepts that don't
        pass
      sort: If true, it will sort concepts in each bottleneck layers based on
        average TCAV score of the concept.
      tcav_score_images: Target class images used for calculating tcav scores.
        If None, the target class source directory images are used.

    Returns:
      A dictionary of the form {'bottleneck layer':{'concept name':
      [list of tcav scores], ...}, ...} containing TCAV scores.
    """

    tcav_scores = {bn: {} for bn in self.bottlenecks}
    randoms = ['random500_{}'.format(i) for i in np.arange(self.num_random_exp)]
    if tcav_score_images is None:  # Load target class images if not given
      raw_imgs = self.load_concept_imgs(self.target_class, 2 * self.max_imgs)
      tcav_score_images = raw_imgs[-self.max_imgs:]
    gradients = self._return_gradients(tcav_score_images)
    print("tcav")
    for bn in self.bottlenecks:
      for concept in self.dic[bn]['concepts'] + [self.random_concept]:
        def t_func(rnd):
          return self._tcav_score(bn, concept, rnd, gradients)
        if self.num_workers:
          pool = multiprocessing.Pool(self.num_workers)
          tcav_scores[bn][concept] = pool.map(lambda rnd: t_func(rnd), randoms)
        else:
          print("wewewe")
          tcav_scores[bn][concept] = [t_func(rnd) for rnd in randoms]
          print("wewewe11111")
    if test:
      self.test_and_remove_concepts(tcav_scores)
    if sort:
      self._sort_concepts(tcav_scores)
    return tcav_scores

  def do_statistical_testings(self, i_ups_concept, i_ups_random):
    """Conducts ttest to compare two set of samples.

    In particular, if the means of the two samples are staistically different.

    Args:
      i_ups_concept: samples of TCAV scores for concept vs. randoms
      i_ups_random: samples of TCAV scores for random vs. randoms

    Returns:
      p value
    """
    min_len = min(len(i_ups_concept), len(i_ups_random))
    _, p = stats.ttest_rel(i_ups_concept[:min_len], i_ups_random[:min_len])
    return p

  def test_and_remove_concepts(self, tcav_scores):
    """Performs statistical testing for all discovered concepts.

    Using TCAV socres of the discovered concepts versurs the random_counterpart
    concept, performs statistical testing and removes concepts that do not pass

    Args:
      tcav_scores: Calculated dicationary of tcav scores of all concepts
    """
    concepts_to_delete = []
    for bn in self.bottlenecks:
      for concept in self.dic[bn]['concepts']:
        pvalue = self.do_statistical_testings\
        (tcav_scores[bn][concept], tcav_scores[bn][self.random_concept])
        if pvalue > 0.01:
          concepts_to_delete.append((bn, concept))
    for bn, concept in concepts_to_delete:
      self.delete_concept(bn, concept)

  def delete_concept(self, bn, concept):
    """Removes a discovered concepts if it's not already removed.

    Args:
      bn: Bottleneck layer where the concepts is discovered.
      concept: concept name
    """
    self.dic[bn].pop(concept, None)
    if concept in self.dic[bn]['concepts']:
      self.dic[bn]['concepts'].pop(self.dic[bn]['concepts'].index(concept))

  def _concept_profile(self, bn, activations, concept, randoms):
    """Transforms data points from activations space to concept space.

    Calculates concept profile of data points in the desired bottleneck
    layer's activation space for one of the concepts

    Args:
      bn: Bottleneck layer
      activations: activations of the data points in the bottleneck layer
      concept: concept name
      randoms: random concepts

    Returns:
      The projection of activations of all images on all CAV directions of
        the given concept
    """
    def t_func(rnd):
      products = self.load_cav_direction(concept, rnd, bn) * activations
      return np.sum(products, -1)
    if self.num_workers:
      pool = multiprocessing.Pool(self.num_workers)
      profiles = pool.map(lambda rnd: t_func(rnd), randoms)
    else:
      profiles = [t_func(rnd) for rnd in randoms]
    return np.stack(profiles, axis=-1)

  def find_profile(self, bn, images, mean=True):
    """Transforms images from pixel space to concept space.

    Args:
      bn: Bottleneck layer
      images: Data points to be transformed
      mean: If true, the profile of each concept would be the average inner
        product of all that concepts' CAV vectors rather than the stacked up
        version.

    Returns:
      The concept profile of input images in the bn layer.
    """
    profile = np.zeros((len(images), len(self.dic[bn]['concepts']),
                        self.num_random_exp))
    class_acts = get_acts_from_images(
        images, self.model, bn).reshape([len(images), -1])
    randoms = ['random500_{}'.format(i) for i in range(self.num_random_exp)]
    for i, concept in enumerate(self.dic[bn]['concepts']):
      profile[:, i, :] = self._concept_profile(bn, class_acts, concept, randoms)
    if mean:
      profile = np.mean(profile, -1)
    return profile

