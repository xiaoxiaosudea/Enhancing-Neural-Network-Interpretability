"""This script runs the whole susu_v1_0.9922_ACE method."""


import sys
import os
import numpy as np
import sklearn.metrics as metrics
from tcav import utils
import tensorflow as tf

import susu_v1_ace_helpers
from susu_v1_ace import ConceptDiscovery
import argparse


def main(args):

  ###### related DIRs on CNS to store results #######
  # discovered_concepts_dir = os.path.join(args.working_dir, 'concepts/')
  discovered_concepts_dir = os.path.join(args.working_dir, args.target_class + '_concepts_patch')
  results_dir = os.path.join(args.working_dir, 'results/')
  cavs_dir = os.path.join(args.working_dir, 'cavs/')
  activations_dir = os.path.join(args.working_dir, 'acts/')
  results_summaries_dir = os.path.join(args.working_dir, 'results_summaries/')
  cav_dir_new = os.path.join(args.cav_dir_new, args.target_class)
  if not tf.gfile.Exists(cav_dir_new):
      tf.gfile.MakeDirs(cav_dir_new)
  if not tf.gfile.Exists(args.working_dir):
      tf.gfile.MakeDirs(args.working_dir)
  # if tf.gfile.Exists(args.working_dir):
  #   tf.gfile.DeleteRecursively(args.working_dir)
  tf.gfile.MakeDirs(args.working_dir)
  tf.gfile.MakeDirs(discovered_concepts_dir)
  tf.gfile.MakeDirs(results_dir)
  tf.gfile.MakeDirs(cavs_dir)
  tf.gfile.MakeDirs(activations_dir)
  tf.gfile.MakeDirs(results_summaries_dir)
  random_concept = 'random_discovery'  # Random concept for statistical testing
  sess = utils.create_session()
  mymodel = susu_v1_ace_helpers.make_model(
      sess, args.model_to_run, args.model_path, args.labels_path)
  # Creating the ConceptDiscovery class instance
  cd = ConceptDiscovery(
      mymodel,
      args.target_class,
      random_concept,
      args.bottlenecks.split(','),
      sess,
      args.source_dir,
      activations_dir,
      cavs_dir,
      cav_dir_new,
      num_random_exp=args.num_random_exp,
      channel_mean=True,    # 对于无监督概念发现，瓶颈激活在通道上平均，而不是使用整个激活向量（降低维数）
      max_imgs=args.max_imgs,
      min_imgs=args.min_imgs,
      num_discovery_imgs=args.max_imgs,
      num_workers=args.num_parallel_workers)
  # Creating the dataset of image patches
  cd.create_patches(param_dict={'n_segments': [5, 30, 80]})    # 每张图像被分割成15、50和80个片段
  # Saving the concept discovery target class images
  image_dir = os.path.join(discovered_concepts_dir, 'images')
  tf.gfile.MakeDirs(image_dir)
  susu_v1_ace_helpers.save_images(image_dir,
                            (cd.discovery_images * 255).astype(np.uint8))
  # Discovering Concepts
  cd.discover_concepts(method='KM', param_dicts={'n_clusters': 50})
  del cd.dataset  # Free memory
  del cd.image_numbers
  del cd.patches
  # Save discovered concept images (resized and original sized)
  print("111111111")
  susu_v1_ace_helpers.save_concepts(cd, discovered_concepts_dir)
  print("2222222222")
  # Calculating CAVs and TCAV scores
  cav_accuraciess = cd.cavs(min_acc=0.0)
  print("33333333333")
  print("cav_accuraciess:",cav_accuraciess)
  # scores = cd.tcavs(test=False)
  # print("33333333333")
  # susu_v1_ace_helpers.save_ace_report(cd, cav_accuraciess, scores,
  #                                results_summaries_dir + 'ace_results.txt')
  # print("33333333333")
  # # Plot examples of discovered concepts
  for bn in cd.bottlenecks:
    susu_v1_ace_helpers.plot_concepts(cd, bn, 40, address=results_dir)
  # # Delete concepts that don't pass statistical testing
  # print("33333333333")
  # cd.test_and_remove_concepts(scores)
  # print("33333333333")

def parse_arguments(argv):   # hammerhead_fore
  """Parses the arguments passed to the run.py script.分析传递给 run.py 脚本的参数"""
  parser = argparse.ArgumentParser()
  """保存网络类图像文件夹和随机概念文件夹的目录."""
  parser.add_argument('--source_dir', type=str,
      help='''Directory where the network's classes image folders and random
      concept folders are saved.''', default='D:\PaperCode\ACE-2\susu_new\ImageNet')
  parser.add_argument('--working_dir', type=str,
      help='Directory to save the results.', default='./ACE')
  parser.add_argument('--model_to_run', type=str,
      help='The name of the model.', default='GoogleNet')
  parser.add_argument('--model_path', type=str,
      help='Path to model checkpoints.', default='D:\PaperCode\ACE-2\susu_new\susu_V1_0.9766.pb')
  parser.add_argument('--labels_path', type=str,
      help='Path to model checkpoints.', default='D:\PaperCode\ACE-2\susu_new\imagenet_labels.txt')
  parser.add_argument('--target_class', type=str,
      help='The name of the target class to be interpreted', default='hammerhead_fore')
  parser.add_argument('--cav_dir_new', type=str,
      help='''Directory where the concepts are saved.''', default='D:\PaperCode\ACE-2\susu_new\CAV\hammerhead_fore')  # new added
  parser.add_argument('--bottlenecks', type=str,
      help='Names of the target layers of the network (comma separated)',
                      default='InceptionV1')
  """用于统计检验的随机实验数."""
  parser.add_argument('--num_random_exp', type=int,
      help="Number of random experiments used for statistical testing, etc",
                      default=20)
  """已发现概念中的最大图像数."""
  parser.add_argument('--max_imgs', type=int,
      help="Maximum number of images in a discovered concept",
                      default=25)
  parser.add_argument('--min_imgs', type=int,
      help="Minimum number of images in a discovered concept",
                      default=20)
  parser.add_argument('--num_parallel_workers', type=int,
      help="Number of parallel jobs.",
                      default=0)
  return parser.parse_args(argv)

def parse_arguments1(argv):   # hammerhead back
  """Parses the arguments passed to the run.py script.分析传递给 run.py 脚本的参数"""
  parser = argparse.ArgumentParser()
  """保存网络类图像文件夹和随机概念文件夹的目录."""
  parser.add_argument('--source_dir', type=str,
      help='''Directory where the network's classes image folders and random
      concept folders are saved.''', default='D:\PaperCode\ACE-2\susu_new\ImageNet')
  parser.add_argument('--working_dir', type=str,
      help='Directory to save the results.', default='./ACE')
  parser.add_argument('--model_to_run', type=str,
      help='The name of the model.', default='GoogleNet')
  parser.add_argument('--model_path', type=str,
      help='Path to model checkpoints.', default='D:\PaperCode\ACE-2\susu_new\susu_V1_0.9766.pb')
  parser.add_argument('--labels_path', type=str,
      help='Path to model checkpoints.', default='D:\PaperCode\ACE-2\susu_new\imagenet_labels.txt')
  parser.add_argument('--target_class', type=str,
      help='The name of the target class to be interpreted', default='hammerhead_back')
  parser.add_argument('--cav_dir_new', type=str,
      help='''Directory where the concepts are saved.''', default='D:\PaperCode\ACE-2\susu_new\CAV\hammerhead_back')  # new added
  parser.add_argument('--bottlenecks', type=str,
      help='Names of the target layers of the network (comma separated)',
                      default='InceptionV1')
  """用于统计检验的随机实验数."""
  parser.add_argument('--num_random_exp', type=int,
      help="Number of random experiments used for statistical testing, etc",
                      default=20)
  """已发现概念中的最大图像数."""
  parser.add_argument('--max_imgs', type=int,
      help="Maximum number of images in a discovered concept",
                      default=25)
  parser.add_argument('--min_imgs', type=int,
      help="Minimum number of images in a discovered concept",
                      default=20)
  parser.add_argument('--num_parallel_workers', type=int,
      help="Number of parallel jobs.",
                      default=0)
  return parser.parse_args(argv)

def parse_arguments2(argv):  # shark fore
  """Parses the arguments passed to the run.py script.分析传递给 run.py 脚本的参数"""
  parser = argparse.ArgumentParser()
  """保存网络类图像文件夹和随机概念文件夹的目录."""
  parser.add_argument('--source_dir', type=str,
      help='''Directory where the network's classes image folders and random
      concept folders are saved.''', default='D:\PaperCode\ACE-2\susu_new\ImageNet')
  parser.add_argument('--working_dir', type=str,
      help='Directory to save the results.', default='./ACE')
  parser.add_argument('--model_to_run', type=str,
      help='The name of the model.', default='GoogleNet')
  parser.add_argument('--model_path', type=str,
      help='Path to model checkpoints.', default='D:\PaperCode\ACE-2\susu_new\susu_V1_0.9766.pb')
  parser.add_argument('--labels_path', type=str,
      help='Path to model checkpoints.', default='D:\PaperCode\ACE-2\susu_new\imagenet_labels.txt')
  parser.add_argument('--target_class', type=str,
      help='The name of the target class to be interpreted', default='shark_fore')
  parser.add_argument('--cav_dir_new', type=str,
      help='''Directory where the concepts are saved.''', default='D:\PaperCode\ACE-2\susu_new\CAV\\shark_fore')  # new added
  parser.add_argument('--bottlenecks', type=str,
      help='Names of the target layers of the network (comma separated)',
                      default='InceptionV1')
  """用于统计检验的随机实验数."""
  parser.add_argument('--num_random_exp', type=int,
      help="Number of random experiments used for statistical testing, etc",
                      default=20)
  """已发现概念中的最大图像数."""
  parser.add_argument('--max_imgs', type=int,
      help="Maximum number of images in a discovered concept",
                      default=25)
  parser.add_argument('--min_imgs', type=int,
      help="Minimum number of images in a discovered concept",
                      default=20)
  parser.add_argument('--num_parallel_workers', type=int,
      help="Number of parallel jobs.",
                      default=0)
  return parser.parse_args(argv)

def parse_arguments3(argv):  # shark back
  """Parses the arguments passed to the run.py script.分析传递给 run.py 脚本的参数"""
  parser = argparse.ArgumentParser()
  """保存网络类图像文件夹和随机概念文件夹的目录."""
  parser.add_argument('--source_dir', type=str,
      help='''Directory where the network's classes image folders and random
      concept folders are saved.''', default='D:\PaperCode\ACE-2\susu_new\ImageNet')
  parser.add_argument('--working_dir', type=str,
      help='Directory to save the results.', default='./ACE')
  parser.add_argument('--model_to_run', type=str,
      help='The name of the model.', default='GoogleNet')
  parser.add_argument('--model_path', type=str,
      help='Path to model checkpoints.', default='D:\PaperCode\ACE-2\susu_new\susu_V1_0.9766.pb')
  parser.add_argument('--labels_path', type=str,
      help='Path to model checkpoints.', default='D:\PaperCode\ACE-2\susu_new\imagenet_labels.txt')
  parser.add_argument('--target_class', type=str,
      help='The name of the target class to be interpreted', default='shark_back')
  parser.add_argument('--cav_dir_new', type=str,
      help='''Directory where the concepts are saved.''', default='D:\PaperCode\ACE-2\susu_new\CAV\\shark_back')  # new added
  parser.add_argument('--bottlenecks', type=str,
      help='Names of the target layers of the network (comma separated)',
                      default='InceptionV1')
  """用于统计检验的随机实验数."""
  parser.add_argument('--num_random_exp', type=int,
      help="Number of random experiments used for statistical testing, etc",
                      default=20)
  """已发现概念中的最大图像数."""
  parser.add_argument('--max_imgs', type=int,
      help="Maximum number of images in a discovered concept",
                      default=25)
  parser.add_argument('--min_imgs', type=int,
      help="Minimum number of images in a discovered concept",
                      default=20)
  parser.add_argument('--num_parallel_workers', type=int,
      help="Number of parallel jobs.",
                      default=0)
  return parser.parse_args(argv)

def parse_arguments4(argv):  # tench fore
  """Parses the arguments passed to the run.py script.分析传递给 run.py 脚本的参数"""
  parser = argparse.ArgumentParser()
  """保存网络类图像文件夹和随机概念文件夹的目录."""
  parser.add_argument('--source_dir', type=str,
      help='''Directory where the network's classes image folders and random
      concept folders are saved.''', default='D:\PaperCode\ACE-2\susu_new\ImageNet')
  parser.add_argument('--working_dir', type=str,
      help='Directory to save the results.', default='./ACE')
  parser.add_argument('--model_to_run', type=str,
      help='The name of the model.', default='GoogleNet')
  parser.add_argument('--model_path', type=str,
      help='Path to model checkpoints.', default='D:\PaperCode\ACE-2\susu_new\susu_V1_0.9766.pb')
  parser.add_argument('--labels_path', type=str,
      help='Path to model checkpoints.', default='D:\PaperCode\ACE-2\susu_new\imagenet_labels.txt')
  parser.add_argument('--target_class', type=str,
      help='The name of the target class to be interpreted', default='tench_fore')
  parser.add_argument('--cav_dir_new', type=str,
      help='''Directory where the concepts are saved.''', default='D:\PaperCode\ACE-2\susu_new\CAV\\tench_fore')  # new added
  parser.add_argument('--bottlenecks', type=str,
      help='Names of the target layers of the network (comma separated)',
                      default='InceptionV1')
  """用于统计检验的随机实验数."""
  parser.add_argument('--num_random_exp', type=int,
      help="Number of random experiments used for statistical testing, etc",
                      default=20)
  """已发现概念中的最大图像数."""
  parser.add_argument('--max_imgs', type=int,
      help="Maximum number of images in a discovered concept",
                      default=25)
  parser.add_argument('--min_imgs', type=int,
      help="Minimum number of images in a discovered concept",
                      default=20)
  parser.add_argument('--num_parallel_workers', type=int,
      help="Number of parallel jobs.",
                      default=0)
  return parser.parse_args(argv)


def parse_arguments5(argv):  # tench back
  """Parses the arguments passed to the run.py script.分析传递给 run.py 脚本的参数"""
  parser = argparse.ArgumentParser()
  """保存网络类图像文件夹和随机概念文件夹的目录."""
  parser.add_argument('--source_dir', type=str,
      help='''Directory where the network's classes image folders and random
      concept folders are saved.''', default='D:\PaperCode\ACE-2\susu_new\ImageNet')
  parser.add_argument('--working_dir', type=str,
      help='Directory to save the results.', default='./ACE')
  parser.add_argument('--model_to_run', type=str,
      help='The name of the model.', default='GoogleNet')
  parser.add_argument('--model_path', type=str,
      help='Path to model checkpoints.', default='D:\PaperCode\ACE-2\susu_new\susu_V1_0.9766.pb')
  parser.add_argument('--labels_path', type=str,
      help='Path to model checkpoints.', default='D:\PaperCode\ACE-2\susu_new\imagenet_labels.txt')
  parser.add_argument('--target_class', type=str,
      help='The name of the target class to be interpreted', default='tench_back')
  parser.add_argument('--cav_dir_new', type=str,
      help='''Directory where the concepts are saved.''', default='D:\PaperCode\ACE-2\susu_new\CAV\\tench_back')  # new added
  parser.add_argument('--bottlenecks', type=str,
      help='Names of the target layers of the network (comma separated)',
                      default='InceptionV1')
  """用于统计检验的随机实验数."""
  parser.add_argument('--num_random_exp', type=int,
      help="Number of random experiments used for statistical testing, etc",
                      default=20)
  """已发现概念中的最大图像数."""
  parser.add_argument('--max_imgs', type=int,
      help="Maximum number of images in a discovered concept",
                      default=25)
  parser.add_argument('--min_imgs', type=int,
      help="Minimum number of images in a discovered concept",
                      default=20)
  parser.add_argument('--num_parallel_workers', type=int,
      help="Number of parallel jobs.",
                      default=0)
  return parser.parse_args(argv)


if __name__ == '__main__':
    # main(parse_arguments(sys.argv[1:]))
    # main(parse_arguments1(sys.argv[1:]))
    # main(parse_arguments2(sys.argv[1:]))
    # main(parse_arguments3(sys.argv[1:]))
    # main(parse_arguments4(sys.argv[1:]))
    main(parse_arguments5(sys.argv[1:]))
    # main(parse_arguments6(sys.argv[1:]))     # test
    # main(parse_arguments7(sys.argv[1:]))     # test
    # main(parse_arguments8(sys.argv[1:]))     # test
    # main(parse_arguments9(sys.argv[1:]))     # test
    # main(parse_arguments10(sys.argv[1:]))    # test
    # main(parse_arguments11(sys.argv[1:]))    # test
    # main(parse_arguments12(sys.argv[1:]))    # test
    # main(parse_arguments13(sys.argv[1:]))    # test


