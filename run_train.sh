#!/bin/sh

python train.py --train_path '/home/yan/datasets/celeb/train' --gallery_path '/home/yan/datasets/celeb/gallery' --query_path '/home/yan/datasets/celeb/query' --logs-dir 'log_celeb_all'

python train.py --train_path '/home/yan/datasets/celeb/train_1_1' --gallery_path '/home/yan/datasets/celeb/gallery_1_1' --query_path '/home/yan/datasets/celeb/query_1_1' --logs-dir 'log_celeb_11' --epochs 50 --step_size 40

python train.py --train_path '/home/yan/datasets/celeb/train_1_2' --gallery_path '/home/yan/datasets/celeb/gallery_1_2' --query_path '/home/yan/datasets/celeb/query_1_2' --logs-dir 'log_celeb_12' --epochs 50 --step_size 40

python train.py --train_path '/home/yan/datasets/celeb/train_1_3' --gallery_path '/home/yan/datasets/celeb/gallery_1_3' --query_path '/home/yan/datasets/celeb/query_1_3' --logs-dir 'log_celeb_13' --epochs 50 --step_size 40

python train.py --train_path '/home/yan/datasets/celeb/train_2_1' --gallery_path '/home/yan/datasets/celeb/gallery_2_1' --query_path '/home/yan/datasets/celeb/query_2_1' --logs-dir 'log_celeb_21' --epochs 50 --step_size 40

python train.py --train_path '/home/yan/datasets/celeb/train_2_2' --gallery_path '/home/yan/datasets/celeb/gallery_2_2' --query_path '/home/yan/datasets/celeb/query_2_2' --logs-dir 'log_celeb_22' --epochs 50 --step_size 40
