model=dict(
    CLOCs=dict(
        # Loss
        loss=dict(weighted_sigmoid_focal=dict(
          alpha=0.15,
          gamma=3.0))),
        
        # Outputs
        use_sigmoid_score=True,
        encode_background_as_zeros=True,
        encode_rad_error_by_sin=True,

        # Loss
        pos_class_weight=1.0,
        neg_class_weight=1.0,

        loss_norm_type='NormByNumPositives',
    
        # Postprocess
        post_center_limit_range=[0, -40, -3.0, 70.4, 40, 0.0],
        use_rotate_nms=True,
        use_multi_class_nms=False,
        nms_pre_max_size=2000,
        nms_post_max_size=20,
        nms_score_threshold=0.5,
        nms_iou_threshold=0.1,

        use_bev=False,
        num_point_features=4,
        without_reflectivity=False,

        box_coder=dict(
            ground_box3d_coder=dict(
                linear_dim=False,
                encode_angle_vector=False)))


train_input_reader=dict(
    used_classes=["Car"],
    max_num_epochs=160,
    batch_size=1,
    prefetch_size=25,
    num_workers=3,
    groundtruth_localization_noise_std=[1.0, 1.0, 0.5],
    # groundtruth_rotation_uniform_noise: [-0.3141592654, 0.3141592654]
    # groundtruth_rotation_uniform_noise: [-1.57, 1.57]
    groundtruth_rotation_uniform_noise=[-0.78539816, 0.78539816],
    global_rotation_uniform_noise=[-0.78539816, 0.78539816],
    global_scaling_uniform_noise=[0.95, 1.05],
    global_random_rotation_range_per_object=[0, 0], # pi/4 ~ 3pi/4
    remove_points_after_sample=True,
    groundtruth_points_drop_percentage=0.0,
    groundtruth_drop_max_keep_points=15,
    database_sampler=dict(
        database_info_path="KITTI_DATASET_ROOT/kitti_dbinfos_train.pkl",
        sample_groups=dict(
          name_to_max_num=dict(key="Car", value=15)),
        database_prep_steps=dict(
            filter_by_min_num_points=dict(
                min_num_point_pairs=dict(key="Car", value=5)),
            filter_by_difficulty=dict(
                removed_difficulties=[-1])),
        global_random_rotation_range_per_object=[0, 0],
        rate=1.0),
    remove_unknown_examples=False,
    remove_environment=False,
    kitti_info_path="KITTI_DATASET_ROOT/kitti_infos_train.pkl",
    kitti_root_path="KITTI_DATASET_ROOT")

train_config=dict(
  optimizer=dict(
      adam_optimizer=dict(
          learning_rate=dict(
              one_cycle=dict(
                  lr_max=3e-3,  # original 3e-3
                  moms=[0.95, 0.85],
                  div_factor=10.0, #original 10
                  pct_start=0.4)),
          weight_decay=0.01), # super converge. decrease this when you increase steps. og 0.01
      fixed_weight_decay=True,
      use_moving_average=False),
  steps=3712*10,   #112215 #113715 #111360 # 619 * 50, super converge. increase this to achieve slightly better results original 30950
  steps_per_eval=3712,   #7481 # 619 * 5
  save_checkpoints_secs=1800,   # half hour 1800
  save_summary_steps=100,
  enable_mixed_precision=False,  # for fp16 training, don't use this.
  loss_scale_factor=512.0,
  loss_function=dict(
      gamma=2.0,
      alpha=0.25),
  clear_metrics_every_epoch=True,
  detection_2d_path="d2_detection_data",
  detection_3d_path="d3_detection_data",)

eval_input_reader=dict(
    batch_size=1,
    max_num_epochs=160,
    prefetch_size=25,
    shuffle_points=False,
    num_workers=3,
    remove_environment=False,
    kitti_info_path="KITTI_DATASET_ROOT/kitti_infos_val.pkl",
    kitti_root_path="KITTI_DATASET_ROOT")


test_input_reader=dict(
    batch_size=1,
    max_num_epochs=160,
    prefetch_size=25,
    shuffle_points=False,
    num_workers=3,
    remove_environment=False,
    kitti_info_path="KITTI_DATASET_ROOT/kitti_infos_test.pkl",
    kitti_root_path="KITTI_DATASET_ROOT")