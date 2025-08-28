python train.py \
  dataset_name=nuplan \
  model_name=ldm3d_image \
  ldm.model.autoencoder_run_name=scenario_dreamer_autoencoder3d_nuplan \
  ldm.train.run_name=train_ldm_firsttry \
  ldm.train.devices=3 \
  ldm.datamodule.train_batch_size=4 \
  ldm.datamodule.val_batch_size=4 \
  ldm.model.num_l2l_blocks=3 \
  ldm.train.track=True