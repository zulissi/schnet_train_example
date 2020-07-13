#!/bin/bash

rm -rf /tmp/train_traj_files
rm -rf /tmp/val_traj_files
unzip -j -q ocp1k_adslab_train.zip checkpoint/mshuaibi/06_23_2020_ocpdata_10k/adslab_ref/filtered_train_b/* -d /tmp/train_traj_files/ 
unzip -j -q ocp1k_adslab_val.zip checkpoint/mshuaibi/06_23_2020_ocpdata_10k/adslab_ref/filtered_train_b/* -d /tmp/val_traj_files/ 

rm ocp1k_adslab_train.db
rm ocp1k_adslab_train_schnet.db
rm ocp1k_adslab_val.db
rm ocp1k_adslab_val_scnet.db

python convert_traj_to_schnet_db.py

