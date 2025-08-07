python  /home/pallavi/DATA/Granular-Project/CLIP_RS_TS_Project/TimeSenCLIP/zeroshot.py \
            --dataset_path '/home/pallavi/DATA/Datasets/Sen4Map/datapub.fz-juelich.de/sen4map/split_wise/1x1_crops/test_with_eunis.h5' \
            --checkpoint '/home/pallavi/DATA/Granular-Project/checkpoints/SenCLIP_TS/V2.0_SenMSTS_tsdrop_temp007_attpool_normonlysen4map_TimeSpectralViT_clip_loss_adamw_0.0001_1024/TimeSpectralViT_tsdrop.ckpt' \
            --input_resolution 1 \
            --BATCH_SIZE 1024 \
            --NUM_WORKERS 8 \
            --device 'cuda:0' \
            --version_fold 'test'\
            --ts_arch 'TimeSenCLIP'\
            --train_size 0.999 \
           