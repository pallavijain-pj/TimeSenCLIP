python  ./TimeSenCLIP/zeroshot.py \
            --dataset_path './Datasets/Sen4Map/datapub.fz-juelich.de/sen4map/split_wise/1x1_crops/test_with_eunis.h5' \
            --checkpoint './TimeSenCLIP/checkpoints/TimeSenCLIP.ckpt' \
            --input_resolution 1 \
            --BATCH_SIZE 1024 \
            --NUM_WORKERS 8 \
            --device 'cuda:0' \
            --version_fold 'test'\
            --ts_arch 'TimeSenCLIP'\
            --train_size 0.999 \
           