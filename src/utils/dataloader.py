def load_data(args):
    print("Script Launched....\nLoading Data....")
    start_time = time.time()

    train_dataset, val_dataset, classes = dataset_sen4map(
        args.h5data_train_path, args.sen_path, args.emb_path,
        args.h5data_val_path,
        crop_size=args.crop_size,
        annual_composite=args.time_frames > 1,
        channels=args.channels,
        resize=args.resize,
        device=args.device,
    )

    
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    print(f"Ground Embedding Shape: {train_dataset[0][0].shape} | Sentinel Image Shape: {train_dataset[0][1].shape}")

    train_loader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=True, drop_last=True,
                              num_workers=args.NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=8)

    val_loader = DataLoader(val_dataset, batch_size=args.BATCH_SIZE, shuffle=True, drop_last=True,
                            num_workers=args.NUM_WORKERS, pin_memory=True, persistent_workers=True, prefetch_factor=8)

    print(f"Data Loaded in {(time.time() - start_time)/60:.2f} minutes.")
    return train_loader, val_loader, classes