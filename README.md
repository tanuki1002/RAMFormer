# RAMFormer

～～～介紹模型～～～

## Environment
The experiments were conducted on Ubuntu 22.04 with CUDA 12.1.
> A Conda environment can be created to run the model.
```
conda env create -f /path/uda_env.yaml -n <name>
```

> or alternatively, use Docker.
```
待補～～～
```

## Execution
Before training, category statistics file for the clear training data should be generated.
```
python -m tools.count_categories <path/to/your/categories.csv> <path/to/your/training_images> <path/to/your/training_labels> <path/to/your/output_file>
```
> For example,
```
python -m tools.count_categories data/csv/rlmd.csv data/rlmd_ac/clear/train/images data/rlmd_ac/clear/train/labels data/rcs_files/rlmd.json
```
Start training.
> Config file: configs/train_uda_multi_tasks.json
```
python -m tools.train_uda_multitask configs/train_uda_multi_tasks.json 
```
Resume training from where it was interrupted.
```
python -m tools.train_uda_multitask \
    configs/train_uda_multi_tasks.json \
    logs/checkpoint_last.pth
```

## Inference
Single-image inference.
```
```
> For example,
```
python -m tools.inference_multitask \
    --config configs/train_uda_multi_tasks.json \
    --checkpoint logs//train_uda_multi_tasks_20251129194529/best_model_iter1800.pth \
    --input /home/rvl/MinHsuan/dataset/temp/b1c9c847-3bda4659.jpg \
    --task da
```


Inference on a folder of images.
```
```
> For example,
```
python -m tools.inference_multitask \
    --config configs/train_uda_multi_tasks.json \
    --checkpoint logs/your_experiment/best_model.pth \
    --input /path/to/test_images_folder \
    --output output_vis \
    --task rm
```

If you want a semi-transparent overlay effect, add the --opacity 0.5 parameter.
> python -m tools.inference ... --opacity 0.5

## Analysis
```
tensorboard --logdir logs/train_uda_multi_tasks_XXX/
```
