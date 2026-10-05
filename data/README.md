# AppleLeaf9 dataset

Official source: https://github.com/JasonYangCode/AppleLeaf9

License: **Creative Commons Attribution 4.0 International (CC BY 4.0)**. The source archive's LICENSE.txt is preserved locally. Attribution is required when sharing images or derived sample figures.

Citation: Yang, Qing; Duan, Shukai; Wang, Lidan (2022). *Efficient Identification of Apple Leaf Diseases in the Wild Using Convolutional Neural Networks*. Agronomy, 12(11), 2784. https://doi.org/10.3390/agronomy12112784

The original dataset fuses PlantVillage, ATLDSD, and Plant Pathology challenge sources. The download contains 14,582 images. This project selects Scab, Rust, Frogeye leaf spot, Powdery mildew, Alternaria leaf spot, and Health; Brown spot, Grey spot, and Mosaic are excluded to keep the requested five-disease-plus-healthy classification scope. Health is displayed as Healthy, and Scab as Apple Scab.

Raw data live under `data/raw/AppleLeaf9-main/`. No raw data are committed. `data/processed/manifest.csv` is the immutable split; symlinks expose train/validation/test class directories without storing a second image copy. Training reads the manifest. The split membership is also preserved as a lightweight CSV in `results/split_manifest.csv` for auditing.

See `results/dataset_audit.json` for measured class counts, readability, dimensions, removed label conflicts, hash evidence, and split sizes. Related pHash groups, including rotations/reflections, stay within one split. Exact byte and decoded-pixel checks are included. There are no acquisition/leaf/orchard group IDs, so this audit cannot guarantee that every related scene or source augmentation was identified.

Download and prepare from the project root:

```bash
python scripts/download_dataset.py
python scripts/prepare_data.py
```

Preparation refuses to overwrite an existing manifest. Use a new configured processed directory for a deliberately new benchmark. Augmentation is applied on demand only to training images after the split, never to validation or test.
