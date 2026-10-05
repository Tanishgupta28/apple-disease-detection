# Error analysis and Grad-CAM observations

These observations follow final test evaluation of the already selected ResNet50 fine-tuned checkpoint. They did not change the model, hyperparameters or checkpoint selection.

## Measured errors

The model correctly classified 1,954 of 2,013 test images: 97.0691% accuracy and 0.962691 macro F1. There were 59 errors. Counts below are bidirectional unless the direction is explicitly given.

| Disease pair | Errors in both directions | Breakdown |
|---|---:|---|
| Apple Scab / Frogeye Leaf Spot | 18 | Scab → Frogeye 14; Frogeye → Scab 4 |
| Apple Scab / Powdery Mildew | 12 | Scab → Mildew 8; Mildew → Scab 4 |
| Apple Scab / Healthy | 9 | Scab → Healthy 7; Healthy → Scab 2 |
| Apple Scab / Rust | 7 | Scab → Rust 1; Rust → Scab 6 |
| Rust / Frogeye Leaf Spot | 5 | Rust → Frogeye 0; Frogeye → Rust 5 |

Healthy has the lowest classwise F1 (0.930818), compared with Rust (0.984127). Counts reflect this dataset's support: Scab has 808 test images, whereas Healthy has 78 and Alternaria has 63. The smaller classes need additional independently collected examples before broad conclusions about field reliability.

## Confidently correct, confidently wrong, and uncertain examples

Figures and exact image paths/probabilities are in `results/figures/` and `results/error_examples.json`. The cases are selected mechanically: highest-confidence correct, highest-confidence wrong, and lowest-confidence below 0.6. They are illustrative examples, not new independent samples.

- `Scab (44).jpg` is labeled Scab but predicted Healthy at 0.999940 confidence. The pictured leaf is mostly green and the visible darker marks are subtle at the displayed scale. Weak visible symptoms are a plausible explanation; this is not evidence that its ground-truth label should be changed.
- `Rust (765).jpg` is labeled Rust but predicted Scab at 0.999914. It contains yellow-green discoloration and darker markings. Similar color/texture cues may contribute, but this observation alone cannot identify the cause of the model's error.
- `Frogeye leaf spot (1283).jpg` is predicted Rust at 0.999442. Two prominent rounded reddish-brown marks are visible near a leaf edge. Their appearance suggests overlapping visual cues with the Rust class; expert review would be needed to assess the source label or symptoms.
- `Scab (3892).jpg` is predicted Frogeye at 0.996009. Multiple small circular brown marks appear on the leaf. This is consistent with why spotting diseases might be confused, but the actual causal feature has not been established.
- The displayed Powdery Mildew → Scab and Frogeye → Mildew errors have variable lighting and background clutter. Symptom contrast may be difficult after resizing, but this is a hypothesis rather than a measured mechanism.

The least-confident shown case, `Health (324).jpg`, has top probability 0.289915 and is predicted Alternaria. The confidence warning helps flag this uncertainty, but it cannot flag every mistake: several wrong predictions exceed 99% confidence. Probabilities are not calibrated. Some correct predictions also have confidence below 0.6, so the threshold is a demo heuristic, not a diagnostic decision rule.

## Grad-CAM inspection

`gradcam_examples.png` contains the first saved test example of every true class, avoiding manual selection of only attractive heatmaps. It shows the original image, true class, predicted class, confidence, predicted-class heatmap and overlay.

Visual inspection of the generated figure:

- Scab: strongest activation overlaps a central discolored area of the leaf, with some activation extending toward its lower edge and surrounding scene.
- Rust: several highlighted regions overlap visible orange/yellow marks on the leaf. Activation also extends toward the lower-right leaf/background boundary.
- Frogeye: the hotspot overlaps a visible round brown mark on the central leaf; weaker activation reaches other scene regions.
- Powdery Mildew: a broad hotspot covers much of the visibly textured leaf rather than a precise lesion boundary.
- Alternaria: the strongest region is near the lower visible spot, with weaker activation extending toward the held leaf and surrounding area.
- Healthy: activation covers parts of leaves and the lower/background region; it is not restricted to an isolated leaf interior.

These examples support inspecting symptom regions, but **do not establish that every prediction depends only on lesions**. Grad-CAM is spatially coarse and class-associated; it is not segmentation, causal attribution or expert disease confirmation. Background shortcuts remain a concern, especially for Healthy and images with multiple leaves.

## Practical implications

Keep the held-out report fixed. Future work should use independent orchard/leaf-group test images, expert label review, higher-resolution lesion checks, validation-based calibration, and explicit unknown-image rejection. Those improvements require a new evaluation plan and cannot be credited to the completed benchmark.
