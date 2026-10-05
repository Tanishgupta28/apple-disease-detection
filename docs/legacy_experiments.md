# Historical NIT internship experiments

These notebooks are historical work, not the upgraded six-class leaf benchmark. No old notebook was rerun. Evidence below is recovered from source cells and stored outputs; cell numbers are zero-based.

## Recovery and dataset

All nine publicly listed Drive files were downloaded, kept byte-for-byte under `original_work/Apple disease detection codes/`, and validated with nbformat. SHA-256 hashes and sizes are in `original_manifest.json`. No dataset, standalone checkpoint, or additional project file was present in the public folder listing.

Every notebook has stored generator outputs confirming **307 training, 75 validation, and 120 test images** (502 total), four classes. Source paths are `/kaggle/input/appledisease/Train` and `/kaggle/input/appledisease/Test`. Stored file listings identify Blotch_Apple, Rot_Apple, Scab_Apple, and Normal_Apple. There is no recoverable source URL/license or original image archive. The labels and filenames do not establish that the dataset consists exclusively of leaves; this cannot be verified without its images.

## Comparison

| Notebook | Last logged train accuracy by phase* | Maximum logged validation accuracy by phase | Genuine stored test accuracy |
|---|---|---|---|
| apple-disease-dectection-cnn (1).ipynb | 0.6579 | 0.2267 | Not stored |
| apple-disease-detection-convnext-insp-transform.ipynb | 0.3525 / 0.4798 | 0.6000 / 0.6800 | Test Accuracy: 0.5750 |
| apple-disease-detection-densenet121-finetuned.ipynb | 0.5818 / 0.8249 | 0.7200 / 0.8533 | Not stored |
| apple-disease-detection-efficientnetb3-finetuned.ipynb | 0.8232 / 0.9061 | 0.9600 / 0.9600 | Not stored |
| apple-disease-detection-inceptionv3-finetuned.ipynb | 0.6932 / 0.8919 | 0.8400 / 0.8933 | Not stored |
| apple-disease-detection-mobilenetv2-finetuned.ipynb | 0.8564 / 0.9003 | 0.9200 / 0.8667 | Not stored |
| apple-disease-detection-resnet50-finetuned (1).ipynb | 0.7668 / 0.7983 | 0.9200 / 0.9067 | Not stored |
| apple-disease-detection-xception-finetuned.ipynb | 0.7723 / 0.8641 | 0.8667 / 0.8933 | Not stored |
| ensemble-apple-disease-detection.ipynb | 0.3170 / 0.8626 / 0.3154 | 0.4000 / 0.9200 / 0.3067 | 🏆 ENSEMBLE ACCURACY: 73.33% |

## apple-disease-dectection-cnn (1).ipynb

File: `original_work/Apple disease detection codes/apple-disease-dectection-cnn (1).ipynb`. No embedded PNG figures. No stored precision/recall/F1 report. Imported plotting libraries alone do not establish plotted results. No written conclusions were stored.

### Training evidence

[
  {
    "cell": 14,
    "logged_epochs": 11,
    "last_train": "0.6579",
    "max_val": 0.2267,
    "last_val": "0.2267"
  }
]

### Printed summaries

Not stored.

### Architecture, preprocessing, weights, fine-tuning, optimizer and callback source

Cell 1:
```python
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from collections import Counter
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.optimizers import SGD
```

Cell 3:
```python
img_size = (128, 128)
batch_size = 32
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=25,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)
```

Cell 7:
```python
model = Sequential()


model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(128,128,3)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))


model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))


model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))


model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(train_gen.num_classes, activation='softmax'))

```

Cell 8:
```python
optimizer = Adam(learning_rate=0.0005)
loss_fn = CategoricalCrossentropy()
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
```

Cell 11:
```python
lr_reduce = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.5)
```

Cell 12:
```python
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
```

Cell 13:
```python
labels_from_gen = train_gen.classes

class_weight = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels_from_gen),
    y=labels_from_gen
)

class_weight = dict(enumerate(class_weight))
```

Cell 14:
```python
training = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=[early_stop, lr_reduce],
    class_weight=class_weight
)
```

Cell 17:
```python
test_loss, test_acc = model.evaluate(test_gen, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")
```


## apple-disease-detection-convnext-insp-transform.ipynb

File: `original_work/Apple disease detection codes/apple-disease-detection-convnext-insp-transform.ipynb`. No embedded PNG figures. No stored precision/recall/F1 report. Imported plotting libraries alone do not establish plotted results. No written conclusions were stored.

### Training evidence

[
  {
    "cell": 14,
    "logged_epochs": 10,
    "last_train": "0.3525",
    "max_val": 0.6,
    "last_val": "0.5733"
  },
  {
    "cell": 19,
    "logged_epochs": 12,
    "last_train": "0.4798",
    "max_val": 0.68,
    "last_val": "0.6800"
  }
]

### Printed summaries

- Cell 21: Best Training Accuracy: 0.5505; Best Validation Accuracy: 0.6800

### Architecture, preprocessing, weights, fine-tuning, optimizer and callback source

Cell 1:
```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import ConvNeXtBase
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image
import os
```

Cell 3:
```python
img_size = (224, 224)
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=25,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)
```

Cell 4:
```python
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)
```

Cell 9:
```python
labels = train_gen.classes
class_weight = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weight = dict(enumerate(class_weight))
```

Cell 10:
```python
base_model = ConvNeXtBase(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3))
base_model.trainable = False
```

Cell 11:
```python
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(train_gen.num_classes, activation='softmax'))
```

Cell 12:
```python
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='categorical_crossentropy', metrics=['accuracy'])
```

Cell 13:
```python
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, verbose=1)
```

Cell 14:
```python
history1=model.fit(train_gen, 
          validation_data=val_gen, 
          epochs=10,
          class_weight=class_weight, 
          callbacks=[early_stop, lr_reduce]
         )
```

Cell 15:
```python
print("\nPhase 2: Fine-tuning ConvNeXt")
base_model.trainable = True
```

Cell 16:
```python
for layer in base_model.layers[:200]:
    layer.trainable = False
```

Cell 17:
```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

Cell 18:
```python
early_stop2 = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
lr_reduce2 = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)
```

Cell 19:
```python
history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=12,
    class_weight=class_weight,
    callbacks=[early_stop2, lr_reduce2]
)
```

Cell 22:
```python
test_loss, test_acc = model.evaluate(test_gen, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")
```


## apple-disease-detection-densenet121-finetuned.ipynb

File: `original_work/Apple disease detection codes/apple-disease-detection-densenet121-finetuned.ipynb`. No embedded PNG figures. No stored precision/recall/F1 report. Imported plotting libraries alone do not establish plotted results. No written conclusions were stored.

### Training evidence

[
  {
    "cell": 14,
    "logged_epochs": 10,
    "last_train": "0.5818",
    "max_val": 0.72,
    "last_val": "0.7200"
  },
  {
    "cell": 16,
    "logged_epochs": 10,
    "last_train": "0.8249",
    "max_val": 0.8533,
    "last_val": "0.8533"
  }
]

### Printed summaries

- Cell 18: Best Training Accuracy: 0.7883; Best Validation Accuracy: 0.8533
- Cell 19: Best Validation Accuracy: 0.8533333539962769

### Architecture, preprocessing, weights, fine-tuning, optimizer and callback source

Cell 1:
```python
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
```

Cell 3:
```python
img_size = (224, 224)
batch_size = 32
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=25,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)
```

Cell 4:
```python
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)
```

Cell 9:
```python
labels_from_gen = train_gen.classes
class_weight = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels_from_gen),
    y=labels_from_gen
)
class_weight = dict(enumerate(class_weight))
```

Cell 10:
```python
lr_reduce = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
```

Cell 11:
```python
print('phase1...training and freezing')
base_model = DenseNet121(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
base_model.trainable = False
```

Cell 12:
```python
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(train_gen.num_classes, activation='softmax'))
```

Cell 13:
```python
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

Cell 14:
```python
trainig1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    class_weight=class_weight,
    callbacks=[early_stop, lr_reduce]
)
```

Cell 15:
```python
print('phase2...finetuning')
base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

```

Cell 16:
```python
training_finetuning = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    class_weight=class_weight,
    callbacks=[early_stop, lr_reduce]
)
```


## apple-disease-detection-efficientnetb3-finetuned.ipynb

File: `original_work/Apple disease detection codes/apple-disease-detection-efficientnetb3-finetuned.ipynb`. No embedded PNG figures. No stored precision/recall/F1 report. Imported plotting libraries alone do not establish plotted results. No written conclusions were stored.

### Training evidence

[
  {
    "cell": 16,
    "logged_epochs": 10,
    "last_train": "0.8232",
    "max_val": 0.96,
    "last_val": "0.9467"
  },
  {
    "cell": 19,
    "logged_epochs": 6,
    "last_train": "0.9061",
    "max_val": 0.96,
    "last_val": "0.9467"
  }
]

### Printed summaries

- Cell 21: Best Training Accuracy: 0.9414; Best Validation Accuracy: 0.9600
- Cell 22: Best Validation Accuracy: 0.9599999785423279

### Architecture, preprocessing, weights, fine-tuning, optimizer and callback source

Cell 1:
```python
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image
```

Cell 3:
```python
img_size = (300, 300) 
batch_size = 32
```

Cell 4:
```python
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=25,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)
```

Cell 5:
```python
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2
)
```

Cell 6:
```python
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
```

Cell 11:
```python
labels = train_gen.classes
class_weight = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weight = dict(enumerate(class_weight))
```

Cell 12:
```python
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, verbose=1)
```

Cell 13:
```python
base_model = EfficientNetB3(include_top=False, weights='imagenet', input_shape=(300, 300, 3))
base_model.trainable = False
```

Cell 14:
```python
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.4))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(train_gen.num_classes, activation='softmax'))
```

Cell 15:
```python
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

Cell 16:
```python
print("phase1..training(freez lower layers)")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    class_weight=class_weight,
    callbacks=[early_stop, lr_reduce]
)

```

Cell 17:
```python
print("phase2...finetuning")
base_model.trainable = True
fine_tune_at = 300  
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
```

Cell 18:
```python
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

Cell 19:
```python
training = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    class_weight=class_weight,
    callbacks=[early_stop, lr_reduce]
)
```


## apple-disease-detection-inceptionv3-finetuned.ipynb

File: `original_work/Apple disease detection codes/apple-disease-detection-inceptionv3-finetuned.ipynb`. No embedded PNG figures. No stored precision/recall/F1 report. Imported plotting libraries alone do not establish plotted results. No written conclusions were stored.

### Training evidence

[
  {
    "cell": 14,
    "logged_epochs": 10,
    "last_train": "0.6932",
    "max_val": 0.84,
    "last_val": "0.8400"
  },
  {
    "cell": 16,
    "logged_epochs": 10,
    "last_train": "0.8919",
    "max_val": 0.8933,
    "last_val": "0.8800"
  }
]

### Printed summaries

- Cell 18: Best Training Accuracy: 0.8990; Best Validation Accuracy: 0.8933
- Cell 19: Best Validation Accuracy: 0.8933333158493042

### Architecture, preprocessing, weights, fine-tuning, optimizer and callback source

Cell 1:
```python
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
```

Cell 3:
```python
img_size = (299, 299)  
batch_size = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=25,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)
```

Cell 4:
```python
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)
```

Cell 9:
```python
labels_from_gen = train_gen.classes
class_weight = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels_from_gen),
    y=labels_from_gen
)
class_weight = dict(enumerate(class_weight))

```

Cell 10:
```python
lr_reduce = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
```

Cell 11:
```python
base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(299, 299, 3))
base_model.trainable = False
```

Cell 12:
```python
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(train_gen.num_classes, activation='softmax'))

```

Cell 13:
```python
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

```

Cell 14:
```python
print('phase1...freez')
training = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    class_weight=class_weight,
    callbacks=[early_stop, lr_reduce]
)
```

Cell 15:
```python
base_model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

Cell 16:
```python
print("Phase...finetuning ")
finetuning = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    class_weight=class_weight,
    callbacks=[early_stop, lr_reduce]
)
```


## apple-disease-detection-mobilenetv2-finetuned.ipynb

File: `original_work/Apple disease detection codes/apple-disease-detection-mobilenetv2-finetuned.ipynb`. No embedded PNG figures. No stored precision/recall/F1 report. Imported plotting libraries alone do not establish plotted results. No written conclusions were stored.

### Training evidence

[
  {
    "cell": 15,
    "logged_epochs": 10,
    "last_train": "0.8564",
    "max_val": 0.92,
    "last_val": "0.8800"
  },
  {
    "cell": 19,
    "logged_epochs": 6,
    "last_train": "0.9003",
    "max_val": 0.8667,
    "last_val": "0.7067"
  }
]

### Printed summaries

- Cell 21: Best Training Accuracy: 0.9023; Best Validation Accuracy: 0.8667
- Cell 22: Best Validation Accuracy: 0.8666666746139526

### Architecture, preprocessing, weights, fine-tuning, optimizer and callback source

Cell 1:
```python
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image
```

Cell 3:
```python
img_size = (128, 128)
batch_size = 32
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=25,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)
```

Cell 4:
```python
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
```

Cell 5:
```python
test_datagen = ImageDataGenerator(rescale=1./255)
```

Cell 10:
```python
labels_from_gen = train_gen.classes
class_weight = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels_from_gen),
    y=labels_from_gen
)
class_weight = dict(enumerate(class_weight))
```

Cell 11:
```python
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.3, verbose=1)
```

Cell 12:
```python
base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
base_model.trainable = False
```

Cell 13:
```python
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(train_gen.num_classes, activation='softmax'))
```

Cell 14:
```python
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

Cell 15:
```python
print("Phase 1: Training with base frozen...")
model.fit(train_gen, validation_data=val_gen,
          epochs=10, class_weight=class_weight,
          callbacks=[early_stop, lr_reduce])
```

Cell 16:
```python
print("Phase 2: Fine-tuning...")
base_model.trainable = True
```

Cell 17:
```python
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
```

Cell 18:
```python
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

Cell 19:
```python
training = model.fit(train_gen, validation_data=val_gen,
          epochs=10, class_weight=class_weight,
          callbacks=[early_stop, lr_reduce])
```


## apple-disease-detection-resnet50-finetuned (1).ipynb

File: `original_work/Apple disease detection codes/apple-disease-detection-resnet50-finetuned (1).ipynb`. No embedded PNG figures. No stored precision/recall/F1 report. Imported plotting libraries alone do not establish plotted results. No written conclusions were stored.

### Training evidence

[
  {
    "cell": 14,
    "logged_epochs": 10,
    "last_train": "0.7668",
    "max_val": 0.92,
    "last_val": "0.9200"
  },
  {
    "cell": 18,
    "logged_epochs": 15,
    "last_train": "0.7983",
    "max_val": 0.9067,
    "last_val": "0.8933"
  }
]

### Printed summaries

- Cell 19: Best Training Accuracy: 0.8436; Best Validation Accuracy: 0.9067
- Cell 20: Best Validation Accuracy: 0.9066666960716248

### Architecture, preprocessing, weights, fine-tuning, optimizer and callback source

Cell 1:
```python
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image

```

Cell 3:
```python
img_size = (224, 224) 
batch_size = 32

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=25,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)
```

Cell 4:
```python
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
```

Cell 9:
```python
labels_from_gen = train_gen.classes
class_weight = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels_from_gen),
    y=labels_from_gen
)
class_weight = dict(enumerate(class_weight))

```

Cell 10:
```python
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.5)
```

Cell 11:
```python
base_model = ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False
```

Cell 12:
```python
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(train_gen.num_classes, activation='softmax')
])
```

Cell 13:
```python
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

Cell 14:
```python
print("Phase 1: Training with frozen ResNet base")
history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    class_weight=class_weight,
    callbacks=[early_stop, lr_reduce]
)
```

Cell 15:
```python
print("Phase 2: Fine-tuning the ResNet model")
base_model.trainable = True
for layer in base_model.layers[:100]:  
    layer.trainable = False
```

Cell 16:
```python
model.compile(optimizer=tf.keras.optimizers.Adam(3e-6),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

Cell 17:
```python
early_stop2 = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_reduce2 = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5)
```

Cell 18:
```python
history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=15,
    class_weight=class_weight,
    callbacks=[early_stop2, lr_reduce2]
)
```


## apple-disease-detection-xception-finetuned.ipynb

File: `original_work/Apple disease detection codes/apple-disease-detection-xception-finetuned.ipynb`. No embedded PNG figures. No stored precision/recall/F1 report. Imported plotting libraries alone do not establish plotted results. No written conclusions were stored.

### Training evidence

[
  {
    "cell": 14,
    "logged_epochs": 10,
    "last_train": "0.7723",
    "max_val": 0.8667,
    "last_val": "0.8667"
  },
  {
    "cell": 18,
    "logged_epochs": 10,
    "last_train": "0.8641",
    "max_val": 0.8933,
    "last_val": "0.8933"
  }
]

### Printed summaries

- Cell 20: Best Training Accuracy: 0.8567; Best Validation Accuracy: 0.8933
- Cell 21: Best Validation Accuracy: 0.8666666746139526

### Architecture, preprocessing, weights, fine-tuning, optimizer and callback source

Cell 1:
```python
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import Xception
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image
```

Cell 3:
```python
img_size = (224, 224)
batch_size = 32
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=25,
    zoom_range=0.3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)
```

Cell 4:
```python
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)
```

Cell 9:
```python
labels_from_gen = train_gen.classes
class_weight = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels_from_gen),
    y=labels_from_gen
)
class_weight = dict(enumerate(class_weight))

```

Cell 10:
```python
base_model = Xception(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False
```

Cell 11:
```python
model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(train_gen.num_classes, activation='softmax'))

```

Cell 12:
```python
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

Cell 13:
```python
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, verbose=1)
```

Cell 14:
```python
print("phase1...training and freezing")
training = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    class_weight=class_weight,
    callbacks=[early_stop, lr_reduce]
)

```

Cell 15:
```python
print("phase2...finetuning")
base_model.trainable = True
```

Cell 16:
```python
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
```

Cell 17:
```python
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

Cell 18:
```python
trainig_finetuned = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    class_weight=class_weight,
    callbacks=[early_stop, lr_reduce]
)
```


## ensemble-apple-disease-detection.ipynb

File: `original_work/Apple disease detection codes/ensemble-apple-disease-detection.ipynb`. No embedded PNG figures. No stored precision/recall/F1 report. Imported plotting libraries alone do not establish plotted results. No written conclusions were stored.

### Training evidence

[
  {
    "cell": 11,
    "logged_epochs": 10,
    "last_train": "0.3170",
    "max_val": 0.4,
    "last_val": "0.3600"
  },
  {
    "cell": 11,
    "logged_epochs": 10,
    "last_train": "0.8626",
    "max_val": 0.92,
    "last_val": "0.9067"
  },
  {
    "cell": 11,
    "logged_epochs": 10,
    "last_train": "0.3154",
    "max_val": 0.3067,
    "last_val": "0.2933"
  }
]

### Printed summaries

Not stored.

### Architecture, preprocessing, weights, fine-tuning, optimizer and callback source

Cell 3:
```python
img_size = (224, 224)
batch_size = 32
train_dir = "/kaggle/input/appledisease/Train"
test_dir = "/kaggle/input/appledisease/Test"
```

Cell 4:
```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.2
)
```

Cell 5:
```python
test_datagen = ImageDataGenerator(rescale=1./255)
```

Cell 9:
```python
def build_model(model_name, num_classes):
    inputs = Input(shape=(224, 224, 3))
    
    if model_name == 'ResNet50':
        base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)
    elif model_name == 'DenseNet121':
        base_model = DenseNet121(include_top=False, weights='imagenet', input_tensor=inputs)
    elif model_name == 'EfficientNetB0':
        base_model = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=inputs)
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name=model_name)
    
    model.compile(optimizer=Adam(learning_rate=1e-3), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    return model
```

Cell 11:
```python
for name in model_names:
    print(f"\n{'='*40}")
    print(f" TRAINING MODEL: {name}")
    print(f"{'='*40}")
    model = build_model(name, train_gen.num_classes)
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
    
    # Train
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,  # Increase this if you have time
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Save the model
    save_path = f"{name}_apple_final.h5"
    model.save(save_path)
    print(f"✅ Saved {name} to {save_path}")
    
    # Clear memory to prevent crashing
    tf.keras.backend.clear_session()
```

Cell 14:
```python
ensemble_preds = (pred_res + pred_dense + pred_eff) / 3.0
final_classes = np.argmax(ensemble_preds, axis=1)
```

## Audit interpretation

*Last logged training accuracies in the comparison are progress-bar measurements, not exact full-epoch history values. Printed best-training summaries above are separate evidence. Maxima over validation logs need not correspond to the checkpoint restored by minimum validation loss.

- All transfer models load ImageNet weights and use global average pooling plus a dense softmax head. DenseNet121 and InceptionV3 unfreeze the full backbone; other notebook source shows partial unfreezing.
- The baseline and ensemble use the same augmented generator for validation; validation predictions therefore vary under augmentation. Most separate transfer notebooks provide a separate unaugmented validation generator.
- MobileNetV2, DenseNet121, InceptionV3 and Xception use only rescaling to [0,1], rather than their Keras ImageNet-specific preprocessing. ConvNeXt also rescales despite its built-in preprocessing. ResNet50 and EfficientNetB3 explicitly use their application preprocess_input.
- No explicit deterministic split seed, duplicate audit, untouched-test model selection policy, or reusable inference metadata was recovered.
- The CNN has 8,530,596 parameters, dominated by Flatten → Dense(256), despite only 307 training images. Its test evaluation source cell has no output, so test accuracy is unknown.
- MobileNetV2 and ResNet50 fine-tuning validation maxima are below their frozen-phase maxima. Fine-tuning is not automatically an improvement.
- Xception printed 0.8933 for fine-tuning validation, then 0.8666666746 from the frozen-phase history in a later cell. This is a history-variable mismatch, not test evidence.
- ConvNeXtBase is a convolutional architecture; a filename mentioning transformers does not make it a transformer.
- Ensemble averages ResNet50, DenseNet121 and EfficientNetB0 probabilities. It genuinely reports 73.33% on the 120-image test set. It saves three .h5 filenames in source/output, but none were present in Drive.
- Apart from ConvNeXt (57.50%) and the ensemble (73.33%), genuine stored test accuracy is unavailable. Validation values such as EfficientNetB3 96% are not test accuracy.
- The old ensemble logs include a failed CUDA initialization and dependency conflicts. These are historical logs; current GPU operation is independently recorded in environment.json.
- The new benchmark changes both classes and dataset. Its numbers must not be presented as a controlled improvement over the old four-class results.
