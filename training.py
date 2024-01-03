import os
import random as rd

# visualizations
import matplotlib.pyplot as plt
import matplotlib.image as img

# preparation and model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LambdaCallback
from tensorflow.keras.layers import BatchNormalization, Dropout, Dense, Flatten
from tensorflow.keras.optimizers import Adam, schedules
# ResMet50
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

AI_IMAGES =  'Ai' # AI images folder
HM_IMAGES = 'Human' # Man-made images folder
CLASSIFIED_FOLDER = [HM_IMAGES, AI_IMAGES]
ROOT = '.'
TRAIN_BASED_PATH = os.path.join(ROOT, 'images')
VALID_BASED_PATH = os.path.join(ROOT, 'validation_images')
IMG_SIZE = (32,32)
BATCH_SIZE=128

train_data = ImageDataGenerator(
    preprocessing_function=preprocess_input, 
    rotation_range=10, 
    width_shift_range=0.2,
    height_shift_range=0.2, 
    brightness_range=(0.1, 0.7), 
    shear_range=20, 
    zoom_range=20,
    horizontal_flip=True
)

validation_data = ImageDataGenerator(
    preprocessing_function=preprocess_input,
)

train_generator = train_data.flow_from_directory(
    TRAIN_BASED_PATH,
    batch_size = BATCH_SIZE,
    class_mode = 'binary',
    target_size = IMG_SIZE
)

validation_generator = validation_data.flow_from_directory(
    VALID_BASED_PATH,
    batch_size = BATCH_SIZE,
    class_mode = 'binary',
    target_size = IMG_SIZE
)


pre_trained_model = ResNet50(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=(IMG_SIZE[0],IMG_SIZE[1],3),
    pooling='avg'
 )
    
model = Sequential()
model.add(pre_trained_model)
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
    
FILEPATH = os.path.join(ROOT, 'generated-classification.keras')
def save_model(epoch, logs):
    model.save(FILEPATH, overwrite=True)


checkpoint = ModelCheckpoint(
    filepath=FILEPATH,
    monitor='val_loss',
    mode='auto',
    save_best_only=True
)

early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
h5_saver = LambdaCallback(on_epoch_end=save_model)

lr_schedule = schedules.ExponentialDecay(initial_learning_rate=0.0001, decay_steps=10000, decay_rate=0.95)
model.compile(loss='binary_crossentropy',optimizer=Adam(lr_schedule), metrics=['accuracy'])
history = model.fit(
    train_generator,
    steps_per_epoch=20,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=20,
    shuffle=True,
    callbacks=[checkpoint, early_stopping, h5_saver]
)
model.summary()

test_loss, test_acc = model.evaluate(validation_generator)
print(f"Test accuracy: {test_acc:.3f}")

epochs_range = range(20) #epochs
#Model Accuracy
plt.figure(figsize=(8,8))
plt.plot(epochs_range,history.history['accuracy'], label="Train acc")
plt.plot(epochs_range,history.history['val_accuracy'],label='Valid acc')
plt.axis(ymin=0.3,ymax=1)
plt.grid()
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])

#Model loss plot
plt.figure(figsize=(8,8))
plt.plot(epochs_range,history.history['loss'], label="Loss")
plt.plot(epochs_range,history.history['val_loss'],label='Valid loss')
plt.axis(ymin=0.3,ymax=3)
plt.grid()
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Loss', 'Validation Loss'])
