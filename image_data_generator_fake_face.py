
datagen = ImageDataGenerator(rescale=1.0/255.0)

train = datagen.flow_from_directory('Fake_Dataset/Train/',
                                    class_mode='binary',
                                    batch_size=64,
                                    target_size=(200,200))

test = datagen.flow_from_directory('Fake_Dataset/Test/',
                                    class_mode='binary',
                                    batch_size=64,
                                    target_size=(200,200))

history = model.fit(train,
                    validation_data=(test),
                    epochs = 50,
                    steps_per_epoch=len(train),
                    validation_steps=len(test))
