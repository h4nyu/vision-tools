#!/bin/sh
kaggle competitions download -c kuzushiji-recognition -p /store/kuzushiji-recognition
cd /store/kuzushiji-recognition
unzip kuzushiji-recognition.zip
unzip test_images.zip
unzip train_images.zip
