#!/bin/sh

[ -d "/store/subaru" ] && mkdir /store/subaru
signate download -c 563 -p /store/subaru

cd /store/subaru
unzip /store/subaru/evaluation.zip
unzip /store/subaru/readme.zip
unzip /store/subaru/test_annotations.zip
unzip /store/subaru/test_videos_2.zip
unzip /store/subaru/train_annotations.zip
unzip /store/subaru/train_videos_4.zip
