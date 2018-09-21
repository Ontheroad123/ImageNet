#!/bin/bash

#cd /home/hq/desktop/workspace/jiaoben/data
#cd /share/users_root/heqiang/ImageNet/ILSVRC2012_img_train
:'ls *.tar > ls.log
data=$(cat ls.log)
for i in $data
do
  #echo $i
  name=${i%%.*}
  #echo $name
  mkdir ./train/$name
  cp $i ./train/$name
  #mkdir ../train/$name
  #cp $i ../train/$name
done
rm -rf ls.log'
#path=/home/hq/desktop/workspace/jiaoben/data/train
path=/share/users_root/heqiang/ImageNet/train
files=$(ls $path)
for filename in $files
do
  newpath=$path/$filename
  cd $newpath
  name=$(ls $newpath)
  #tar -xvf $name
  rm -rf $name
done
