#path=/home/hq/desktop/workspace/jiaoben/data/train
path=/share/users_root/heqiang/ImageNet/train
files=$(ls $path)
for filename in $files
do
  newpath=$path/$filename
  cd $newpath
  ls *.tar > ls.log
  data=$(cat ls.log)
  echo $data
  rm -rf ls.log
  rm -rf $data
  #tar -xvf $name
  
done
