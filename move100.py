import os
import shutil

def movetrain(path):
    crepath = os.path.join(path, "train100")
    if not os.path.exists(crepath):
        os.makedirs(crepath)

    count = 0
    path1 = os.path.join(path, "train")
    for files in os.listdir(path1):

        if count<100:
            srcpath = os.path.join(path1,files)
            despath = os.path.join(crepath,files)
            os.makedirs(despath)
            for file in os.listdir(srcpath):

                old = os.path.join(srcpath,file)
                #print(despath,old)

                shutil.copy(old,despath)

            count+=1

def movetest(path):
    crepath = os.path.join(path, "test100")
    if not os.path.exists(crepath):
        os.makedirs(crepath)

    count = 0
    path1 = os.path.join(path, "test")
    for files in os.listdir(path1):

        if count<100:
            srcpath = os.path.join(path1,files)
            despath = os.path.join(crepath,files)
            os.makedirs(despath)
            for file in os.listdir(srcpath):

                old = os.path.join(srcpath,file)
                #print(despath,old)

                shutil.copy(old,despath)

            count+=1

#path = "/home/hq/desktop/ImageNet/data"
path = "/share/users_root/heqiang/ImageNet"
movetrain(path)
