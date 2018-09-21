'''
创建训练集的txt文件
'''
import os
def create_train_test_txt(path):
    dict = create_dict()

    if not os.path.exists('./train100.txt'):
        os.mknod('train100.txt')
    fp1 = open('./train100.txt','w')

    if not os.path.exists('./test100.txt'):
        os.mknod('test100.txt')
    fp2 = open('./test100.txt','w')
    i=0
    num1=0
    npath = os.path.join(path, 'train100')
    os.chdir(npath)
    for dir_path in os.listdir(npath):
        newpath = os.path.join(npath,dir_path)
        for name in os.listdir(newpath):

            num = name.split('_')[0]
            label = num1
            line = os.path.join(newpath,name)+' '+str(label)+'\n'
            i+=1
            fp1.write(line)
        num1+=1
    print("total write in train is :",i)


    j=0
    num2=0
    npath = os.path.join(path, 'test100')
    os.chdir(npath)
    for dir_path in os.listdir(npath):
        newpath = os.path.join(npath, dir_path)
        for name in os.listdir(newpath):
            num = name.split('_')[0]
            label = num2
            line = os.path.join(newpath,name) + ' ' + str(label) + '\n'
            j += 1
            fp2.write(line)
        num2+=1
    print("total write in test is :", j)

def create_dict():

    path = './synset_words.txt'
    dict = {}
    i=-1
    with open(path) as names:
        for line in names:
            i+=1
            name = line.strip()
            num,lable = name.split(' ',1)

            dict[num] = (i)
    print("the dict size is :",len(dict))
    return dict


if __name__ == '__main__':
    #path = '/home/hq/desktop/ImageNet/data/'
    path = '/share/users_root/heqiang/ImageNet'
    create_train_test_txt(path)


