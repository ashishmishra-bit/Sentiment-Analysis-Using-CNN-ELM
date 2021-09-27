import numpy as np
# from keras.datasets import mnist
from keras.utils import to_categorical

# from model import ELM


dim=(256,256)
imageShape = (dim[0],dim[1],3)
numClasses = 2
batchSize = 10
epochs = 1
folderWithPics='twitter'
dirs=os.listdir('./'+folderWithPics)
clsLabels=pd.read_csv('./'+folderWithPics+'/groundTruthLabel.txt',delimiter='\t')
clsLabels.index=clsLabels.index+1
subDirPath=[('./'+folderWithPics+'/'+di) for di in dirs if('txt' not in di)]
allImagesTrainPath=[(si+'/'+ii) for si in subDirPath[:-1] for ii in os.listdir(si) if('jpg' in ii)]
allImagesTestPath=[(si+'/'+ii) for si in [subDirPath[-1]] for ii in os.listdir(si) if('jpg' in ii)]

def formImageSet(allImagesFoldrPath,dim,clsLabels):
    x_imageSet=np.empty((len(allImagesFoldrPath),dim[0],dim[1],3))
    y_Set=np.empty((len(allImagesFoldrPath),1))
    for im in range(len(allImagesFoldrPath)):
        readImage=imread(allImagesFoldrPath[im])
        
        imNum=int(allImagesFoldrPath[im].split('/')[-1].split('.')[0])
        actualClass=clsLabels.loc[imNum][1]
        
        if (actualClass=='positive'):
            y_Set[im]=1
        else:
            y_Set[im]=0
            
        if (len(readImage.shape)>=3):
            if readImage.shape[2]>3:
                readImage=readImage[:,:,:3]            
        else:
            print(im,readImage.shape)
            readImage=gray2rgb(readImage)            
        readImage=resize(readImage,dim)
        x_imageSet[im]=readImage
    return x_imageSet,y_Set



def main():
    num_classes = 10
    num_hidden_layers = 1024
    (xTrainImSet, yTrainSet), ( xTestImSet, yTestSet) = folderWithPics.load_data()

    # Process images into input vectors
    # each mnist image is a 28x28 picture with value ranges between 0 and 255
    xTrainImSet,yTrainSet=formImageSet(allImagesTrainPath,dim,clsLabels)
    xTestImSet,yTestSet=formImageSet(allImagesTestPath,dim,clsLabels)
    
    xTrainImSet= xTrainImSet.astype('float32')
    xTestImSet= xTestImSet.astype('float32')
    xTrainImSet /= 255.0
    xTestImSet /= 255.0

    yTrainSet= keras.utils.np_utils.to_categorical(yTrainSet, numClasses)
    yTestSet= keras.utils.np_utils.to_categorical(yTestSet, numClasses)

    # converts [1,2] into [[0,1,0], [0,0,1]]
#     y_train = to_categorical(y_train, num_classes).astype(np.float32)
#     y_test = to_categorical(y_test, num_classes).astype(np.float32)
    
    print('Train Dataset size: ', xTrainImSet.shape[0])
    print('Test Dataset size: ', yTestSet.shape[0])

    # create instance of our model
    model = ELM(
        28 ** 2,
        num_hidden_layers,
        num_classes
    )

    # Train
    model.fit(x_train, y_train)
    train_loss, train_acc = model.evaluate(x_train, y_train)
    print('train loss: %f' % train_loss)
    print('train acc: %f' % train_acc)

    # Validation
    val_loss, val_acc = model.evaluate(x_test, y_test)
    print('val loss: %f' % val_loss)
    print('val acc: %f' % val_acc)


if __name__ == '__main__':
    main()
