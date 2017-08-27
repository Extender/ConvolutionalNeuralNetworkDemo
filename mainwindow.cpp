#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    srand(time(0));

    examplesSeen=0;
    training=false;
    classified=false;
    currentImageId=0xFFFFFFFF;

    ui->setupUi(this);
    scene=new GraphicsSceneEx();
    pixmapItem=new QGraphicsPixmapItem();
    scene->addItem(pixmapItem);
    ui->graphicsView->setScene(scene);

    imageData=(uint32_t**)malloc(IMAGES_PER_BATCH*BATCH_COUNT*sizeof(uint32_t*));
    imageLabels=(uint8_t*)malloc(IMAGES_PER_BATCH*BATCH_COUNT*sizeof(uint8_t));
    imageInputData=(double****)malloc(IMAGES_PER_BATCH*BATCH_COUNT*sizeof(double***));

    QString dir=QString(IMAGE_DATA_DIR).replace("%APP_DIR%",QApplication::applicationDirPath());

    for(uint32_t batch=0;batch<BATCH_COUNT;batch++)
    {
        QFile f(dir+QString("data_batch_")+QString::number(batch+1)+".bin");
        if(!f.exists()||!f.open(QFile::ReadOnly))
        {
            int m=QMessageBox::critical(this,"Error",QString("CIFAR-10 dataset not found in directory specified by IMAGE_DATA_DIR.\n\
\n\
Please download the CIFAR-10 dataset archive and extract it into the directory.\n\
If you have already downloaded the dataset, copy the files into IMAGE_DATA_DIR (currently \"%DIR%\").\n\
\n\
Would you like to download the archive now?").replace("%DIR%",dir),QMessageBox::Yes|QMessageBox::No);
            if(m==QMessageBox::Yes)
                QDesktopServices::openUrl(QUrl("https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"));
            QApplication::exit();
        }
        uint64_t size=f.size();
        char *fileData=(char*)malloc(size);
        f.read(fileData,size);
        for(uint32_t image=0;image<IMAGES_PER_BATCH;image++)
        {
            uint32_t pos=batch*IMAGES_PER_BATCH+image;
            imageData[pos]=(uint32_t*)malloc(IMAGE_HEIGHT*IMAGE_WIDTH*sizeof(uint32_t)); // Desired format: 0xAARRGGBB
            imageLabels[pos]=fileData[image*BYTES_PER_IMAGE_IN_FILE];
            imageInputData[pos]=(double***)malloc(3/*Feature maps/color channels*/*sizeof(double**));
            imageInputData[pos][0]=(double**)malloc(IMAGE_HEIGHT*sizeof(double*)); // R channel
            imageInputData[pos][1]=(double**)malloc(IMAGE_HEIGHT*sizeof(double*)); // G channel
            imageInputData[pos][2]=(double**)malloc(IMAGE_HEIGHT*sizeof(double*)); // B channel
            for(uint32_t y=0;y<IMAGE_HEIGHT;y++)
            {
                imageInputData[pos][0][y]=(double*)malloc(IMAGE_WIDTH*sizeof(double)); // R channel
                imageInputData[pos][1][y]=(double*)malloc(IMAGE_WIDTH*sizeof(double)); // G channel
                imageInputData[pos][2][y]=(double*)malloc(IMAGE_WIDTH*sizeof(double)); // B channel
                for(uint32_t x=0;x<IMAGE_WIDTH;x++)
                {
                    // The first 1024 bytes of the image in the file (after the label bit) are the red channel values, the next 1024 the green, and the final 1024 the blue.

                    uint8_t a=255; // A
                    uint8_t r=(uint8_t)fileData[image*BYTES_PER_IMAGE_IN_FILE+1/*Label bit*/+y*IMAGE_WIDTH+x]; // R
                    uint8_t g=(uint8_t)fileData[image*BYTES_PER_IMAGE_IN_FILE+1/*Label bit*/+1024+y*IMAGE_WIDTH+x]; // G
                    uint8_t b=(uint8_t)fileData[image*BYTES_PER_IMAGE_IN_FILE+1/*Label bit*/+2*1024+y*IMAGE_WIDTH+x]; // B
                    imageData[pos][y*IMAGE_WIDTH+x]=(a<<24)|(r<<16)|(g<<8)|b;

                    imageInputData[pos][0][y][x]=((double)r)/255.0; // R channel
                    imageInputData[pos][1][y][x]=((double)g)/255.0; // G channel
                    imageInputData[pos][2][y][x]=((double)b)/255.0; // B channel
                }
            }
        }
        free(fileData);
        f.close();
    }

    desiredOutputValueCache=(double****)malloc(LABEL_COUNT*sizeof(double***));
    for(uint32_t label=0;label<LABEL_COUNT;label++)
    {
        desiredOutputValueCache[label]=(double***)malloc(LABEL_COUNT*sizeof(double**));
        for(uint32_t labelInLabelCacheEntry=0;labelInLabelCacheEntry<LABEL_COUNT;labelInLabelCacheEntry++)
        {
            desiredOutputValueCache[label][labelInLabelCacheEntry]=(double**)malloc(1*sizeof(double*));
            desiredOutputValueCache[label][labelInLabelCacheEntry][0]=(double*)malloc(1*sizeof(double));
            desiredOutputValueCache[label][labelInLabelCacheEntry][0][0]=(labelInLabelCacheEntry==label?1.0:0.0);
        }
    }

    connect(ui->nextBtn,SIGNAL(clicked(bool)),this,SLOT(nextBtnClicked()));
    connect(ui->classifyBtn,SIGNAL(clicked(bool)),this,SLOT(classifyBtnClicked()));
    connect(ui->trainBtn,SIGNAL(clicked(bool)),this,SLOT(trainBtnClicked()));

    nextBtnClicked();


    // The actual convolutional neural network:

    CNNLayer *layer1; // Type: CONV
    CNNLayer *layer2; // Type: RELU
    CNNLayer *layer3; // Type: MAXPOOL
    CNNLayer *layer4; // Type: CONV
    CNNLayer *layer5; // Type: RELU
    CNNLayer *layer6; // Type: MAXPOOL
    CNNLayer *layer7; // Type: CONV
    CNNLayer *layer8; // Type: RELU
    CNNLayer *layer9; // Type: MAXPOOL
    CNNLayer *layer10; // Type: FC
    CNNLayer *layer11; // Type: SOFTMAX

    layer1=new CNNLayer(1,CNN_LAYER_TYPE_CONV,16,5,5,1,1,2,2,3,IMAGE_WIDTH,IMAGE_HEIGHT);
    layer2=new CNNLayer(2,CNN_LAYER_TYPE_RELU,layer1->featureMapCount,1,1,1,1,2,2,layer1->featureMapCount,layer1->singleFeatureMapWidth,layer1->singleFeatureMapHeight);
    layer3=new CNNLayer(3,CNN_LAYER_TYPE_MAXPOOL,layer2->featureMapCount,2,2,2,2,0,0,layer2->featureMapCount,layer2->singleFeatureMapWidth,layer2->singleFeatureMapHeight);
    layer4=new CNNLayer(4,CNN_LAYER_TYPE_CONV,20,5,5,1,1,2,2,layer3->featureMapCount,layer3->singleFeatureMapWidth,layer3->singleFeatureMapHeight);
    layer5=new CNNLayer(5,CNN_LAYER_TYPE_RELU,layer4->featureMapCount,1,1,1,1,2,2,layer4->featureMapCount,layer4->singleFeatureMapWidth,layer4->singleFeatureMapHeight);
    layer6=new CNNLayer(6,CNN_LAYER_TYPE_MAXPOOL,layer5->featureMapCount,2,2,2,2,0,0,layer5->featureMapCount,layer5->singleFeatureMapWidth,layer5->singleFeatureMapHeight);
    layer7=new CNNLayer(7,CNN_LAYER_TYPE_CONV,20,5,5,1,1,2,2,layer6->featureMapCount,layer6->singleFeatureMapWidth,layer6->singleFeatureMapHeight);
    layer8=new CNNLayer(8,CNN_LAYER_TYPE_RELU,layer7->featureMapCount,1,1,1,1,2,2,layer7->featureMapCount,layer7->singleFeatureMapWidth,layer7->singleFeatureMapHeight);
    layer9=new CNNLayer(9,CNN_LAYER_TYPE_MAXPOOL,layer8->featureMapCount,2,2,2,2,0,0,layer8->featureMapCount,layer8->singleFeatureMapWidth,layer8->singleFeatureMapHeight);

    layer10=new CNNLayer(10,CNN_LAYER_TYPE_FC,10,0,0,1,1,0,0,layer9->featureMapCount,layer9->singleFeatureMapWidth,layer9->singleFeatureMapHeight);
    layer11=new CNNLayer(11,CNN_LAYER_TYPE_SOFTMAX,10,0,0,1,1,0,0,layer10->featureMapCount,layer10->singleFeatureMapWidth,layer10->singleFeatureMapHeight);

    // Store layers in layer array:

    layers=(CNNLayer**)malloc(LAYER_COUNT*sizeof(CNNLayer*));
    layers[0]=layer1;
    layers[1]=layer2;
    layers[2]=layer3;
    layers[3]=layer4;
    layers[4]=layer5;
    layers[5]=layer6;
    layers[6]=layer7;
    layers[7]=layer8;
    layers[8]=layer9;
    layers[9]=layer10;
    layers[10]=layer11;

    trainingThread=new TrainingThread(this,DEFAULT_LEARNING_RATE,DEFAULT_MOMENTUM,DEFAULT_WEIGHT_DECAY);
    // Use Qt::QueuedConnection to indicate that the slot is to be executed in the receiving QObject's thread.
    connect(trainingThread,SIGNAL(iterationFinished(unsigned int,double***)),this,SLOT(trainingThreadIterationFinished(unsigned int,double***)),Qt::QueuedConnection);
    connect(trainingThread,SIGNAL(finished()),this,SLOT(trainingThreadFinishedWorking()),Qt::QueuedConnection);

    accuracyVector=new std::vector<double>();
    ui->accuracyLbl->setText(QString("<b>0.0</b> - accuracy of last ")+QString::number(ACCURACY_VECTOR_MAX_SIZE)+QString(" classifications"));

    ui->learningRateBox->setValue(DEFAULT_LEARNING_RATE);
    ui->momentumBox->setValue(DEFAULT_MOMENTUM);
    ui->weightDecayBox->setValue(DEFAULT_WEIGHT_DECAY);

    connect(ui->learningRateBox,SIGNAL(valueChanged(double)),this,SLOT(learningRateBoxValueChanged(double)));
    connect(ui->momentumBox,SIGNAL(valueChanged(double)),this,SLOT(momentumBoxValueChanged(double)));
    connect(ui->weightDecayBox,SIGNAL(valueChanged(double)),this,SLOT(weightDecayBoxValueChanged(double)));
}

MainWindow::~MainWindow()
{
    for(uint32_t batch=0;batch<BATCH_COUNT;batch++)
    {
        for(uint32_t image=0;image<IMAGES_PER_BATCH;image++)
        {
            uint32_t pos=batch*IMAGES_PER_BATCH+image;
            free(imageData[pos]);
            for(int32_t y=0;y<IMAGE_HEIGHT;y++)
            {
                free(imageInputData[pos][0][y]); // R channel
                free(imageInputData[pos][1][y]); // G channel
                free(imageInputData[pos][2][y]); // B channel
            }
            free(imageInputData[pos][0]); // R channel
            free(imageInputData[pos][1]); // G channel
            free(imageInputData[pos][2]); // B channel
            free(imageInputData[pos]);
        }
    }
    free(imageData);
    free(imageLabels);
    free(imageInputData);
    delete ui;
    delete pixmapItem;
    delete scene;

    for(uint32_t layer=0;layer<LAYER_COUNT;layer++)
        delete layers[layer];
    free(layers);

    for(uint32_t label=0;label<LABEL_COUNT;label++)
    {
        free(desiredOutputValueCache[label][0][0]);
        free(desiredOutputValueCache[label][0]);
        free(desiredOutputValueCache[label]);
    }
    free(desiredOutputValueCache);
    delete accuracyVector;
}

QString MainWindow::getLabelName(uint8_t label)
{
    if(label==0)
        return "airplane";
    else if(label==1)
        return "automobile";
    else if(label==2)
        return "bird";
    else if(label==3)
        return "cat";
    else if(label==4)
        return "deer";
    else if(label==5)
        return "dog";
    else if(label==6)
        return "frog";
    else if(label==7)
        return "horse";
    else if(label==8)
        return "ship";
    else if(label==9)
        return "truck";
    return "unknown";
}

void MainWindow::loadImage(uint32_t imageId)
{
    classified=false;
    currentImageId=imageId;
    QImage img=QImage((uchar*)imageData[imageId],IMAGE_WIDTH,IMAGE_HEIGHT,4*IMAGE_WIDTH,QImage::Format_ARGB32);
    pixmapItem->setPixmap(QPixmap::fromImage(img));
    ui->graphicsView->update();
    QString labelName=getLabelName(imageLabels[imageId]);
    ui->labelLbl->setText(QString("Label: <b>")+labelName.replace(0,1,labelName.at(0).toUpper())+QString("</b>"));
    ui->labelLbl->update();
}

uint32_t MainWindow::getHighestIndex(double *array, uint32_t elementCount)
{
    double highestValue=std::numeric_limits<double>::min();
    uint32_t highestIndex=0xFFFFFFFF;
    for(uint32_t thisIndex=0;thisIndex<elementCount;thisIndex++)
    {
        double thisValue=array[thisIndex];
        if(highestValue<thisValue)
        {
            highestValue=thisValue;
            highestIndex=thisIndex;
        }
    }
    return highestIndex;
}

void MainWindow::displayOutput(double ***output, uint8_t correctLabel)
{
    std::vector<std::pair<uint8_t,double> > resultVector=std::vector<std::pair<uint8_t,double> >();
    for(uint8_t label=0;label<LABEL_COUNT;label++)
        resultVector.push_back(std::pair<uint8_t,double>(label,output[label][0][0]));

    std::sort(resultVector.begin(),resultVector.end(),ResultVectorLessThanKey());

    QString out="Result: ";

    for(uint8_t result=0;result<SHOW_FIRST_RESULTS;result++)
    {
        std::pair<uint8_t,double> p=resultVector.at(result);
        uint8_t label=p.first;
        double value=p.second;
        QString labelName=getLabelName(label);
        //name=name.replace(0,1,name.at(0).toUpper()); // Capitalize
        out+=QString(result>0?"; ":"")
            +QString(result==0?"<b>":"")
            +labelName
            +QString(result==0?"</b>":"")
            +QString(" (")
            +QString::number(value)
            +QString(")");
    }

    ui->resultLbl->setText(out);
    ui->resultLbl->update();

    if(accuracyVector->size()==ACCURACY_VECTOR_MAX_SIZE)
        accuracyVector->erase(accuracyVector->begin());
    accuracyVector->push_back(resultVector.at(0).first==correctLabel?1.0:0.0);

    double accuracySum=0.0;
    uint32_t accuracyVectorSize=accuracyVector->size();
    for(uint32_t i=0;i<accuracyVectorSize;i++)
        accuracySum+=accuracyVector->at(i);
    double accuracyOfLastN=accuracySum/accuracyVectorSize;

    ui->accuracyLbl->setText(QString("<b>")+QString::number(accuracyOfLastN,'g',3)+QString("</b> - accuracy of last ")+QString::number(ACCURACY_VECTOR_MAX_SIZE)+QString(" classifications"));
}

void MainWindow::updateExamplesSeenLbl()
{
    ui->examplesSeenLbl->setText(QString("<b>")+QString::number(examplesSeen)+QString("</b> - examples seen during training"));
    ui->examplesSeenLbl->update();
}

void MainWindow::nextBtnClicked()
{
    loadImage(((double)rand())/((double)RAND_MAX)*(IMAGES_PER_BATCH*BATCH_COUNT-1)); // Start at 0
}

void MainWindow::classifyBtnClicked()
{
    if(classified)
    {
        // Classify next
        nextBtnClicked();
        classifyBtnClicked();
    }
    else
    {
        classified=true;
        ui->statusLbl->setText("Classifying...");
        ui->statusLbl->update();

        // Forward pass
        // MODIFY IN TRAININGTHREAD.CPP, TOO!

        // Input for first layer: Image data
        double ***previousLayerOutput=imageInputData[currentImageId];

        for(uint32_t layerIndex=0;layerIndex<LAYER_COUNT;layerIndex++)
        {
            CNNLayer *thisLayer=layers[layerIndex];
            double ***output=thisLayer->forwardPass(previousLayerOutput);

            if(layerIndex>0)
                CNNLayer::freeArray(previousLayerOutput,thisLayer->previousLayerFeatureMapCount,thisLayer->previousLayerSingleFeatureMapHeight);
            previousLayerOutput=output;
        }

        // MODIFY IN TRAININGTHREAD.CPP, TOO!

        uint8_t thisLabel=imageLabels[currentImageId];
        displayOutput(previousLayerOutput,thisLabel);
        //examplesSeen++;
        //updateExamplesSeenLbl();

        CNNLayer::freeArray(previousLayerOutput,layers[LAYER_COUNT-1]->featureMapCount,layers[LAYER_COUNT-1]->singleFeatureMapHeight);

        ui->statusLbl->setText("Ready.");
        ui->statusLbl->update();
    }
}

void MainWindow::trainBtnClicked()
{
    if(training)
    {
        trainingThread->stopRequested=true;
    }
    else
    {
        training=true;
        trainingThread->start();
        ui->trainBtn->setText("Stop training");
        ui->statusLbl->setText(QString("Training..."));
        ui->trainBtn->update();
        ui->statusLbl->update();
    }
}

void MainWindow::learningRateBoxValueChanged(double newValue)
{
    trainingThread->learningRate=newValue;
}

void MainWindow::momentumBoxValueChanged(double newValue)
{
    trainingThread->momentum=newValue;
}

void MainWindow::weightDecayBoxValueChanged(double newValue)
{
    trainingThread->weightDecay=newValue;
}

void MainWindow::trainingThreadIterationFinished(unsigned int imageId, double ***output)
{
    loadImage(imageId);
    examplesSeen++;
    updateExamplesSeenLbl();
    displayOutput(output,imageLabels[imageId]);

    CNNLayer::freeArray(output,layers[LAYER_COUNT-1]->featureMapCount,layers[LAYER_COUNT-1]->singleFeatureMapHeight);
}

void MainWindow::trainingThreadFinishedWorking()
{
    training=false;
    ui->trainBtn->setText("Train");
    ui->statusLbl->setText("Ready.");
    ui->trainBtn->update();
    ui->statusLbl->update();
}
