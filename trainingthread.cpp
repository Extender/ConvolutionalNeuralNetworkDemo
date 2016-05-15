#include "trainingthread.h"

TrainingThread::TrainingThread(MainWindow *_window, double _learningRate, double _momentum, double _weightDecay)
{
    stopRequested=false;
    window=_window;
    mutex=new QMutex();

    learningRate=_learningRate;
    momentum=_momentum;
    weightDecay=_weightDecay;
}

TrainingThread::~TrainingThread()
{
    mutex->unlock();
    delete mutex;
}

void TrainingThread::run()
{
    srand(time(0));

    for(/*;;*/uint64_t cycle=0;cycle<100000000;cycle++)
    {
        if(stopRequested)
            break;
        uint32_t imageId=((double)rand())/((double)RAND_MAX)*IMAGES_PER_BATCH*BATCH_COUNT;
        uint8_t imageLabel=window->imageLabels[imageId];

        //forwardPass(imageId);

        // Input for first layer: Image data
        double ***previousLayerOutput=window->imageInputData[imageId];

        // Forward pass
        // MODIFY IN MAINWINDOW.CPP, TOO!

        for(uint32_t layerIndex=0;layerIndex<LAYER_COUNT;layerIndex++)
        {
            CNNLayer *thisLayer=window->layers[layerIndex];
            double ***output=thisLayer->forwardPass(previousLayerOutput);

            if(layerIndex>0)
                CNNLayer::freeArray(previousLayerOutput,thisLayer->previousLayerFeatureMapCount,thisLayer->previousLayerSingleFeatureMapHeight);
            previousLayerOutput=output;
        }

        // previousLayerOutput now contains the output of the last layer

        // Backward pass

        double ***higherLayerInputDiffs=0;

        for(uint32_t _layerIndex=LAYER_COUNT;_layerIndex>0;_layerIndex--) // _layerIndex is of type uint32_t and cannot be <0, therefore we have to artificially increment it by 1
        {
            uint32_t layerIndex=_layerIndex-1; // The actual index of this layer
            CNNLayer *thisLayer=window->layers[layerIndex];
            CNNLayer *higherLayer=layerIndex<LAYER_COUNT-1?window->layers[layerIndex+1]:0;

            double ***inputDiffs=0;
            double ****weightDiffs=0;
            double *biasDiffs=0;

            // Error in calculateDiffs (tested)

            thisLayer->calculateDiffs(weightDiffs,biasDiffs,higherLayerInputDiffs,inputDiffs,imageLabel);
            thisLayer->applyDiffs(weightDiffs,biasDiffs,learningRate,momentum,weightDecay);

            if(weightDiffs!=0)
            {
                if(thisLayer->type==CNN_LAYER_TYPE_CONV)
                    CNNLayer::freeWeightTypeArray(weightDiffs,thisLayer->previousLayerFeatureMapCount,thisLayer->featureMapCount,thisLayer->receptiveFieldHeight,thisLayer->receptiveFieldWidth);
                else if(thisLayer->type==CNN_LAYER_TYPE_FC)
                    CNNLayer::freeWeightTypeArray(weightDiffs,thisLayer->previousLayerFeatureMapCount,thisLayer->previousLayerSingleFeatureMapHeight,thisLayer->previousLayerSingleFeatureMapWidth,thisLayer->featureMapCount);
            }

            if(biasDiffs!=0)
                CNNLayer::freeBiasTypeArray(biasDiffs);

            if(layerIndex<LAYER_COUNT-1)
                CNNLayer::freeArray(higherLayerInputDiffs,higherLayer->previousLayerFeatureMapCount,higherLayer->previousLayerSingleFeatureMapHeight);

            higherLayerInputDiffs=inputDiffs; // Will be freed when processing the next layer
        }

        if(higherLayerInputDiffs!=window->desiredOutputValueCache[imageLabel]&&higherLayerInputDiffs!=0)
            CNNLayer::freeArray(higherLayerInputDiffs,window->layers[0]->previousLayerFeatureMapCount,window->layers[0]->previousLayerSingleFeatureMapHeight);

        // previousLayerOutput now contains the output of the last layer

        iterationFinished(imageId,previousLayerOutput);
    }
    stopRequested=false;
    window->training=false;
}

