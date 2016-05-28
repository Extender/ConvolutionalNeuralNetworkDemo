#include "cnnlayer.h"

double CNNLayer::sig(double input)
{
    // Derivative: sig(input)*(1.0-sig(input))
    return 1.0/(1.0+pow(M_E,-input));
}

double CNNLayer::tanh(double input)
{
    // Derivative: 1.0-pow(tanh(input),2.0)
    return (1.0-pow(M_E,-2.0*input))/(1.0+pow(M_E,-2.0*input));
}

CNNLayer::CNNLayer(uint32_t _layerId, uint8_t _type, uint32_t _featureMapCount, int32_t _receptiveFieldWidth, int32_t _receptiveFieldHeight, uint32_t _strideX /*Default: 1*/, uint32_t _strideY /*Default: 1*/, uint32_t _zeroPaddingX, uint32_t _zeroPaddingY, uint32_t _previousLayerFeatureMapCount, int32_t _previousLayerSingleFeatureMapWidth, int32_t _previousLayerSingleFeatureMapHeight)
{
    layerId=_layerId; // Useful when debugging
    input=0;
    output=0;
    srand((unsigned int)time(0));
    type=_type;
    receptiveFieldWidth=_receptiveFieldWidth;
    receptiveFieldHeight=_receptiveFieldHeight;
    totalReceptiveFieldSize=receptiveFieldWidth*receptiveFieldHeight;
    previousLayerFeatureMapCount=_previousLayerFeatureMapCount;
    previousLayerSingleFeatureMapWidth=_previousLayerSingleFeatureMapWidth;
    previousLayerSingleFeatureMapHeight=_previousLayerSingleFeatureMapHeight;

    strideX=_strideX;
    strideY=_strideY;

    zeroPaddingX=_zeroPaddingX;
    zeroPaddingY=_zeroPaddingY;

    // ATTENTION:

    // Quote from A. Karpathy:
    // If this number is not an integer, then the strides are set incorrectly and the neurons cannot be tiled so that they "fit" across the input volume neatly, in a symmetric way.
    // (see http://cs231n.github.io/convolutional-networks/)

    // Formula for calculating receptive field size from desired single feature map size (the size can either be width or height, formula is applicable to both):
    // With:
    // Desired single feature map size: d; previous single layer feature map size: p; zero padding: z; stride: s; required receptive field size to get desired single feature map size: r
    // r=s(1-d)+p+2*z
    // (or use "getRequiredReceptiveFieldSizeForDesiredSingleFeatureMapSize")


    if(type==CNN_LAYER_TYPE_CONV)
    {
        // Modify CNN_LAYER_TYPE_MAXPOOL, too!

        featureMapCount=_featureMapCount;

        double _singleFeatureMapWidth=((double)(previousLayerSingleFeatureMapWidth-receptiveFieldWidth+2*zeroPaddingX))/((double)strideX)+1.0;
        double _singleFeatureMapHeight=((double)(previousLayerSingleFeatureMapHeight-receptiveFieldHeight+2*zeroPaddingY))/((double)strideY)+1.0;

        singleFeatureMapWidth=(int32_t)_singleFeatureMapWidth;
        singleFeatureMapHeight=(int32_t)_singleFeatureMapHeight;

        if(singleFeatureMapWidth!=_singleFeatureMapWidth||singleFeatureMapHeight!=_singleFeatureMapHeight)
            throw; // (See comment above)

        if(singleFeatureMapWidth<=0||singleFeatureMapHeight<=0) // Receptive field too large.
            throw;

        maxPixelMatrix=0;

        double initialMaxWeightValue=0.1;

        // Initialize weights and biases (modify for CNN_LAYER_TYPE_FC, too!)

        weights=(double****)malloc(previousLayerFeatureMapCount*sizeof(double***));
        previousWeightDiffDeltas=(double****)malloc(previousLayerFeatureMapCount*sizeof(double***));
        biasWeights=(double*)malloc(featureMapCount*sizeof(double));
        previousBiasWeightDiffDeltas=(double*)malloc(featureMapCount*sizeof(double));

        for(uint32_t featureMapInThisLayer=0;featureMapInThisLayer<featureMapCount;featureMapInThisLayer++)
        {
            biasWeights[featureMapInThisLayer]=0.0; // Initialize bias weights to 0
            previousBiasWeightDiffDeltas[featureMapInThisLayer]=0.0; // "previousBiasWeightDiffs" needs to be zero-initialized
        }
        for(uint32_t featureMapInPreviousLayer=0;featureMapInPreviousLayer<previousLayerFeatureMapCount;featureMapInPreviousLayer++)
        {
            weights[featureMapInPreviousLayer]=(double***)malloc(featureMapCount*sizeof(double**));
            previousWeightDiffDeltas[featureMapInPreviousLayer]=(double***)malloc(featureMapCount*sizeof(double**));
            for(uint32_t featureMapInThisLayer=0;featureMapInThisLayer<featureMapCount;featureMapInThisLayer++)
            {
                weights[featureMapInPreviousLayer][featureMapInThisLayer]=(double**)malloc(_receptiveFieldHeight*sizeof(double*));
                previousWeightDiffDeltas[featureMapInPreviousLayer][featureMapInThisLayer]=(double**)malloc(_receptiveFieldHeight*sizeof(double*));
                for(int32_t y=0;y<_receptiveFieldHeight;y++)
                {
                    weights[featureMapInPreviousLayer][featureMapInThisLayer][y]=(double*)malloc(_receptiveFieldWidth*sizeof(double));
                    previousWeightDiffDeltas[featureMapInPreviousLayer][featureMapInThisLayer][y]=(double*)malloc(_receptiveFieldWidth*sizeof(double));
                    for(int32_t x=0;x<_receptiveFieldWidth;x++)
                    {
                        weights[featureMapInPreviousLayer][featureMapInThisLayer][y][x]=-initialMaxWeightValue+(((double)rand())/((double)RAND_MAX))*2.0*initialMaxWeightValue;
                        previousWeightDiffDeltas[featureMapInPreviousLayer][featureMapInThisLayer][y][x]=0.0; // "previousWeightDiffs" needs to be zero-initialized
                    }
                }
            }
        }
    }
    else if(type==CNN_LAYER_TYPE_MAXPOOL)
    {
        // Modify CNN_LAYER_TYPE_CONV, too!

        if(_featureMapCount!=_previousLayerFeatureMapCount)
            throw;

        featureMapCount=_previousLayerFeatureMapCount;

        double _singleFeatureMapWidth=((double)(previousLayerSingleFeatureMapWidth-receptiveFieldWidth+2*zeroPaddingX))/((double)strideX)+1.0;
        double _singleFeatureMapHeight=((double)(previousLayerSingleFeatureMapHeight-receptiveFieldHeight+2*zeroPaddingY))/((double)strideY)+1.0;

        singleFeatureMapWidth=(int32_t)_singleFeatureMapWidth;
        singleFeatureMapHeight=(int32_t)_singleFeatureMapHeight;

        if(singleFeatureMapWidth!=_singleFeatureMapWidth||singleFeatureMapHeight!=_singleFeatureMapHeight)
            throw; // (See comment above)

        if(singleFeatureMapWidth<=0||singleFeatureMapHeight<=0) // Receptive field too large.
            throw;

        weights=0;
        biasWeights=0;
        previousWeightDiffDeltas=0;
        previousBiasWeightDiffDeltas=0;
        maxPixelMatrix=(double***)malloc(previousLayerFeatureMapCount*sizeof(double**));
        for(uint32_t featureMapInPreviousLayer=0;featureMapInPreviousLayer<previousLayerFeatureMapCount;featureMapInPreviousLayer++)
        {
            maxPixelMatrix[featureMapInPreviousLayer]=(double**)malloc(previousLayerSingleFeatureMapHeight*sizeof(double*));
            // These values do not need to be initialized.
            for(int32_t y=0;y<previousLayerSingleFeatureMapHeight;y++)
                maxPixelMatrix[featureMapInPreviousLayer][y]=(double*)malloc(previousLayerSingleFeatureMapWidth*sizeof(double));
        }
    }
    else if(type==CNN_LAYER_TYPE_RELU)
    {
        // Zero padding needed only to adjust output size (used in the calculation above)

        weights=0;
        biasWeights=0;
        maxPixelMatrix=0;

        featureMapCount=_previousLayerFeatureMapCount;
        singleFeatureMapWidth=_previousLayerSingleFeatureMapWidth;
        singleFeatureMapHeight=_previousLayerSingleFeatureMapHeight;
    }
    else if(type==CNN_LAYER_TYPE_SOFTMAX)
    {
        // Zero padding needed only to adjust output size (used in the calculation above)

        weights=0;
        biasWeights=0;
        maxPixelMatrix=0;

        if(_featureMapCount!=_previousLayerFeatureMapCount)
            throw;

        featureMapCount=_featureMapCount;
        singleFeatureMapWidth=1;
        singleFeatureMapHeight=1;
    }
    else if(type==CNN_LAYER_TYPE_FC)
    {
        // Zero padding needed only to adjust output size (used in the calculation above)

        featureMapCount=_featureMapCount;

        singleFeatureMapWidth=1;
        singleFeatureMapHeight=1;

        if(strideX!=1||strideY!=1)
            throw;

        maxPixelMatrix=0;

        double initialMaxWeightValue=0.1; // A FC layer can and should have negative weights.

        // This layer's dimensions: 1x1xneuronCount; featureMapCount=neuronCount

        weights=(double****)malloc(previousLayerFeatureMapCount*sizeof(double***));
        previousWeightDiffDeltas=(double****)malloc(previousLayerFeatureMapCount*sizeof(double***));
        for(uint32_t featureMapInPreviousLayer=0;featureMapInPreviousLayer<previousLayerFeatureMapCount;featureMapInPreviousLayer++)
        {
            weights[featureMapInPreviousLayer]=(double***)malloc(previousLayerSingleFeatureMapHeight*sizeof(double**));
            previousWeightDiffDeltas[featureMapInPreviousLayer]=(double***)malloc(previousLayerSingleFeatureMapHeight*sizeof(double**));
            for(int32_t y=0;y<previousLayerSingleFeatureMapHeight;y++)
            {
                weights[featureMapInPreviousLayer][y]=(double**)malloc(previousLayerSingleFeatureMapWidth*sizeof(double*));
                previousWeightDiffDeltas[featureMapInPreviousLayer][y]=(double**)malloc(previousLayerSingleFeatureMapWidth*sizeof(double*));
                for(int32_t x=0;x<previousLayerSingleFeatureMapWidth;x++)
                {
                    weights[featureMapInPreviousLayer][y][x]=(double*)malloc(featureMapCount*sizeof(double));
                    previousWeightDiffDeltas[featureMapInPreviousLayer][y][x]=(double*)malloc(featureMapCount*sizeof(double));
                    for(uint32_t featureMapInThisLayer=0;featureMapInThisLayer<featureMapCount;featureMapInThisLayer++)
                    {
                        weights[featureMapInPreviousLayer][y][x][featureMapInThisLayer]=-initialMaxWeightValue+(((double)rand())/((double)RAND_MAX))*2.0*initialMaxWeightValue;
                        previousWeightDiffDeltas[featureMapInPreviousLayer][y][x][featureMapInThisLayer]=0.0; // "previousWeightDiffs" needs to be zero-initialized
                    }
                }
            }
        }

        biasWeights=(double*)malloc(featureMapCount*sizeof(double));
        previousBiasWeightDiffDeltas=(double*)malloc(featureMapCount*sizeof(double));
        for(uint32_t featureMapInThisLayer=0;featureMapInThisLayer<featureMapCount;featureMapInThisLayer++)
        {
            biasWeights[featureMapInThisLayer]=0.0; // Initialize bias weights to 0
            previousBiasWeightDiffDeltas[featureMapInThisLayer]=0.0; // "previousBiasWeightDiffs" needs to be zero-initialized
        }
    }
    else
        throw;
}

CNNLayer::~CNNLayer()
{
    if(type==CNN_LAYER_TYPE_CONV)
    {
        for(uint32_t featureMapInPreviousLayer=0;featureMapInPreviousLayer<previousLayerFeatureMapCount;featureMapInPreviousLayer++)
        {
            for(uint32_t featureMapInThisLayer=0;featureMapInThisLayer<featureMapCount;featureMapInThisLayer++)
            {
                for(int32_t y=0;y<receptiveFieldHeight;y++)
                {
                    free(weights[featureMapInPreviousLayer][featureMapInThisLayer][y]);
                    free(previousWeightDiffDeltas[featureMapInPreviousLayer][featureMapInThisLayer][y]);
                }
                free(weights[featureMapInPreviousLayer][featureMapInThisLayer]);
                free(previousWeightDiffDeltas[featureMapInPreviousLayer][featureMapInThisLayer]);
            }
            free(weights[featureMapInPreviousLayer]);
            free(previousWeightDiffDeltas[featureMapInPreviousLayer]);
        }
        free(weights);
        free(previousWeightDiffDeltas);
        free(biasWeights);
        free(previousBiasWeightDiffDeltas);
    }
    else if(type==CNN_LAYER_TYPE_MAXPOOL)
    {
        for(uint32_t featureMapInPreviousLayer=0;featureMapInPreviousLayer<previousLayerFeatureMapCount;featureMapInPreviousLayer++)
        {
            for(int32_t y=0;y<previousLayerSingleFeatureMapHeight;y++)
                free(maxPixelMatrix[featureMapInPreviousLayer][y]);
            free(maxPixelMatrix[featureMapInPreviousLayer]);
        }
        free(maxPixelMatrix);
    }
    else if(type==CNN_LAYER_TYPE_FC)
    {
        for(uint32_t featureMapInPreviousLayer=0;featureMapInPreviousLayer<previousLayerFeatureMapCount;featureMapInPreviousLayer++)
        {
            for(int32_t y=0;y<previousLayerSingleFeatureMapHeight;y++)
            {
                for(int32_t x=0;x<previousLayerSingleFeatureMapWidth;x++)
                {
                    free(weights[featureMapInPreviousLayer][y][x]);
                    free(previousWeightDiffDeltas[featureMapInPreviousLayer][y][x]);
                }
                free(weights[featureMapInPreviousLayer][y]);
                free(previousWeightDiffDeltas[featureMapInPreviousLayer][y]);
            }
            free(weights[featureMapInPreviousLayer]);
            free(previousWeightDiffDeltas[featureMapInPreviousLayer]);
        }
        free(weights);
        free(previousWeightDiffDeltas);
        free(biasWeights);
        free(previousBiasWeightDiffDeltas);
    }

    if(output!=0)
        freeArray(output,featureMapCount,singleFeatureMapHeight);
    if(input!=0)
        freeArray(input,previousLayerFeatureMapCount,previousLayerSingleFeatureMapHeight);
}

void CNNLayer::freeArray(double ***_array, uint32_t zDimension, int32_t yDimension)
{
    for(uint32_t z=0;z<zDimension;z++)
    {
        for(int32_t y=0;y<yDimension;y++)
            free(_array[z][y]);
        free(_array[z]);
    }
    free(_array);
}

void CNNLayer::freeWeightTypeArray(double ****_array, uint32_t _dimension1, uint32_t _dimension2, int32_t _dimension3, int32_t _dimension4)
{
    // Disable "unused" warning
    (void)_dimension4;
    for(uint32_t d1=0;d1<_dimension1;d1++)
    {
        for(uint32_t d2=0;d2<_dimension2;d2++)
        {
            for(int32_t d3=0;d3<_dimension3;d3++)
                free(_array[d1][d2][d3]);
            free(_array[d1][d2]);
        }
        free(_array[d1]);
    }
    free(_array);
}

void CNNLayer::freeBiasTypeArray(double *_array)
{
    free(_array);
}

double ***CNNLayer::cloneArray(double ***_array, uint32_t zDimension, int32_t yDimension, int32_t xDimension)
{
    double ***out=(double***)malloc(zDimension*sizeof(double**));
    uint32_t yDimensionDoublePointerArraySize=yDimension*sizeof(double*);
    uint32_t xDimensionDoubleArraySize=xDimension*sizeof(double);
    for(uint32_t z=0;z<zDimension;z++)
    {
        out[z]=(double**)malloc(yDimensionDoublePointerArraySize);
        for(int32_t y=0;y<yDimension;y++)
        {
            out[z][y]=(double*)malloc(xDimensionDoubleArraySize);
            memcpy(out[z][y],_array[z][y],xDimensionDoubleArraySize);
        }
    }
    return out;
}

int32_t CNNLayer::getRequiredReceptiveFieldSizeForDesiredSingleFeatureMapSize(int32_t _previousLayerSingleFeatureMapSize, int32_t _desiredSingleFeatureMapSize, int32_t _stride, int32_t _zeroPadding)
{
    // NOTE/WARNING: The receptive field size should not be too big (>5 for conv layers, >2 for pooling layers), since this usually leads to poorer performance.
    double result=((double)_stride)*(1.0-((double)_desiredSingleFeatureMapSize))+((double)_previousLayerSingleFeatureMapSize)+2.0*((double)_zeroPadding);

    // If the result is not an integer/is less than or equal to 0, it means that this combination of inputs is invalid.
    if(floor(result)!=result||result<=0.0)
        throw;
    return (int32_t)result;
}

int32_t CNNLayer::getRequiredZeroPaddingForDesiredSingleFeatureMapAndReceptiveFieldSize(int32_t _previousLayerSingleFeatureMapSize, int32_t _desiredSingleFeatureMapSize, int32_t _desiredReceptiveFieldSize, int32_t _stride)
{
    // NOTE/WARNING: The receptive field size should not be too big (>5 for conv layers, >2 for pooling layers), since this usually leads to poorer performance.

    // If the result is not an integer/is less than or equal to 0, it means that this combination of inputs is invalid.
    double result=(((double)_stride)*(((double)_desiredSingleFeatureMapSize)-1.0)-((double)_previousLayerSingleFeatureMapSize)+((double)_desiredReceptiveFieldSize))/2.0;
    if(floor(result)!=result||result<=0.0)
        throw;
    return (int32_t)result;
}

double ***CNNLayer::conv(double ***_input)
{
    // Modify "maxpool"/"relu"/"fc"/"softmax", too!

    // Store for backpropagation

    if(input!=0)
        freeArray(input,previousLayerFeatureMapCount,previousLayerSingleFeatureMapHeight);
    input=cloneArray(_input,previousLayerFeatureMapCount,previousLayerSingleFeatureMapHeight,previousLayerSingleFeatureMapWidth);

    if(output!=0)
        freeArray(output,featureMapCount,singleFeatureMapHeight);
    output=(double***)malloc(featureMapCount*sizeof(double**));

    for(uint32_t featureMapInThisLayer=0;featureMapInThisLayer<featureMapCount;featureMapInThisLayer++)
    {
        output[featureMapInThisLayer]=(double**)malloc(singleFeatureMapHeight*sizeof(double*));

        for(int32_t y=0;y<singleFeatureMapHeight;y++)
        {
            // Initialize values with 0.0
            output[featureMapInThisLayer][y]=(double*)malloc(singleFeatureMapWidth*sizeof(double));
            for(int32_t x=0;x<singleFeatureMapWidth;x++)
                output[featureMapInThisLayer][y][x]=biasWeights[featureMapInThisLayer]; // Initialize output pixels with bias weights here to avoid having to add them later
        }

        // Store sums of pixel values in each feature map of the previous layer multiplied by their weights
        // in "output", then add bias.

        // Fill pixels of current feature map in this layer with weight-multiplied pixels of feature maps in previous layer
        for(int32_t y=0;y<singleFeatureMapHeight;y++)
        {
            for(int32_t x=0;x<singleFeatureMapWidth;x++)
            {
                int32_t offsetX=-zeroPaddingX+strideX*x;
                int32_t offsetY=-zeroPaddingY+strideY*y;
                for(uint32_t featureMapInPreviousLayer=0;featureMapInPreviousLayer<previousLayerFeatureMapCount;featureMapInPreviousLayer++)
                {
                    for(int32_t receptiveFieldY=0;receptiveFieldY<receptiveFieldHeight;receptiveFieldY++)
                    {
                        for(int32_t receptiveFieldX=0;receptiveFieldX<receptiveFieldWidth;receptiveFieldX++)
                        {
                            // Coordinates of pixel in feature map in previous layer:
                            // (note that we check whether such a pixel exists below)
                            int32_t pixelInFeatureMapInPreviousLayerX=offsetX+receptiveFieldX;
                            int32_t pixelInFeatureMapInPreviousLayerY=offsetY+receptiveFieldY;

                            bool inZeroPaddingField=pixelInFeatureMapInPreviousLayerX<0||pixelInFeatureMapInPreviousLayerY<0
                                                    ||pixelInFeatureMapInPreviousLayerX>=previousLayerSingleFeatureMapWidth
                                                    ||pixelInFeatureMapInPreviousLayerY>=previousLayerSingleFeatureMapHeight;

                            if(inZeroPaddingField)
                                continue;

                            output[featureMapInThisLayer][y][x]+=input[featureMapInPreviousLayer][pixelInFeatureMapInPreviousLayerY][pixelInFeatureMapInPreviousLayerX]
                                    *weights[featureMapInPreviousLayer][featureMapInThisLayer][receptiveFieldY][receptiveFieldX];
                        }
                    }
                }
            }
        }

        // Bias weights added during initialization
    }

    // Return a copy of "output" to prevent changes from being made to "output".

    return cloneArray(output,featureMapCount,singleFeatureMapHeight,singleFeatureMapWidth);
}

double ***CNNLayer::fc(double ***_input)
{
    // Modify "maxpool"/"relu"/"maxpool"/"softmax", too!

    // Store for backpropagation

    if(input!=0)
        freeArray(input,previousLayerFeatureMapCount,previousLayerSingleFeatureMapHeight);
    input=cloneArray(_input,previousLayerFeatureMapCount,previousLayerSingleFeatureMapHeight,previousLayerSingleFeatureMapWidth);

    if(output!=0)
        freeArray(output,featureMapCount,singleFeatureMapHeight);


    // featureMapCount=neuronCount
    output=(double***)malloc(featureMapCount*sizeof(double**));

    // Initialize output values with bias weights here to avoid having to add them later (since no activation function is used, this is permissible)

    for(uint32_t featureMapInThisLayer=0;featureMapInThisLayer<featureMapCount;featureMapInThisLayer++)
    {
        output[featureMapInThisLayer]=(double**)malloc(1*sizeof(double*));
        output[featureMapInThisLayer][0]=(double*)malloc(1*sizeof(double));
        output[featureMapInThisLayer][0][0]=biasWeights[featureMapInThisLayer];
    }

    for(uint32_t featureMapInPreviousLayer=0;featureMapInPreviousLayer<previousLayerFeatureMapCount;featureMapInPreviousLayer++)
    {
        for(int32_t y=0;y<previousLayerSingleFeatureMapHeight;y++)
        {
            for(int32_t x=0;x<previousLayerSingleFeatureMapWidth;x++)
            {
                for(uint32_t featureMapInThisLayer=0;featureMapInThisLayer<featureMapCount;featureMapInThisLayer++)
                    output[featureMapInThisLayer][0][0]+=
                            input[featureMapInPreviousLayer][y][x]*
                            weights[featureMapInPreviousLayer][y][x][featureMapInThisLayer];
            }
        }
    }

    // Return a copy of "output" to prevent changes from being made to "output".

    return cloneArray(output,featureMapCount,singleFeatureMapHeight,singleFeatureMapWidth);
}

double ***CNNLayer::maxpool(double ***_input)
{
    // Modify "conv"/"relu"/"fc"/"softmax", too!

    // A maxpool layer has the same depth as the layer preceding it

    // Store for backpropagation

    if(input!=0)
        freeArray(input,previousLayerFeatureMapCount,previousLayerSingleFeatureMapHeight);
    input=cloneArray(_input,previousLayerFeatureMapCount,previousLayerSingleFeatureMapHeight,previousLayerSingleFeatureMapWidth);

    if(output!=0)
        freeArray(output,featureMapCount,singleFeatureMapHeight);
    output=(double***)malloc(featureMapCount*sizeof(double**));

    // featureMapInPreviousLayer = featureMapInThisLayer (each depth slice is processed independently)

    for(uint32_t featureMap=0;featureMap<featureMapCount;featureMap++)
    {
        // featureMapInPreviousLayer = featureMapInThisLayer (each depth slice is processed independently)
        output[featureMap]=(double**)malloc(singleFeatureMapHeight*sizeof(double*));

        for(int32_t y=0;y<singleFeatureMapHeight;y++)
            output[featureMap][y]=(double*)malloc(singleFeatureMapWidth*sizeof(double));

        // Move over feature map in previous layer, map max pixels to pixels in feature map in this layer

        for(int32_t y=0;y<singleFeatureMapHeight;y++)
        {
            for(int32_t x=0;x<singleFeatureMapWidth;x++)
            {
                int32_t offsetX=-zeroPaddingX+strideX*x;
                int32_t offsetY=-zeroPaddingY+strideY*y;

                int32_t highestValueX=-1;
                int32_t highestValueY=-1;
                double highestValue=-std::numeric_limits<double>::max(); // Lowest possible value of type "sdouble"

                for(int32_t receptiveFieldY=0;receptiveFieldY<receptiveFieldHeight;receptiveFieldY++)
                {
                    for(int32_t receptiveFieldX=0;receptiveFieldX<receptiveFieldWidth;receptiveFieldX++)
                    {
                        // Coordinates of pixel in feature map in previous layer:
                        // (note that we check whether such a pixel exists below)
                        int32_t pixelInFeatureMapInPreviousLayerX=offsetX+receptiveFieldX;
                        int32_t pixelInFeatureMapInPreviousLayerY=offsetY+receptiveFieldY;

                        bool inZeroPaddingField=pixelInFeatureMapInPreviousLayerX<0||pixelInFeatureMapInPreviousLayerY<0
                                                ||pixelInFeatureMapInPreviousLayerX>=previousLayerSingleFeatureMapWidth
                                                ||pixelInFeatureMapInPreviousLayerY>=previousLayerSingleFeatureMapHeight;

                        if(inZeroPaddingField)
                            continue; // This pixel doesn't exist, so there's nothing to be set in "maxPixelMatrix"

                        double pixelValue=input[featureMap][pixelInFeatureMapInPreviousLayerY][pixelInFeatureMapInPreviousLayerX];
                        if(pixelValue>highestValue)
                        {
                            highestValue=pixelValue;
                            highestValueY=pixelInFeatureMapInPreviousLayerY;
                            highestValueX=pixelInFeatureMapInPreviousLayerX;
                        }
                        // We still don't know whether this will be the highest pixel, so we only set the highest pixel's maxPixelMatrix value once we're sure.
                        maxPixelMatrix[featureMap][pixelInFeatureMapInPreviousLayerY][pixelInFeatureMapInPreviousLayerX]=0.0;
                    }
                }

                // Store coordinates (for backpropagation):

                maxPixelMatrix[featureMap][highestValueY][highestValueX]=1.0;

                // Set value of pixel in this layer's feature map to the highest value found.

                output[featureMap][y][x]=highestValue;
            }
        }
    }

    // Return a copy of "output" to prevent changes from being made to "output".

    return cloneArray(output,featureMapCount,singleFeatureMapHeight,singleFeatureMapWidth);
}

double ***CNNLayer::relu(double ***_input)
{
    // Modify "conv"/"maxpool"/"fc"/"softmax", too!

    // Store for backpropagation

    if(input!=0)
        freeArray(input,previousLayerFeatureMapCount,previousLayerSingleFeatureMapHeight);
    input=cloneArray(_input,previousLayerFeatureMapCount,previousLayerSingleFeatureMapHeight,previousLayerSingleFeatureMapWidth);

    if(output!=0)
        freeArray(output,featureMapCount,singleFeatureMapHeight);
    output=(double***)malloc(featureMapCount*sizeof(double**));


    // Note that a relu layer has exactly the same dimensions as the layer preceding it,
    // thus featureMapInPreviousLayer==featureMapInThisLayer, etc.

    for(uint32_t featureMap=0;featureMap<previousLayerFeatureMapCount;featureMap++)
    {
        output[featureMap]=(double**)malloc(previousLayerSingleFeatureMapHeight*sizeof(double*)); // See comment above for explanation.
        for(int32_t y=0;y<previousLayerSingleFeatureMapHeight;y++)
        {
            output[featureMap][y]=(double*)malloc(previousLayerSingleFeatureMapWidth*sizeof(double)); // See comment above for explanation.
            for(int32_t x=0;x<previousLayerSingleFeatureMapWidth;x++)
            {
                 output[featureMap][y][x]=__max(0.0,input[featureMap][y][x]); // See comment above for explanation.
            }
        }
    }

    // Return a copy of "output" to prevent changes from being made to "output".

    return cloneArray(output,featureMapCount,singleFeatureMapHeight,singleFeatureMapWidth);
}

double ***CNNLayer::softmax(double ***_input)
{
    // Modify "conv"/"maxpool"/"fc"/"relu", too!

    // A softmax layer has the same depth (feature count) as the layer preceding it (intended to be used after a FC layer)

    // Store for backpropagation

    if(input!=0)
        freeArray(input,previousLayerFeatureMapCount,previousLayerSingleFeatureMapHeight);
    input=cloneArray(_input,previousLayerFeatureMapCount,previousLayerSingleFeatureMapHeight,previousLayerSingleFeatureMapWidth);

    if(output!=0)
        freeArray(output,featureMapCount,singleFeatureMapHeight);
    output=(double***)malloc(featureMapCount*sizeof(double**));

    // READ THIS:
    // A softmax layer has the same depth (feature count) as the layer preceding it (intended to be used after a FC layer)
    // Thus, featureMapInThisLayer=featureMapInPreviousLayer

    // Compute highest activation

    double highestValue=-std::numeric_limits<double>::max();

    for(uint32_t featureMap=0;featureMap<featureMapCount;featureMap++)
    {
        double value=input[featureMap][0][0];
        if(value>highestValue)
            highestValue=value;
    }

    double ePowSum=0.0;

    for(uint32_t featureMap=0;featureMap<featureMapCount;featureMap++)
    {
        double value=input[featureMap][0][0];
        ePowSum+=exp(value-highestValue);
    }

    for(uint32_t featureMap=0;featureMap<featureMapCount;featureMap++)
    {
        double value=input[featureMap][0][0];
        output[featureMap]=(double**)malloc(1*sizeof(double*));
        output[featureMap][0]=(double*)malloc(1*sizeof(double));
        output[featureMap][0][0]=exp(value-highestValue)/ePowSum;
    }

    // Return a copy of "output" to prevent changes from being made to "output".

    return cloneArray(output,featureMapCount,singleFeatureMapHeight,singleFeatureMapWidth);
}

void CNNLayer::calculateConvDiffs(double ****&weightDiffs, double *&biasWeightDiffs, double ***outputDiffs, double ***&inputDiffs)
{
    weightDiffs=(double****)malloc(previousLayerFeatureMapCount*sizeof(double***));
    biasWeightDiffs=(double*)malloc(featureMapCount*sizeof(double));
    inputDiffs=(double***)malloc(previousLayerFeatureMapCount*sizeof(double**));

    for(uint32_t featureMapInThisLayer=0;featureMapInThisLayer<featureMapCount;featureMapInThisLayer++)
        biasWeightDiffs[featureMapInThisLayer]=0.0;

    for(uint32_t featureMapInPreviousLayer=0;featureMapInPreviousLayer<previousLayerFeatureMapCount;featureMapInPreviousLayer++)
    {
        weightDiffs[featureMapInPreviousLayer]=(double***)malloc(featureMapCount*sizeof(double**));
        inputDiffs[featureMapInPreviousLayer]=(double**)malloc(previousLayerSingleFeatureMapHeight*sizeof(double*));
        // Initialize input diffs:
        for(int32_t y=0;y<previousLayerSingleFeatureMapHeight;y++)
        {
            inputDiffs[featureMapInPreviousLayer][y]=(double*)malloc(previousLayerSingleFeatureMapWidth*sizeof(double));
            for(int32_t x=0;x<previousLayerSingleFeatureMapWidth;x++)
                inputDiffs[featureMapInPreviousLayer][y][x]=0.0;
        }

        for(uint32_t featureMapInThisLayer=0;featureMapInThisLayer<featureMapCount;featureMapInThisLayer++)
        {
            weightDiffs[featureMapInPreviousLayer][featureMapInThisLayer]=(double**)malloc(receptiveFieldHeight*sizeof(double*));
            // Initialize weight diffs:
            for(int32_t receptiveFieldY=0;receptiveFieldY<receptiveFieldHeight;receptiveFieldY++)
            {
                weightDiffs[featureMapInPreviousLayer][featureMapInThisLayer][receptiveFieldY]=(double*)malloc(receptiveFieldWidth*sizeof(double));
                for(int32_t receptiveFieldX=0;receptiveFieldX<receptiveFieldWidth;receptiveFieldX++)
                    weightDiffs[featureMapInPreviousLayer][featureMapInThisLayer][receptiveFieldY][receptiveFieldX]=0.0;
            }
            // biasWeightDiffs already initialized above.
            for(int32_t y=0;y<singleFeatureMapHeight;y++)
            {
                for(int32_t x=0;x<singleFeatureMapWidth;x++)
                {
                    int32_t offsetX=-zeroPaddingX+strideX*x;
                    int32_t offsetY=-zeroPaddingY+strideY*y;
                    double outputValue=output[featureMapInThisLayer][y][x];

                    // Derivative of the loss function w.r.t. the value inside of the activation function call

                    double errorTerm=outputDiffs[featureMapInThisLayer][y][x]; //Derivative of the loss function w.r.t. the value of the pixel in the current feature map of this layer

                    for(int32_t receptiveFieldY=0;receptiveFieldY<receptiveFieldHeight;receptiveFieldY++)
                    {
                        for(int32_t receptiveFieldX=0;receptiveFieldX<receptiveFieldWidth;receptiveFieldX++)
                        {
                            // Coordinates of pixel in feature map in previous layer:
                            // (note that we check whether such a pixel exists below)
                            int32_t pixelInFeatureMapInPreviousLayerX=offsetX+receptiveFieldX;
                            int32_t pixelInFeatureMapInPreviousLayerY=offsetY+receptiveFieldY;

                            bool inZeroPaddingField=pixelInFeatureMapInPreviousLayerX<0||pixelInFeatureMapInPreviousLayerY<0
                                                    ||pixelInFeatureMapInPreviousLayerX>=previousLayerSingleFeatureMapWidth
                                                    ||pixelInFeatureMapInPreviousLayerY>=previousLayerSingleFeatureMapHeight;

                            if(inZeroPaddingField)
                                continue;

                            // We compute the error term sum indirectly by looping over each calculation made to calculate the output of this layer:

                            // Derivative of the loss function w.r.t. this weight
                            weightDiffs[featureMapInPreviousLayer][featureMapInThisLayer][receptiveFieldY][receptiveFieldX] +=
                                    errorTerm*input[featureMapInPreviousLayer][pixelInFeatureMapInPreviousLayerY][pixelInFeatureMapInPreviousLayerX];
                            // Derivative of the loss function w.r.t. the bias of this layer

                            // Derivative of the loss function w.r.t. the value of the input pixel (in the current feature map of the previous layer)

                            // Check this for errors again, then inspect layer structure.

                            inputDiffs[featureMapInPreviousLayer][pixelInFeatureMapInPreviousLayerY][pixelInFeatureMapInPreviousLayerX]+=
                                    errorTerm*weights[featureMapInPreviousLayer][featureMapInThisLayer][receptiveFieldY][receptiveFieldX];
                        }
                    }
                    // The bias is applied once to each output pixel
                    biasWeightDiffs[featureMapInThisLayer]+=errorTerm;
                }
            }
        }
    }
}

void CNNLayer::calculateFcDiffs(double ****&weightDiffs, double *&biasWeightDiffs, double ***outputDiffs, double ***&inputDiffs)
{
    // featureMapCount=neuronCount

    // Initialize biasWeightDiffs

    // "biasWeightDiffs" does not need to be zero-initialized
    biasWeightDiffs=(double*)malloc(featureMapCount*sizeof(double));

    // Initialize weightDiffs and inputDiffs

    weightDiffs=(double****)malloc(previousLayerFeatureMapCount*sizeof(double***));
    inputDiffs=(double***)malloc(previousLayerFeatureMapCount*sizeof(double**));

    for(uint32_t featureMapInPreviousLayer=0;featureMapInPreviousLayer<previousLayerFeatureMapCount;featureMapInPreviousLayer++)
    {
        weightDiffs[featureMapInPreviousLayer]=(double***)malloc(previousLayerSingleFeatureMapHeight*sizeof(double**));
        inputDiffs[featureMapInPreviousLayer]=(double**)malloc(previousLayerSingleFeatureMapHeight*sizeof(double*));
        for(int32_t previousLayerY=0;previousLayerY<previousLayerSingleFeatureMapHeight;previousLayerY++)
        {
            weightDiffs[featureMapInPreviousLayer][previousLayerY]=(double**)malloc(previousLayerSingleFeatureMapWidth*sizeof(double*));
            inputDiffs[featureMapInPreviousLayer][previousLayerY]=(double*)malloc(previousLayerSingleFeatureMapWidth*sizeof(double));
            for(int32_t previousLayerX=0;previousLayerX<previousLayerSingleFeatureMapWidth;previousLayerX++)
            {
                weightDiffs[featureMapInPreviousLayer][previousLayerY][previousLayerX]=(double*)malloc(featureMapCount*sizeof(double));
                inputDiffs[featureMapInPreviousLayer][previousLayerY][previousLayerX]=0.0;
                // "weightDiffs" does not need to be zero-initialized.
            }
        }
    }

    // Calculate diffs

    for(uint32_t featureMapInThisLayer=0;featureMapInThisLayer<featureMapCount;featureMapInThisLayer++)
    {
        double errorTerm=outputDiffs[featureMapInThisLayer][0][0];
        for(uint32_t featureMapInPreviousLayer=0;featureMapInPreviousLayer<previousLayerFeatureMapCount;featureMapInPreviousLayer++)
        {
            for(int32_t previousLayerY=0;previousLayerY<previousLayerSingleFeatureMapHeight;previousLayerY++)
            {
                for(int32_t previousLayerX=0;previousLayerX<previousLayerSingleFeatureMapWidth;previousLayerX++)
                {
                    double inputValue=input[featureMapInPreviousLayer][previousLayerY][previousLayerX];

                    weightDiffs[featureMapInPreviousLayer][previousLayerY][previousLayerX][featureMapInThisLayer]=errorTerm*inputValue;
                    inputDiffs[featureMapInPreviousLayer][previousLayerY][previousLayerX]+=errorTerm*weights[featureMapInPreviousLayer][previousLayerY][previousLayerX][featureMapInThisLayer];
                }
            }
        }
        // The bias is applied once to each output neuron
        biasWeightDiffs[featureMapInThisLayer]=errorTerm;
    }
}

void CNNLayer::calculateMaxpoolDiffs(double ***outputDiffs, double ***&inputDiffs)
{
    // A maxpool layer has no weight/bias diffs; it only re-routes the gradients from outputDiffs to the pixels with the highest values (into inputDiffs) during backpropagation.

    // Also note that a maxpool layer has exactly the same _depth_ as the layer preceding it.

    inputDiffs=(double***)malloc(previousLayerFeatureMapCount*sizeof(double**));

    for(uint32_t featureMap=0;featureMap<previousLayerFeatureMapCount;featureMap++)
    {
        inputDiffs[featureMap]=(double**)malloc(previousLayerSingleFeatureMapHeight*sizeof(double*));
        // Initialize input diffs:
        for(int32_t y=0;y<previousLayerSingleFeatureMapHeight;y++)
        {
            inputDiffs[featureMap][y]=(double*)malloc(previousLayerSingleFeatureMapWidth*sizeof(double));
            for(int32_t x=0;x<previousLayerSingleFeatureMapWidth;x++)
                inputDiffs[featureMap][y][x]=0.0;
        }
        for(int32_t y=0;y<singleFeatureMapHeight;y++)
        {
            for(int32_t x=0;x<singleFeatureMapWidth;x++)
            {
                int32_t offsetX=-zeroPaddingX+strideX*x;
                int32_t offsetY=-zeroPaddingY+strideY*y;

                for(int32_t receptiveFieldY=0;receptiveFieldY<receptiveFieldHeight;receptiveFieldY++)
                {
                    for(int32_t receptiveFieldX=0;receptiveFieldX<receptiveFieldWidth;receptiveFieldX++)
                    {
                        // Coordinates of pixel in feature map in previous layer:
                        // (note that we check whether such a pixel exists below)
                        int32_t pixelInFeatureMapInPreviousLayerX=offsetX+receptiveFieldX;
                        int32_t pixelInFeatureMapInPreviousLayerY=offsetY+receptiveFieldY;

                        bool inZeroPaddingField=pixelInFeatureMapInPreviousLayerX<0||pixelInFeatureMapInPreviousLayerY<0
                                                ||pixelInFeatureMapInPreviousLayerX>=previousLayerSingleFeatureMapWidth
                                                ||pixelInFeatureMapInPreviousLayerY>=previousLayerSingleFeatureMapHeight;

                        if(inZeroPaddingField)
                            continue;

                        // maxPixelMatrix[...][...][...] contains 1.0 if this was the pixel with the highest value, or else 0.0

                        // The values of isMaxPixel determines whether the gradient of this feature map's [x,y] pixel is routed to the pixel at [pixelInFeatureMapInPreviousLayerX,pixelInFeatureMapInPreviousLayerY] or not.
                        double isMaxPixel=maxPixelMatrix[featureMap][pixelInFeatureMapInPreviousLayerY][pixelInFeatureMapInPreviousLayerX];
                        inputDiffs[featureMap][pixelInFeatureMapInPreviousLayerY][pixelInFeatureMapInPreviousLayerX]+=
                                isMaxPixel*outputDiffs[featureMap][y][x];
                        if(isMaxPixel>0.0)
                            goto NextOutputPixel; // Found max value for this output pixel (inputDiff values get zero-initialized above)
                    }
                }
                NextOutputPixel:
                continue;
            }
        }
    }
}

void CNNLayer::calculateReluDiffs(double ***outputDiffs, double ***&inputDiffs)
{
    // Just pass on the gradients of all output pixels that received a value higher than 0.0.
    // Also note that a relu layer has exactly the same dimensions as the layer preceding it,
    // thus featureMapInPreviousLayer==featureMapInThisLayer, etc.

    inputDiffs=(double***)malloc(previousLayerFeatureMapCount*sizeof(double**));
    for(uint32_t featureMap=0;featureMap<previousLayerFeatureMapCount;featureMap++)
    {
        inputDiffs[featureMap]=(double**)malloc(previousLayerSingleFeatureMapHeight*sizeof(double*));
        for(int32_t y=0;y<previousLayerSingleFeatureMapHeight;y++)
        {
            inputDiffs[featureMap][y]=(double*)malloc(previousLayerSingleFeatureMapWidth*sizeof(double));
            for(int32_t x=0;x<previousLayerSingleFeatureMapWidth;x++)
            {
                inputDiffs[featureMap][y][x]=(output[featureMap][y][x]>0.0/*featureMapInPreviousLayer=featureMapInThisLayer (see comment above)*/?1.0:0.0)*outputDiffs[featureMap][y][x];
            }
        }
    }
}

void CNNLayer::calculateSoftmaxDiffs(double ***&inputDiffs, uint32_t desiredLabel)
{
    // Note that a softmax layer has exactly the same depth as the layer preceding it,
    // and the depth is always equal to the amount of classes.
    // The dimension of a softmax layer is always equal to 1 x 1 x featureMapCount.

    inputDiffs=(double***)malloc(featureMapCount*sizeof(double**));

    for(uint32_t featureMap=0;featureMap<featureMapCount;featureMap++)
    {
        inputDiffs[featureMap]=(double**)malloc(1*sizeof(double*));
        inputDiffs[featureMap][0]=(double*)malloc(1*sizeof(double));

        inputDiffs[featureMap][0][0]=-((featureMap==desiredLabel?1.0:0.0)-output[featureMap][0][0]);
    }
}

void CNNLayer::applyConvDiffs(double ****weightDiffs, double *biasWeightDiffs, double learningRate, double momentum, double weightDecay)
{
    if(type!=CNN_LAYER_TYPE_CONV)
        throw;

    // Adjust bias weight of each feature map
    for(uint32_t featureMapInThisLayer=0;featureMapInThisLayer<featureMapCount;featureMapInThisLayer++)
    {
        double previousDelta=previousBiasWeightDiffDeltas[featureMapInThisLayer];
        double currentWeight=biasWeights[featureMapInThisLayer];
        double weightDiff=biasWeightDiffs[featureMapInThisLayer];
        double thisDelta=(1.0-momentum)*-learningRate*weightDiff+momentum*previousDelta-weightDecay*currentWeight;
        biasWeights[featureMapInThisLayer]+=thisDelta;
        previousBiasWeightDiffDeltas[featureMapInThisLayer]=thisDelta;
    }
    for(uint32_t featureMapInPreviousLayer=0;featureMapInPreviousLayer<previousLayerFeatureMapCount;featureMapInPreviousLayer++)
    {
        for(uint32_t featureMapInThisLayer=0;featureMapInThisLayer<featureMapCount;featureMapInThisLayer++)
        {
            for(int32_t receptiveFieldY=0;receptiveFieldY<receptiveFieldHeight;receptiveFieldY++)
            {
                for(int32_t receptiveFieldX=0;receptiveFieldX<receptiveFieldWidth;receptiveFieldX++)
                {
                    double previousDelta=previousWeightDiffDeltas[featureMapInPreviousLayer][featureMapInThisLayer][receptiveFieldY][receptiveFieldX];
                    double currentWeight=weights[featureMapInPreviousLayer][featureMapInThisLayer][receptiveFieldY][receptiveFieldX];
                    double weightDiff=weightDiffs[featureMapInPreviousLayer][featureMapInThisLayer][receptiveFieldY][receptiveFieldX];
                    double thisDelta=(1.0-momentum)*-learningRate*weightDiff+momentum*previousDelta-weightDecay*currentWeight;

                    weights[featureMapInPreviousLayer][featureMapInThisLayer][receptiveFieldY][receptiveFieldX]+=thisDelta;
                    previousWeightDiffDeltas[featureMapInPreviousLayer][featureMapInThisLayer][receptiveFieldY][receptiveFieldX]=thisDelta;
                }
            }
        }
    }
}

void CNNLayer::applyFcDiffs(double ****weightDiffs, double *biasWeightDiffs, double learningRate, double momentum, double weightDecay)
{
    if(type!=CNN_LAYER_TYPE_FC)
        throw;

    // Adjust bias weight of each neuron
    for(uint32_t featureMapInThisLayer=0;featureMapInThisLayer<featureMapCount;featureMapInThisLayer++)
    {
        double previousDelta=previousBiasWeightDiffDeltas[featureMapInThisLayer];
        double currentWeight=biasWeights[featureMapInThisLayer];
        double weightDiff=biasWeightDiffs[featureMapInThisLayer];
        double thisDelta=(1.0-momentum)*-learningRate*weightDiff+momentum*previousDelta-weightDecay*currentWeight;

        biasWeights[featureMapInThisLayer]+=thisDelta;
        previousBiasWeightDiffDeltas[featureMapInThisLayer]=thisDelta;
    }
    for(uint32_t featureMapInPreviousLayer=0;featureMapInPreviousLayer<previousLayerFeatureMapCount;featureMapInPreviousLayer++)
    {
        for(int32_t y=0;y<previousLayerSingleFeatureMapHeight;y++)
        {
            for(int32_t x=0;x<previousLayerSingleFeatureMapWidth;x++)
            {
                for(uint32_t featureMapInThisLayer=0;featureMapInThisLayer<featureMapCount;featureMapInThisLayer++)
                {
                    double previousDelta=previousWeightDiffDeltas[featureMapInPreviousLayer][y][x][featureMapInThisLayer];
                    double currentWeight=weights[featureMapInPreviousLayer][y][x][featureMapInThisLayer];
                    double weightDiff=weightDiffs[featureMapInPreviousLayer][y][x][featureMapInThisLayer];
                    double thisDelta=(1.0-momentum)*-learningRate*weightDiff+momentum*previousDelta-weightDecay*currentWeight;

                    weights[featureMapInPreviousLayer][y][x][featureMapInThisLayer]+=thisDelta;
                    previousWeightDiffDeltas[featureMapInPreviousLayer][y][x][featureMapInThisLayer]=thisDelta;
                }
            }
        }
    }
}

double ***CNNLayer::forwardPass(double ***_input)
{
    if(type==CNN_LAYER_TYPE_CONV)
        return conv(_input);
    else if(type==CNN_LAYER_TYPE_MAXPOOL)
        return maxpool(_input);
    else if(type==CNN_LAYER_TYPE_RELU)
        return relu(_input);
    else if(type==CNN_LAYER_TYPE_FC)
        return fc(_input);
    else if(type==CNN_LAYER_TYPE_SOFTMAX)
        return softmax(_input);
    else
        return 0;
}

void CNNLayer::calculateDiffs(double ****&weightDiffs, double *&biasWeightDiffs, double ***outputDiffs, double ***&inputDiffs, uint32_t desiredLabel)
{
    if(type==CNN_LAYER_TYPE_CONV)
    {
        calculateConvDiffs(weightDiffs,biasWeightDiffs,outputDiffs,inputDiffs);
    }
    else if(type==CNN_LAYER_TYPE_MAXPOOL)
    {
        calculateMaxpoolDiffs(outputDiffs,inputDiffs);
        weightDiffs=0;
        biasWeightDiffs=0;
    }
    else if(type==CNN_LAYER_TYPE_RELU)
    {
        calculateReluDiffs(outputDiffs,inputDiffs);
        weightDiffs=0;
        biasWeightDiffs=0;
    }
    else if(type==CNN_LAYER_TYPE_FC)
    {
        calculateFcDiffs(weightDiffs,biasWeightDiffs,outputDiffs,inputDiffs);
    }
    else if(type==CNN_LAYER_TYPE_SOFTMAX)
    {
        calculateSoftmaxDiffs(inputDiffs,desiredLabel);
        weightDiffs=0;
        biasWeightDiffs=0;
    }
}

void CNNLayer::applyDiffs(double ****weightDiffs, double *biasWeightDiffs, double learningRate, double momentum, double weightDecay)
{
    if(type==CNN_LAYER_TYPE_CONV)
        applyConvDiffs(weightDiffs,biasWeightDiffs,learningRate,momentum,weightDecay);
    else if(type==CNN_LAYER_TYPE_FC)
        applyFcDiffs(weightDiffs,biasWeightDiffs,learningRate,momentum,weightDecay);
}
