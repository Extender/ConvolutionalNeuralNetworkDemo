#ifndef CNNLAYER_H
#define CNNLAYER_H

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#define CNN_LAYER_TYPE_CONV 1
#define CNN_LAYER_TYPE_MAXPOOL 2
#define CNN_LAYER_TYPE_RELU 3
#define CNN_LAYER_TYPE_FC 4 // Fully connected layer, just like a feedforward neural network layer. The input to the first fully connected layer is the set of all features maps at the layer below. Must be of dimension 1x1xneuronCount
#define CNN_LAYER_TYPE_SOFTMAX 5 // Softmax layer; can only follow a FC layer. Must be of dimension 1x1xclassCount, where classCount=previousLayerNeuronCount=previousLayerFeatureMapCount

#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <vector>
#include <string>
#include <ctime>
#include <limits>
#include <iostream>

#include "../_DefaultLibrary/text.h"

class CNNLayer
{
public:
    // A feature map is a filter layer of a layer (example: R layer of a RGB layer)
    uint32_t featureMapCount; // Amount of feature maps in this layer
    int32_t singleFeatureMapWidth; // Width of a single feature map
    int32_t singleFeatureMapHeight; // Height of a single feature map
    int32_t receptiveFieldWidth; // How many width pixels in the previous layer are to be translated into one pixel in this layer
    int32_t receptiveFieldHeight; // How many height pixels in the previous layer are to be translated into one pixel in this layer
    int32_t totalReceptiveFieldSize; // receptiveFieldWidth * receptiveFieldHeight
    uint32_t strideX; // Horizontal stride of neurons to be used (determines feature map width/height)
    uint32_t strideY; // Vertical stride of neurons to be used (determines feature map width/height)

    uint32_t previousLayerFeatureMapCount; // Amount of feature maps in previous layer
    int32_t previousLayerSingleFeatureMapWidth; // The width of a single feature map in previous layer
    int32_t previousLayerSingleFeatureMapHeight; // The height of a single feature map in previous layer

    int32_t zeroPaddingX; // Horizontal zero padding to be used in the output
    int32_t zeroPaddingY; // Vertical zero padding to be used with the input

    uint32_t layerId;

    uint8_t type; // Type of this layer

    // Dimensions: feature map in previous layer -> row of pixels -> value of pixel at x coordinate (1.0)
    double ***maxPixelMatrix; // Store the coordinates of the pixels with the highest values for use in backpropagation (1.0 for highest pixel, else 0.0).

    // Dimensions for CONV: feature map in previous layer -> feature map in this layer -> row of receptive field pixel -> weight of receptive field pixel at x coordinate
    // ("receptive field pixel to pixel in this layer"-weights)

    // Dimensions for FC: feature map in previous layer -> row of pixel in feature map in previous layer -> pixel in feature map in previous layer -> weight of connection of pixel in feature map in previous layer to neuron in this layer
    double ****weights;

    // Dimensions for CONV: feature map in this layer (bias of receptive field)
    // ("pixel in this layer"-bias weights)
    // Dimensions for FC: feature map in this layer (1 neuron = 1 feature map)
    double *biasWeights;

    double ****previousWeightDiffDeltas;
    double *previousBiasWeightDiffDeltas;

    // Store for backpropagation:

    // Dimensions:
    // Feature map -> row of pixels in feature map -> value of pixel at x coordinate

    double ***input;
    double ***output;

    static double sig(double input); // sigmoid function
    static double tanh(double input); // tanh function

    // Single feature map width/height calculated from receptiveFieldWidth/receptiveFieldHeight.

    // Note that a relu layer has exactly the same dimensions as the layer preceding it.
    // Note that a maxpool layer has exactly the same _depth_ as the layer preceding it.

    // For constructing FC layers: use _featureMapCount=1
    CNNLayer(uint32_t _layerId,uint8_t _type,uint32_t _featureMapCount,int32_t _receptiveFieldWidth,int32_t _receptiveFieldHeight,uint32_t _strideX /*Default: 1*/,uint32_t _strideY /*Default: 1*/,uint32_t _zeroPaddingX,uint32_t _zeroPaddingY,uint32_t _previousLayerFeatureMapCount,int32_t _previousLayerSingleFeatureMapWidth,int32_t _previousLayerSingleFeatureMapHeight);
    ~CNNLayer();

    static void freeArray(double ***_array,uint32_t zDimension,int32_t yDimension);
    static void freeWeightTypeArray(double ****_array, uint32_t _dimension1, uint32_t _dimension2, int32_t _dimension3, int32_t _dimension4);
    static void freeBiasTypeArray(double *_array);
    double ***cloneArray(double ***_array,uint32_t zDimension,int32_t yDimension,int32_t xDimension);

    // Can be used to calculate both the width and the height of the required receptive field size:

    // NOTE/WARNING: The receptive field size should not be too big (>5 for conv layers, >2 for pooling layers), since this usually leads to poorer performance.
    static int32_t getRequiredReceptiveFieldSizeForDesiredSingleFeatureMapSize(int32_t _previousLayerSingleFeatureMapSize,int32_t _desiredSingleFeatureMapSize,int32_t _stride,int32_t _zeroPadding);
    // WARNING: The zero padding returned must be used in the _previous_ layer, not in this layer!
    static int32_t getRequiredZeroPaddingForDesiredSingleFeatureMapAndReceptiveFieldSize(int32_t _previousLayerSingleFeatureMapSize,int32_t _desiredSingleFeatureMapSize,int32_t _desiredReceptiveFieldSize,int32_t _stride);

    double ***conv(double ***_input);
    double ***fc(double ***_input);
    // A maxpool layer has the same depth as the layer preceding it
    double ***maxpool(double ***_input);
    double ***relu(double ***_input);
    // Note that all input values have to be positive in order for the softmax layer to work
    // A softmax layer has the same depth (feature count) as the layer preceding it (intended to be used after a FC layer)
    double ***softmax(double ***_input);

    // Learning functions:

    // Weight diff dimensions: feature map in previous layer -> feature map in this layer -> row of receptive field pixels -> diff of weight of receptive field pixel at x coordinate
    // Bias diff dimensions: feature map in previous layer -> diff of bias weight of feature map in this layer
    // Output diff dimensions: feature map in this layer -> row of pixels -> diff of pixel at x coordinate
    // outputDiffs: diffs of pixels; inputDiffs: diffs of pixels in previous layer (to be passed as outputDiffs to the next layer)
    void calculateConvDiffs(double ****&weightDiffs,double *&biasWeightDiffs,double ***outputDiffs,double ***&inputDiffs);
    void calculateFcDiffs(double ****&weightDiffs, double *&biasWeightDiffs, double ***outputDiffs, double ***&inputDiffs);
    void calculateMaxpoolDiffs(double ***outputDiffs,double ***&inputDiffs);
    void calculateReluDiffs(double ***outputDiffs,double ***&inputDiffs);
    // The softmax diff calculation function needs the desired values to compute the input diffs (remember that the feature map count of a softmax layer is always 1)
    void calculateSoftmaxDiffs(double ***&inputDiffs,  uint32_t desiredLabel);
    void applyConvDiffs(double ****weightDiffs, double *biasWeightDiffs, double learningRate, double momentum, double weightDecay);
    void applyFcDiffs(double ****weightDiffs, double *biasWeightDiffs, double learningRate, double momentum, double weightDecay);

    // Universal functions:

    // Input dimensions:  feature maps of previous layer -> rows (y) -> columns (x)
    // Output dimensions: feature maps of this layer -> rows (y) -> columns (x)
    double ***forwardPass(double ***_input);

    void calculateDiffs(double ****&weightDiffs,double *&biasWeightDiffs,double ***outputDiffs,double ***&inputDiffs,uint32_t desiredLabel);
    void applyDiffs(double ****weightDiffs, double *biasWeightDiffs, double learningRate, double momentum, double weightDecay);
};

#endif // CNNLAYER_H
