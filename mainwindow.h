#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <ctime>
#include <algorithm>
#include <map>

#include <QMainWindow>
#include <QGraphicsScene>
#include <QGraphicsItem>
#include <QFile>

#include "cnnlayer.h"
#include "graphicssceneex.h"
#include "trainingthread.h"

namespace Ui {
class MainWindow;
}

class TrainingThread;

struct ResultVectorLessThanKey
{
    bool operator() (const std::pair<uint8_t,double> e1,const std::pair<uint8_t,double> e2)
    {
        return e1.second>e2.second; // Sort in descending order
    }
};

#define IMAGE_DATA_DIR "%APP_DIR%/cifar-10/"
#define IMAGE_WIDTH 32
#define IMAGE_HEIGHT 32
#define IMAGES_PER_BATCH 10000
#define BATCH_COUNT 5
#define BYTES_PER_IMAGE_IN_FILE 3073 // 1 label byte + 3072 color bytes
#define LABEL_COUNT 10
#define LAYER_COUNT 11
#define DEFAULT_LEARNING_RATE 0.005 // 0.005
#define DEFAULT_MOMENTUM 0.1 // 0.1
#define DEFAULT_WEIGHT_DECAY 0.0001
#define SHOW_FIRST_RESULTS 4
#define ACCURACY_VECTOR_MAX_SIZE 100

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    GraphicsSceneEx *scene;
    QGraphicsPixmapItem *pixmapItem;
    uint32_t **imageData;
    uint8_t *imageLabels;
    uint32_t examplesSeen;
    uint32_t currentImageId;
    TrainingThread *trainingThread;
    bool training;
    bool classified;
    // Cache of desired output values for all labels
    double ****desiredOutputValueCache;
    // Dimensions: image with id -> feature map (R, G or B channel) -> pixel row -> value of pixel in column
    double ****imageInputData;

    std::vector<double> *accuracyVector;

    CNNLayer **layers;

    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    static QString getLabelName(uint8_t label);
    void loadImage(uint32_t imageId);
    static uint32_t getHighestIndex(double *array,uint32_t elementCount);
    void displayOutput(double ***output, uint8_t correctLabel);

public slots:
    void updateExamplesSeenLbl();
    void nextBtnClicked();
    void classifyBtnClicked();
    void trainBtnClicked();
    void learningRateBoxValueChanged(double newValue);
    void momentumBoxValueChanged(double newValue);
    void weightDecayBoxValueChanged(double newValue);
    void trainingThreadIterationFinished(unsigned int imageId,double ***output);
    void trainingThreadFinishedWorking();

private:
    Ui::MainWindow *ui;
};

#endif // MAINWINDOW_H
