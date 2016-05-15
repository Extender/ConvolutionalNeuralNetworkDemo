#ifndef TRAININGTHREAD_H
#define TRAININGTHREAD_H

#include <stdlib.h>
#include <math.h>
#include <ctime>

#include <QThread>
#include <QMutex>

#include "cnnlayer.h"
#include "mainwindow.h"

class MainWindow;

class TrainingThread : public QThread
{
    Q_OBJECT

public:
    bool stopRequested;
    MainWindow *window;
    QMutex *mutex;

    double learningRate;
    double momentum;
    double weightDecay;

    TrainingThread(MainWindow *_window,double _learningRate,double _momentum,double _weightDecay);
    ~TrainingThread();

    void run();

signals:
    void iterationFinished(unsigned int imageId,double ***output);
};

#endif // TRAININGTHREAD_H
