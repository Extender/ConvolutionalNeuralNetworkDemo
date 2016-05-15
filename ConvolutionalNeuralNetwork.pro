#-------------------------------------------------
#
# Project created by QtCreator 2016-05-07T10:25:58
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = ConvolutionalNeuralNetwork
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    graphicssceneex.cpp \
    graphicsviewex.cpp \
    cnnlayer.cpp \
    trainingthread.cpp \
    ../_DefaultLibrary/io.cpp \
    ../_DefaultLibrary/text.cpp

HEADERS  += mainwindow.h \
    graphicssceneex.h \
    graphicsviewex.h \
    cnnlayer.h \
    trainingthread.h \
    ../_DefaultLibrary/io.h \
    ../_DefaultLibrary/text.h

FORMS    += mainwindow.ui
