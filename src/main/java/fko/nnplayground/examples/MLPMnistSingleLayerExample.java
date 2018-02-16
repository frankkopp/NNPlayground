/*
 * MIT License
 *
 * Copyright (c) 2018 Frank Kopp
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */

package fko.nnplayground.examples;

import fko.nnplayground.util.DataUtilities;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Random;

/**
 * A Simple Multi Layered Perceptron (MLP) applied to digit classification for the MNIST Dataset
 * (http://yann.lecun.com/exdb/mnist/).
 *
 * <p>This file builds one input layer and one hidden layer.
 *
 * <p>The input layer has input dimension of numRows*numColumns where these variables indicate the
 * number of vertical and horizontal pixels in the image. This layer uses a rectified linear unit
 * (relu) activation function. The weights for this layer are initialized by using Xavier
 * initialization
 * (https://prateekvjoshi.com/2016/03/29/understanding-xavier-initialization-in-deep-neural-networks/)
 * to avoid having a steep learning curve. This layer will have 1000 output signals to the hidden
 * layer.
 *
 * <p>The hidden layer has input dimensions of 1000. These are fed from the input layer. The weights
 * for this layer is also initialized using Xavier initialization. The activation function for this
 * layer is a softmax, which normalizes all the 10 outputs such that the normalized sums add up to
 * 1. The highest of these normalized values is picked as the predicted class.
 */
public class MLPMnistSingleLayerExample {

  private static Logger log = LoggerFactory.getLogger(MLPMnistSingleLayerExample.class);

  private static final String basePath = "./var/data" + "/mnist";
  private static final String dataUrl = "http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz";

  public static void main(String[] args) throws Exception {
    // number of rows and columns in the input pictures
    final int height = 28;
    final int width = 28;
    final int channels = 1; // single channel for grayscale images

    int outputNum = 10; // number of output classes
    int batchSize = 128; // batch size for each epoch
    int numEpochs = 15; // number of epochs to perform

    int seed = 1234;
    Random randNumGen = new Random(seed);

    //Get the DataSetIterators:
    DataSetIterator trainIter = new MnistDataSetIterator(batchSize, true, seed);
    DataSetIterator testIter = new MnistDataSetIterator(batchSize, false, seed);

//    log.info("Data load and vectorization...");
//    String localFilePath = basePath + "/mnist_png.tar.gz";
//    if (DataUtilities.downloadFile(dataUrl, localFilePath))
//      log.debug("Data downloaded from {}", dataUrl);
//    if (!new File(basePath + "/mnist_png").exists())
//      DataUtilities.extractTarGz(localFilePath, basePath);
//
//    // vectorization of train data
//    File trainData = new File(basePath + "/mnist_png/training");
//    FileSplit trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
//    ParentPathLabelGenerator labelMaker =
//        new ParentPathLabelGenerator(); // parent path as the image label
//    ImageRecordReader trainRR = new ImageRecordReader(height, width, channels, labelMaker);
//    trainRR.initialize(trainSplit);
//    DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRR, batchSize, 1, outputNum);
//
//    // pixel values from 0-255 to 0-1 (min-max scaling)
//    DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
//    scaler.fit(trainIter);
//    trainIter.setPreProcessor(scaler);
//
//    // vectorization of test data
//    File testData = new File(basePath + "/mnist_png/testing");
//    FileSplit testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
//    ImageRecordReader testRR = new ImageRecordReader(height, width, channels, labelMaker);
//    testRR.initialize(testSplit);
//    DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, batchSize, 1, outputNum);
//    testIter.setPreProcessor(scaler); // same normalization for better results

    log.info("Build model....");
    MultiLayerConfiguration conf =
        new NeuralNetConfiguration.Builder()
            .seed(seed) // include a random seed for reproducibility
            // use stochastic gradient descent as an optimization algorithm
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .iterations(1)
            .learningRate(0.006) // specify the learning rate
            .updater(Updater.NESTEROVS)
            .regularization(true)
            .l2(1e-4)
            .list()
            .layer(
                0,
                new DenseLayer.Builder() // create the first, input layer with xavier initialization
                    .nIn(height * width)
                    .nOut(100)
                    .activation(Activation.RELU)
                    .weightInit(WeightInit.XAVIER)
                    .build())
            .layer(
                1,
                new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD) // create hidden layer
                    .nIn(100)
                    .nOut(outputNum)
                    .activation(Activation.SOFTMAX)
                    .weightInit(WeightInit.XAVIER)
                    .build())
            .pretrain(false)
            .backprop(true) // use backpropagation to adjust weights
            .build();

    MultiLayerNetwork model = new MultiLayerNetwork(conf);
    model.init();
    // print the score with every 1 iteration
    model.setListeners(new ScoreIterationListener(1));

    log.info("Train model....");
    for (int i = 0; i < numEpochs; i++) {
      model.fit(trainIter);
    }

    log.info("Evaluate model....");
    Evaluation eval =
        new Evaluation(outputNum); // create an evaluation object with 10 possible classes
    while (testIter.hasNext()) {
      DataSet next = testIter.next();
      INDArray output = model.output(next.getFeatureMatrix()); // get the networks prediction
      eval.eval(next.getLabels(), output); // check the prediction against the true class
    }

    log.info(eval.stats());
    log.info("****************Example finished********************");
  }
}
