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

package fko.nnplayground.UseCases;

import fko.nnplayground.API.INeuralNetwork;
import fko.nnplayground.nn.*;
import fko.nnplayground.ui.TrainingUI;
import fko.nnplayground.util.DataUtilities;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Random;

/**
 * MinstNN
 */
public class MinstNN {

  private static final Logger LOG = LoggerFactory.getLogger(MinstNN.class);

  private static final String basePath = "./var/data" + "/mnist";
  private static final String dataUrl =
          "http://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz";

  // where to store the trained network
  private static final String folderPathPlain = "./var/";
  private static final String NN_SAVE_FILE =
          folderPathPlain + MinstNN.class.getName() + "_" + System.currentTimeMillis() + ".zip";

  public static void main(String[] args) throws Exception {

    // determines what ND4j uses internally as precision for floating point numbers
    Nd4j.setDataType(DataBuffer.Type.DOUBLE);

    int height = 28;
    int width = 28;
    int channels = 1; // single channel for grayscale images
    int outputNum = 10; // 10 digits classification
    int batchSize = 5;

    int seed = 1234;
    Random randNumGen = new Random(seed);

    LOG.info("Data load and vectorization...");
    String localFilePath = basePath + "/mnist_png.tar.gz";

    if (DataUtilities.downloadFile(dataUrl, localFilePath)) {
      LOG.debug("Data downloaded from {}", dataUrl);
    }
    if (!new File(basePath + "/mnist_png").exists()) {
      DataUtilities.extractTarGz(localFilePath, basePath);
    }

    // vectorization of train data
    File trainData = new File(basePath + "/mnist_png/training");
    LOG.debug("Preparing training data...{}", trainData);
    FileSplit trainSplit = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
    ParentPathLabelGenerator labelMaker =
            new ParentPathLabelGenerator(); // parent path as the image label
    ImageRecordReader trainRR = new ImageRecordReader(height, width, channels, labelMaker);
    trainRR.initialize(trainSplit);
    DataSetIterator trainIter = new RecordReaderDataSetIterator(trainRR, batchSize, 1, outputNum);

    // pixel values from 0-255 to 0-1 (min-max scaling)
    DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
    scaler.fit(trainIter);
    trainIter.setPreProcessor(scaler);

    // vectorization of test data
    File testData = new File(basePath + "/mnist_png/testing");
    LOG.debug("Preparing test data...{}", testData);
    FileSplit testSplit = new FileSplit(testData, NativeImageLoader.ALLOWED_FORMATS, randNumGen);
    ImageRecordReader testRR = new ImageRecordReader(height, width, channels, labelMaker);
    testRR.initialize(testSplit);
    DataSetIterator testIter = new RecordReaderDataSetIterator(testRR, batchSize, 1, outputNum);
    testIter.setPreProcessor(scaler); // same normalization for better results

    INeuralNetwork neuralNetwork = new NeuralNetwork(height, width, channels, outputNum);

    neuralNetwork.addListener(new TrainingUI(neuralNetwork, 100));

    // layer (hidden layer)
    neuralNetwork.addLayer(
            new Layer(height * width * channels, 1000,
                      WeightInitializer.WeightInit.XAVIER, Activation.Activations.SIGMOID, seed));
    // z_output layer
    neuralNetwork.addLayer(
            new OutputLayer(1000, outputNum,
                            WeightInitializer.WeightInit.XAVIER, Activation.Activations.SIGMOID, seed));

    int nEpochs = 2;
    int iterations = 5;
    neuralNetwork.setLearningRate(0.01d);

    neuralNetwork.train(trainIter, nEpochs, iterations);

    LOG.info("Writing model to file {}", NN_SAVE_FILE);
    neuralNetwork.saveToFile(NN_SAVE_FILE);

    neuralNetwork.eval(testIter);
  }
}
