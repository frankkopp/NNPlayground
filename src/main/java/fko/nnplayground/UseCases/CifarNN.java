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
import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * CifarNN
 */
public class CifarNN {

  private static final Logger LOG = LoggerFactory.getLogger(CifarNN.class);

  private static final String basePath = "./var/data" + "/cifar10";

  // where to store the trained network
  private static final String folderPathPlain = "./var/";
  private static final String NN_SAVE_FILE =
          folderPathPlain + CifarNN.class.getName() + "_" + System.currentTimeMillis() + ".zip";

  private static int height = 32;
  private static int width = 32;
  private static int channels = 3;
  private static int numSamples = 50000;
  private static int batchSize = 64;
  private static int outputNum = 10;
  private static int seed = 123;

  private static boolean preProcessCifar = false; // use Zagoruyko's preprocess for Cifar

  public static void main(String[] args) throws Exception {

    // determines what ND4j uses internally as precision for floating point numbers
    Nd4j.setDataType(DataBuffer.Type.FLOAT);

    CifarDataSetIterator trainIter =
            new CifarDataSetIterator(
                    batchSize, numSamples, new int[]{height, width, channels}, preProcessCifar, true);

    CifarDataSetIterator testIter =
            new CifarDataSetIterator(
                    batchSize, numSamples, new int[]{height, width, channels}, preProcessCifar, false);

    // scaling the dataset to 0..1.0
    DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
    scaler.fit(trainIter);
    trainIter.setPreProcessor(scaler);
    scaler.fit(testIter);
    testIter.setPreProcessor(scaler);

    INeuralNetwork neuralNetwork = new NeuralNetwork(height, width, channels, outputNum);

    neuralNetwork.addListener(new TrainingUI(neuralNetwork, 25));

    // layer (hidden layer)
    final Layer layer1 = new Layer(height * width * channels, 1000,
                                   WeightInitializer.WeightInit.XAVIER, Activation.Activations.RELU, seed);

    // output layer
    final OutputLayer outputLayer = new OutputLayer(1000, outputNum,
                                                    WeightInitializer.WeightInit.XAVIER, Activation.Activations.SOFTMAX,
                                                    seed);

    layer1.setL2Strength(.005d);
    outputLayer.setL2Strength(.005d);

    neuralNetwork.addLayer(layer1, outputLayer);

    int nEpochs = 5;
    int iterations = 10;
    neuralNetwork.setLearningRate(.1d);

    neuralNetwork.train(trainIter, nEpochs, iterations);

    LOG.info("Writing model to file {}", NN_SAVE_FILE);
    neuralNetwork.saveToFile(NN_SAVE_FILE);

    neuralNetwork.eval(testIter);

  }
}
