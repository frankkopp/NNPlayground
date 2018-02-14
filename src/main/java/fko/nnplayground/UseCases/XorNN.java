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
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * XOR Classifier
 */
public class XorNN {

  private static final Logger LOG = LoggerFactory.getLogger(XorNN.class);

  // where to store the trained network
  private static final String folderPathPlain = "./var/";
  private static final String NN_SAVE_FILE =
      folderPathPlain + XorNN.class.getName() + "_" + System.currentTimeMillis() + ".zip";

  public static void main(String[] args) {

    // determines what ND4j uses internally as precision for floating point numbers
    Nd4j.setDataType(DataBuffer.Type.DOUBLE);

    // list off input values, 4 training samples with data for 2
    // input-neurons each
    INDArray input = Nd4j.zeros(4, 2);

    // corresponding list with expected z_output values, 4 training samples
    // with data for 2 z_output-neurons each
    INDArray labels = Nd4j.zeros(4, 2);

    // create first dataset
    // when first input=0 and second input=0
    input.putScalar(new int[] {0, 0}, 0);
    input.putScalar(new int[] {0, 1}, 0);
    labels.putScalar(new int[] {0, 0}, 1); // False
    labels.putScalar(new int[] {0, 1}, 0); // True

    // when first input=1 and second input=0
    input.putScalar(new int[] {1, 0}, 1);
    input.putScalar(new int[] {1, 1}, 0);
    labels.putScalar(new int[] {1, 0}, 0); // False
    labels.putScalar(new int[] {1, 1}, 1); // True

    // same as above
    input.putScalar(new int[] {2, 0}, 0);
    input.putScalar(new int[] {2, 1}, 1);
    labels.putScalar(new int[] {2, 0}, 0); // False
    labels.putScalar(new int[] {2, 1}, 1); // True

    // when both inputs fire, xor is false again
    input.putScalar(new int[] {3, 0}, 1);
    input.putScalar(new int[] {3, 1}, 1);
    labels.putScalar(new int[] {3, 0}, 1); // False
    labels.putScalar(new int[] {3, 1}, 0); // True

    // create dataset object
    DataSet dataSet = new DataSet(input, labels);

    final int seed = 1234;

    INeuralNetwork neuralNetwork = new NeuralNetwork(2, 2);

    // layer (hidden layer)
    final Layer layer1 = new Layer(2, 16,
            WeightInitializer.WeightInit.XAVIER, Activation.Activations.SIGMOID, 0.001d, seed);

    // z_output layer
    final OutputLayer layer2 = new OutputLayer(16, 2,
            WeightInitializer.WeightInit.XAVIER, Activation.Activations.SIGMOID, 0.001d, seed);

    neuralNetwork.addLayer(layer1, layer2);

    int nEpochs = 1;
    int iterations = 1100;
    neuralNetwork.setLearningRate(1d);

    neuralNetwork.train(dataSet, nEpochs, iterations);

    LOG.info("Writing model to file {}", NN_SAVE_FILE);
    neuralNetwork.saveToFile(NN_SAVE_FILE);

    neuralNetwork.eval(dataSet);

  }
}
