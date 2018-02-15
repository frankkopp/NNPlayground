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

package fko.nnplayground.nn;

import fko.nnplayground.API.ILayer;
import fko.nnplayground.API.INeuralNetwork;
import fko.nnplayground.UseCases.XorNN;
import org.apache.commons.io.IOUtils;
import org.apache.commons.math3.util.FastMath;
import org.junit.jupiter.api.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NeuralNetworkTest {

  private static final Logger LOG = LoggerFactory.getLogger(NeuralNetworkTest.class);

  // where to store the trained network
  private static final String folderPathPlain = "./var/";
  private static final String NN_SAVE_FILE =
          folderPathPlain + XorNN.class.getSimpleName() + "_TEST.zip";

  @Test
  void saveToFile() {

    INDArray matrixb1 = Nd4j.randn(1000, 1, 123);
    INDArray matrixb2 = Nd4j.randn(2, 1, 321);

    // layer (hidden layer)
    final Layer layer1 = new Layer(2, 1000,
            WeightInitializer.WeightInit.XAVIER, Activation.Activations.SIGMOID, 0.001d, 1234);
    layer1.setBiasMatrix(matrixb1);

    // z_output layer
    final OutputLayer layer2 = new OutputLayer(1000, 2,
            WeightInitializer.WeightInit.XAVIER, Activation.Activations.SIGMOID, 0.001d, 1234);
    layer2.setBiasMatrix(matrixb2);

    INeuralNetwork neuralNetwork = new NeuralNetwork(2, 2);
    neuralNetwork.addLayer(layer1, layer2);
    neuralNetwork.setLearningRate(1d);

    LOG.info("Writing model to file {}", NN_SAVE_FILE);
    neuralNetwork.saveToFile(NN_SAVE_FILE);

    LOG.info("Reading model from file {}", NN_SAVE_FILE);
    INeuralNetwork nn = NeuralNetwork.loadFromFile(NN_SAVE_FILE);

    LOG.info("Read NeuralNetwork: {}\n", nn);

    assertEquals(neuralNetwork.getInputLength(), nn.getInputLength());
    assertEquals(neuralNetwork.getOutputLength(), nn.getOutputLength());
    assertEquals(2, neuralNetwork.getLayerList().size());

    final ILayer read_layer1 = nn.getLayerList().get(0);
    assertEquals(read_layer1.getClass().getName(), layer1.getClass().getName());
    assertEquals(read_layer1.getInputSize(), layer1.getInputSize());
    assertEquals(read_layer1.getOutputSize(), layer1.getOutputSize());
    assertEquals(read_layer1.getSeed(), layer1.getSeed());
    assertEquals(read_layer1.getRegLamba(), layer1.getRegLamba());
    assertEquals(read_layer1.getActivationFunction(), layer1.getActivationFunction());
    assertEquals(read_layer1.getWeightInit(), layer1.getWeightInit());
    assertTrue(read_layer1.getWeightsMatrix().equalsWithEps(layer1.getWeightsMatrix(), 1e-6d));
    assertTrue(read_layer1.getBiasMatrix().equalsWithEps(layer1.getBiasMatrix(), 1e-6d));

    final ILayer read_layer2 = nn.getLayerList().get(1);
    assertEquals(read_layer2.getClass().getName(), layer2.getClass().getName());
    assertEquals(read_layer2.getInputSize(), layer2.getInputSize());
    assertEquals(read_layer2.getOutputSize(), layer2.getOutputSize());
    assertEquals(read_layer2.getSeed(), layer2.getSeed());
    assertEquals(read_layer2.getRegLamba(), layer2.getRegLamba());
    assertEquals(read_layer2.getActivationFunction(), layer2.getActivationFunction());
    assertEquals(read_layer2.getWeightInit(), layer2.getWeightInit());
    assertTrue(read_layer2.getWeightsMatrix().equalsWithEps(layer2.getWeightsMatrix(), 1e-6d));
    assertTrue(read_layer2.getBiasMatrix().equalsWithEps(layer2.getBiasMatrix(), 1e-6d));

  }

  @Test
  void loadFromFile() {
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
    input.putScalar(new int[]{0, 0}, 0);
    input.putScalar(new int[]{0, 1}, 0);
    labels.putScalar(new int[]{0, 0}, 1); // False
    labels.putScalar(new int[]{0, 1}, 0); // True

    // when first input=1 and second input=0
    input.putScalar(new int[]{1, 0}, 1);
    input.putScalar(new int[]{1, 1}, 0);
    labels.putScalar(new int[]{1, 0}, 0); // False
    labels.putScalar(new int[]{1, 1}, 1); // True

    // same as above
    input.putScalar(new int[]{2, 0}, 0);
    input.putScalar(new int[]{2, 1}, 1);
    labels.putScalar(new int[]{2, 0}, 0); // False
    labels.putScalar(new int[]{2, 1}, 1); // True

    // when both inputs fire, xor is false again
    input.putScalar(new int[]{3, 0}, 1);
    input.putScalar(new int[]{3, 1}, 1);
    labels.putScalar(new int[]{3, 0}, 1); // False
    labels.putScalar(new int[]{3, 1}, 0); // True

    // create dataset object
    DataSet dataSet = new DataSet(input, labels);

    INeuralNetwork neuralNetwork = null;

    if (true || Files.notExists(Paths.get(NN_SAVE_FILE))) {
      LOG.info("Safe file  {} for model does not exist. Training new model...", NN_SAVE_FILE);

      final int seed = 1234;

      neuralNetwork = new NeuralNetwork(2, 2);

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
      neuralNetwork.eval(dataSet);

      LOG.info("Writing model to file {}", NN_SAVE_FILE);
      neuralNetwork.saveToFile(NN_SAVE_FILE);
    }

    LOG.info("Reading model from file {}", NN_SAVE_FILE);
    INeuralNetwork newNN = NeuralNetwork.loadFromFile(NN_SAVE_FILE);
    newNN.eval(dataSet);

    LOG.info("Finished");
  }

  @Test
  public void NDArraySaveReadTest() throws IOException {

    final String folderPathPlain = "./var/";
    final String testFile = folderPathPlain + "TEST_NDArrayRestore.txt";

    INDArray matrix4x2 = Nd4j.randn(4, 2, 123);
    INDArray matrix4x1 = Nd4j.randn(4, 1, 321);

    // WRITE NDArrays to txt file

    BufferedOutputStream outputStream = new BufferedOutputStream(new FileOutputStream(testFile));
    DataOutputStream dos = new DataOutputStream(outputStream);

    dos.writeBytes("[matrix4x2]" + System.lineSeparator());
    Nd4j.writeTxtString(matrix4x2, dos, 15);
    dos.writeBytes(System.lineSeparator());

    dos.writeBytes("[matrix4x1]" + System.lineSeparator());
    // FIXME - Workaround for bug in library
    INDArray extraCol = Nd4j.zeros(matrix4x1.rows(),2).putiColumnVector(matrix4x1);
    Nd4j.writeTxtString(extraCol, dos, 15);
    dos.writeBytes(System.lineSeparator());

    dos.flush();
    outputStream.close();

    LOG.debug("Matrix16x2 ORIG:\n{}\n{}", matrix4x2.shapeInfoToString(), matrix4x2);
    LOG.debug("Matrix16x1 ORIG:\n{}\n{}", matrix4x1.shapeInfoToString(), matrix4x1);

    // READ NDArrays from txt file

    INDArray read_Matrix4x2 = null;
    INDArray read_Matrix4x1 = null;

    BufferedReader inputStream = new BufferedReader(new FileReader(testFile));
    String line = "";
    while ((line = inputStream.readLine()) != null) {
      if (line.equals("[matrix4x2]")) {

        // weightMatrix
        StringBuilder sb_weightMatrix = new StringBuilder();
        while ((line = inputStream.readLine()) != null) {
          if (line.equals("[matrix4x1]")) break;
          // restore the line separator as the readLine() strips it
          sb_weightMatrix.append(line).append(System.lineSeparator());
        }
        read_Matrix4x2 = Nd4j.readTxtString(IOUtils.toInputStream(sb_weightMatrix.toString().trim()));

        // biasMatrix
        StringBuilder sb_biasMatrix = new StringBuilder();
        while ((line = inputStream.readLine()) != null) {
          // restore the line separator as the readLine() strips it
          sb_biasMatrix.append(line).append(System.lineSeparator());
        }
        read_Matrix4x1 = Nd4j.readTxtString(IOUtils.toInputStream(sb_biasMatrix.toString().trim()));
      }
      // FIXME - Workaround for bug in library
      read_Matrix4x1 = read_Matrix4x1.getColumn(0);
      LOG.debug("Matrix16x2 READ:\n{}\n{}", read_Matrix4x2.shapeInfoToString(), read_Matrix4x2);
      LOG.debug("Matrix16x1 READ:\n{}\n{}", read_Matrix4x1.shapeInfoToString(), read_Matrix4x1);
    }
  }
}
