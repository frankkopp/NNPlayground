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
import org.apache.commons.io.IOUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;

public class LayerReaderWriter {
  private static final Logger LOG = LoggerFactory.getLogger(LayerReaderWriter.class);

  private final ILayer layer;

  public LayerReaderWriter(ILayer layer) {
    this.layer = layer;
  }

  public void write(final DataOutputStream dos) throws IOException {
    // type
    dos.writeBytes(String.format("class=%s%n", layer.getClass().getName()));
    // input length
    dos.writeBytes(String.format("inputSize=%d%n", layer.getInputSize()));
    // output length
    dos.writeBytes(String.format("outputSize=%d%n", layer.getOutputSize()));
    // seed
    dos.writeBytes(String.format("seed=%d%n", layer.getSeed()));
    // regLambda
    dos.writeBytes(String.format("regLambda=%s%n", Double.toString(layer.getL2Strength())));
    // activation
    dos.writeBytes(String.format("ActivationFunction=%s%n", layer.getActivationFunction().name()));
    // weight init
    dos.writeBytes(String.format("WeightInit=%s%n", layer.getWeightInit().name()));
    // weightsMatrix
    dos.writeBytes("[WeightsMatrix]" + System.lineSeparator());
    Nd4j.writeTxtString(layer.getWeightsMatrix(), dos, 15); // double has a precision of 15
//    dos.writeBytes(System.lineSeparator());
    // biasMatrix
    dos.writeBytes("[BiasMatrix]" + System.lineSeparator()); // double has a precision of 15
    // FIXME - Workaround for bug in library
    INDArray extraCol = Nd4j.zeros(layer.getBiasMatrix().rows(),2).putiColumnVector(layer.getBiasMatrix());
    Nd4j.writeTxtString(extraCol, dos, 15);
    dos.writeBytes(System.lineSeparator());
  }

  public static ILayer read(final DataInputStream stream) throws IOException {

    String read_class = "";
    int read_inputSize = 1;
    int read_outputSize = 1;
    int read_seed = 1;
    double read_regLambda = 0.001d;
    Activation.Activations read_activation = null;
    WeightInitializer.WeightInit read_weightInit = null;
    INDArray read_weightMatrix = null;
    INDArray read_biasMatrix = null;

    // read configuration
    BufferedReader reader = new BufferedReader(new InputStreamReader(stream));
    String line = "";
    while ((line = reader.readLine()) != null) {

      if (line.matches(".*=.*")) {
        String[] keyValuePair = line.split("=");
        switch (keyValuePair[0]) {
          case "class":
            read_class = keyValuePair[1];
            break;
          case "inputSize":
            read_inputSize = Integer.valueOf(keyValuePair[1]);
            break;
          case "outputSize":
            read_outputSize = Integer.valueOf(keyValuePair[1]);
            break;
          case "seed":
            read_seed = Integer.valueOf(keyValuePair[1]);
            break;
          case "regLambda":
            read_regLambda = Double.valueOf(keyValuePair[1]);
            break;
          case "ActivationFunction":
            read_activation = Activation.Activations.valueOf(keyValuePair[1]);
            break;
          case "WeightInit":
            read_weightInit = WeightInitializer.WeightInit.valueOf(keyValuePair[1]);
            break;
        }

      } else if (line.equals("[WeightsMatrix]")) {

        // weightMatrix
        StringBuilder sb_weightMatrix = new StringBuilder();
        while ((line = reader.readLine()) != null) {
          if (line.equals("[BiasMatrix]")) break;
          // restore the line separator as the readLine() strips it
          sb_weightMatrix.append(line).append(System.lineSeparator());
        }
        read_weightMatrix = Nd4j.readTxtString(IOUtils.toInputStream(sb_weightMatrix.toString().trim()));

        // biasMatrix
        StringBuilder sb_biasMatrix = new StringBuilder();
        while ((line = reader.readLine()) != null) {
          // restore the line separator as the readLine() strips it
          sb_biasMatrix.append(line).append(System.lineSeparator());
        }
        read_biasMatrix = Nd4j.readTxtString(IOUtils.toInputStream(sb_biasMatrix.toString().trim()));
        // FIXME - Workaround for bug in library
        read_biasMatrix = read_biasMatrix.getColumn(0);
      }
    }

    // create layer
    ILayer newLayer = null;
    switch (read_class) {
      case "fko.nnplayground.nn.Layer":
        newLayer = new Layer(read_inputSize, read_outputSize, read_weightInit, read_activation, read_regLambda, read_seed);
        break;
      case "fko.nnplayground.nn.OutputLayer":
        newLayer = new OutputLayer(read_inputSize, read_outputSize, read_weightInit, read_activation, read_regLambda, read_seed);
        break;
      default:
        RuntimeException e = new RuntimeException("Unknown Layer type: " + read_class);
        LOG.error("", e);
        throw e;
    }

    // update weightMatrix and biasMatrix
    newLayer.setWeightsMatrix(read_weightMatrix);
    newLayer.setBiasMatrix(read_biasMatrix);

    return newLayer;
  }
}