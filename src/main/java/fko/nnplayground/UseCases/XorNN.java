package fko.nnplayground.UseCases;

import fko.nnplayground.API.Network;
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

    // corresponding list with expected output values, 4 training samples
    // with data for 2 output-neurons each
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

    Network neuralNetwork = new NeuralNetwork(2, 1, 1, 2, seed);

    // layer (hidden layer)
    neuralNetwork.addLayer(
        new Layer(
            2, 32, WeightInitializer.WeightInit.XAVIER, Activation.Activations.SIGMOID, seed));
    // output layer
    neuralNetwork.addLayer(
        new OutputLayer(
            32,
            2,
            labels.transpose(),
            WeightInitializer.WeightInit.XAVIER,
            Activation.Activations.SIGMOID,
            seed));

    int nEpochs = 1;
    int iterations = 1000;
    neuralNetwork.setLearningRate(1d);

    neuralNetwork.train(dataSet, nEpochs, iterations);

//    LOG.info("Writing model to file {}", NN_SAVE_FILE);
//    neuralNetwork.saveToFile(NN_SAVE_FILE);
//    neuralNetwork.loadFromFile(NN_SAVE_FILE);

    neuralNetwork.eval(dataSet);
  }
}
