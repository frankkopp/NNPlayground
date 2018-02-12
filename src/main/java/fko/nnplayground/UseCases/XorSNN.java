package fko.nnplayground.UseCases;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

/**
 * This basic example shows how to manually create a DataSet and train it to an basic Network.
 *
 * <p>The network consists in 2 input-neurons, 1 hidden-layer with 4 hidden-neurons, and 2
 * output-neurons.
 *
 * <p>I choose 2 output neurons, (the first fires for false, the second fires for true) because the
 * Evaluation class needs one neuron per classification.
 *
 * @author Peter Großmann
 */
public class XorSNN {
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
    // then the first output fires for false, and the second is 0 (see class
    // comment)
    labels.putScalar(new int[] {0, 0}, 1);
    labels.putScalar(new int[] {0, 1}, 0);

    // when first input=1 and second input=0
    input.putScalar(new int[] {1, 0}, 1);
    input.putScalar(new int[] {1, 1}, 0);
    // then xor is true, therefore the second output neuron fires
    labels.putScalar(new int[] {1, 0}, 0);
    labels.putScalar(new int[] {1, 1}, 1);

    // same as above
    input.putScalar(new int[] {2, 0}, 0);
    input.putScalar(new int[] {2, 1}, 1);
    labels.putScalar(new int[] {2, 0}, 0);
    labels.putScalar(new int[] {2, 1}, 1);

    // when both inputs fire, xor is false again - the first output should
    // fire
    input.putScalar(new int[] {3, 0}, 1);
    input.putScalar(new int[] {3, 1}, 1);
    labels.putScalar(new int[] {3, 0}, 1);
    labels.putScalar(new int[] {3, 1}, 0);

    // create dataset object
    DataSet dataSet = new DataSet(input, labels);

    final int seed = 1234;

    SimpleNeuralNetwork neuralNetwork = new SimpleNeuralNetwork(2, 1, 1, 2, 32, seed);

    int nEpochs = 1;
    int iterations = 50000;
    neuralNetwork.setLearningRate(0.1d);

    neuralNetwork.train(dataSet, nEpochs, iterations);
  }
}
