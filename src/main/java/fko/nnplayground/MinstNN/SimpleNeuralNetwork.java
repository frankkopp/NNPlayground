package fko.nnplayground.MinstNN;

import fko.nnplayground.API.ILayer;
import fko.nnplayground.API.IOutputLayer;
import fko.nnplayground.API.Network;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

import static org.nd4j.linalg.ops.transforms.Transforms.exp;

/** SimpleNeuralNetwork */
public class SimpleNeuralNetwork implements Network {

  private static final Logger LOG = LoggerFactory.getLogger(SimpleNeuralNetwork.class);

  private final int inputLength;
  private final int nLabels;
  private final int seed;

  private int sizeHiddenLayer;
  private int epochs;
  private int iterations;
  private double learningRate;

  private int totalIterations;

  /**
   * TODO: add Regularization
   * TODO: generalize for many layers
   * TODO: add other activations / SOFTMAX
   * TODO: add listener
   * @param height
   * @param width
   * @param channels
   * @param nLabels
   * @param sizeHiddenLayer
   * @param seed
   */
  public SimpleNeuralNetwork(
      final int height,
      final int width,
      final int channels,
      int nLabels,
      int sizeHiddenLayer,
      int seed) {

    this(height * width * channels, nLabels, sizeHiddenLayer, seed);
  }

  public SimpleNeuralNetwork(
      final int inputLength, final int nLabels, final int sizeHiddenLayer, final int seed) {

    this.inputLength = inputLength;
    this.nLabels = nLabels;
    this.sizeHiddenLayer = sizeHiddenLayer;
    this.seed = seed;

    // defaults
    epochs = 1;
    iterations = 1;
    learningRate = 0.1d;
  }

  /**
   * Train the network with a given DataSet Iterator
   *
   * @param dataSetIter
   * @param epochs
   * @param iterations
   */
  public void train(DataSetIterator dataSetIter, int epochs, int iterations) {
    this.epochs = epochs;
    this.iterations = iterations;
    totalIterations = 0;

    // Epoch
    for (int epoch = 0; epoch < this.epochs; epoch++) {
      LOG.info("Train epoch {} of {}:", epoch + 1, epochs);
      if (dataSetIter.resetSupported()) {
        dataSetIter.reset();
      }
      // Batch
      while (dataSetIter.hasNext()) { // one batch
        // get the next batch of examples
        DataSet batch = dataSetIter.next();
        // DataSet has shape "numEx,Channels,height,width" -> needs to become "numEx,inputlength"
        // also needs to be transposed so the numExp are columns and inputData is rows
        final INDArray features = batch.getFeatures()
                .reshape(batch.numExamples(), inputLength).transpose();
        final INDArray labels = batch.getLabels().transpose();
        optimize(features, labels);
      }
    }
  }

  /**
   * Train the network with a given DataSet
   *
   * @param dataSet
   * @param epochs
   * @param iterations
   */
  @Override
  public void train(DataSet dataSet, int epochs, int iterations) {
    this.epochs = epochs;
    this.iterations = iterations;
    totalIterations = 0;
    // DataSet has shape "numEx,Channels,height,width" -> needs to become "numEx,inputlength"
    // also needs to be transposed so the numExp are columns and inputData is rows
    final INDArray features = dataSet.getFeatures()
            .reshape(dataSet.numExamples(), inputLength).transpose();
    final INDArray labels = dataSet.getLabels().transpose();

    // Epoch
    for (int epoch = 0; epoch < epochs; epoch++) {
      LOG.info("Train epoch {} of {}:", epoch + 1, epochs);
      optimize(features, labels);
    }
  }

  /**
   * Train the network with INDArray for features and one for labels.
   *
   * @param features (rows = number of inputs, columns = number of examples)
   * @param labels (rows = number of outputs, columns = number of examples)
   * @param epochs
   * @param iterations
   */
  @Override
  public void train(INDArray features, INDArray labels, int epochs, int iterations) {
    this.epochs = epochs;
    this.iterations = iterations;
    totalIterations = 0;

    // Epoch
    for (int epoch = 0; epoch < epochs; epoch++) {
      LOG.info("Train epoch {} of {}:", epoch + 1, epochs);
      optimize(features, labels);
    }
  }

  /**
   * Runs the optimization loop - forward pass, loss, back propagation, update params
   * @param features
   * @param labels
   */
  private void optimize(final INDArray features, final INDArray labels) {

    // layer 1 (hidden layer)
    final ILayer layer_1 = new Layer(inputLength, sizeHiddenLayer, WeightInitializer.WeightInit.XAVIER, Activations.SIGMOID, seed);
    // layer 2 (output layer)
    final IOutputLayer outputLayer = new OutputLayer(sizeHiddenLayer, nLabels, labels, WeightInitializer.WeightInit.XAVIER, Activations.SIGMOID, seed);

    // Iterations
    for (int iteration = 0; iteration < iterations; iteration++) {

      // forward pass
      final INDArray outputLastLayer = layer_1.forwardPass(features);
      outputLayer.forwardPass(outputLastLayer, true);

      // output loss
      if (totalIterations++ % 100 == 0) {
        LOG.info("Loss at iteration {} (batch size {}) = {}",
                totalIterations-1, features.columns(), outputLayer.getTotalError());
      }

      // back propagation
      final INDArray errorPreviousLayer = outputLayer.backwardPass();
      layer_1.backwardPass(errorPreviousLayer);

      // update parameters
      outputLayer.updateWeights(layer_1.getActivation(), learningRate);
      layer_1.updateWeights(features, learningRate);

    }
  }

  public double getLearningRate() {
    return learningRate;
  }

  public void setLearningRate(final double learningRate) {
    this.learningRate = learningRate;
  }


  @Override
  public List<ILayer> getLayerList() {
    return null;
  }

  @Override
  public void addLayer(final Layer layer) {

  }
}
