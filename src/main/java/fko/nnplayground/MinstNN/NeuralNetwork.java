package fko.nnplayground.MinstNN;

import fko.nnplayground.API.ILayer;
import fko.nnplayground.API.IOutputLayer;
import fko.nnplayground.API.Network;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/** SimpleNeuralNetwork */
public class NeuralNetwork implements Network {

  private static final Logger LOG = LoggerFactory.getLogger(NeuralNetwork.class);

  private final int inputLength;
  private final int nLabels;
  private final int seed;

  private int epochs;
  private int iterations;
  private double learningRate;

  private int totalIterations;

  private List<ILayer> layerList = new ArrayList<>();

  /**
   * TODO: add Regularization
   * TODO: generalize for many layers
   * TODO: add other activations / SOFTMAX
   * TODO: add listener
   * @param height
   * @param width
   * @param channels
   * @param nLabels
   * @param seed
   */
  public NeuralNetwork(
      final int height,
      final int width,
      final int channels,
      int nLabels,
      int seed) {

    this(height * width * channels, nLabels, seed);
  }

  public NeuralNetwork(final int inputLength, final int nLabels, final int seed) {
    this.inputLength = inputLength;
    this.nLabels = nLabels;
    this.seed = seed;
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
    // we need at least one layer
    if (layerList.isEmpty()) {
      final IllegalStateException e = new IllegalStateException("No layers");
      LOG.error("Cannot train without layers.", e);
      throw e;
    }

    // last layer needs to be output layer
    ILayer tmp = layerList.get(layerList.size() - 1);
    IOutputLayer outputLayer = null;

    if (tmp instanceof IOutputLayer) {
      outputLayer = (IOutputLayer) tmp;
    } else {
      final IllegalStateException e = new IllegalStateException("Last layer not IOutputLayer");
      LOG.error("Last layer needs to be IOutputLayer", e);
      throw e;
    }

    // Iterations
    for (int iteration = 0; iteration < iterations; iteration++) {

      // forward pass through all layers
      INDArray outputLastLayer = features;
      for (ILayer layer : layerList) {
        if (layer == outputLayer) {
          outputLastLayer = outputLayer.forwardPass(outputLastLayer, true);
        } else {
          outputLastLayer = layer.forwardPass(outputLastLayer);
        }
      }

      // output loss
      if (totalIterations++ % 100 == 0) {
        LOG.info("Loss at iteration {} (batch size {}) = {}",
                totalIterations-1, features.columns(), outputLayer.getTotalError());
      }

      // back propagation through all layers
      INDArray errorPreviousLayer = outputLayer.backwardPass(outputLayer.computeError(true));
      for (int i=layerList.size()-2;i>=0;i--) {
        errorPreviousLayer = layerList.get(i).backwardPass(errorPreviousLayer);
      }

      // update parameters of all layers
      INDArray lastLayerActivation = features;
      for (ILayer layer : layerList) {
        layer.updateWeights(lastLayerActivation, learningRate);
        lastLayerActivation = layer.getActivation();
      }
    }
  }

  @Override
  public double getLearningRate() {
    return learningRate;
  }

  @Override
  public void setLearningRate(final double learningRate) {
    this.learningRate = learningRate;
  }

  @Override
  public List<ILayer> getLayerList() {
    return layerList;
  }

  @Override
  public void addLayer(final Layer layer) {
    layerList.add(layer);
  }

}
