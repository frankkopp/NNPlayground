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

import fko.nnplayground.API.ILayer;
import fko.nnplayground.API.IOutputLayer;
import fko.nnplayground.API.Network;
import fko.nnplayground.nn.*;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
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
    final ILayer layer_1 = new Layer(inputLength, sizeHiddenLayer, WeightInitializer.WeightInit.XAVIER, Activation.Activations.SIGMOID, seed);
    // layer 2 (output layer)
    final IOutputLayer outputLayer = new OutputLayer(sizeHiddenLayer, nLabels, WeightInitializer.WeightInit.XAVIER, Activation.Activations.SIGMOID, seed);

    // Iterations
    for (int iteration = 0; iteration < iterations; iteration++) {

      // forward pass
      final INDArray outputLastLayer = layer_1.forwardPass(features);
      outputLayer.forwardPass(outputLastLayer);

      // output loss
      if (totalIterations++ % 100 == 0) {
        LOG.info("Loss at iteration {} (batch size {}) = {}",
                totalIterations-1, features.columns(), outputLayer.computeTotalError(labels, true));
      }

      // back propagation
      final INDArray errorPreviousLayer = outputLayer.backwardPass();
      layer_1.backwardPass(errorPreviousLayer);

      // update parameters
      outputLayer.updateWeights(layer_1.getActivation(), learningRate);
      layer_1.updateWeights(features, learningRate);

    }
  }

  @Override
  public void eval(final DataSetIterator dataSetIterator) {
    // not implemented
  }

  @Override
  public void eval(final DataSet dataSet) {
    // not implemented
  }

  @Override
  public INDArray predict(final INDArray features) {
    // not implemented
    return null;
  }

  public double getLearningRate() {
    return learningRate;
  }

  public void setLearningRate(final double learningRate) {
    this.learningRate = learningRate;
  }


  @Override
  public void saveToFile(final String nnSaveFile) {
    // not implemented
  }

  @Override
  public void loadFromFile(final String nnSaveFile) {
    // not implemented
  }

  @Override
  public List<ILayer> getLayerList() {
    return null;
  }

  @Override
  public void addLayer(final Layer layer) {

  }
}
