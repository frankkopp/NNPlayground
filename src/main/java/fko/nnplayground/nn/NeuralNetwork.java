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
import fko.nnplayground.API.IOutputLayer;
import fko.nnplayground.API.Network;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** Multi layered NeuralNetwork */
public class NeuralNetwork implements Network {

  private static final Logger LOG = LoggerFactory.getLogger(NeuralNetwork.class);

  // state to serialize
  private int inputLength;
  private int outputLength;
  private List<ILayer> layerList = new ArrayList<>();

  // can be regenerated after loading
  private int epochs;
  private int iterations;
  private double learningRate;

  private int totalIterations;
  private int truePositives;
  private int trueNegatives;
  private int falsePositives;
  private int falseNegatives;
  private int totalPositives;
  private int totalNegatives;

  /**
   * TODO: improve evaluation
   * TODO: save and load train data
   * TODO: add SOFTMAX
   * TODO: add listener
   *  @param height
   * @param width
   * @param channels
   * @param nLabels
   */
  public NeuralNetwork(
          final int height, final int width, final int channels, int nLabels) {
    this(height * width * channels, nLabels);
  }

  public NeuralNetwork(final int inputLength, final int nLabels) {
    this.inputLength = inputLength;
    this.outputLength = nLabels;
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
        final INDArray features =
            batch.getFeatures().reshape(batch.numExamples(), inputLength).transpose();
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
    final INDArray features =
        dataSet.getFeatures().reshape(dataSet.numExamples(), inputLength).transpose();
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
   *
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

    // last layer needs to be z_output layer
    ILayer tmp = layerList.get(layerList.size() - 1);
    IOutputLayer outputLayer = null;

    if (tmp instanceof IOutputLayer) {
      outputLayer = (IOutputLayer) tmp;
    } else {
      final IllegalStateException e = new IllegalStateException("Last layer not IOutputLayer");
      LOG.error("Last layer needs to be IOutputLayer", e);
      throw e;
    }

    final int nExamples = features.columns();

    // Iterations
    for (int iteration = 0; iteration < iterations; iteration++) {

      // BACKPROPAGATION ALGORITHM
      // http://neuralnetworksanddeeplearning.com/chap2.html#the_backpropagation_algorithm

      // forward pass through all layers
      INDArray activationLastLayer = features;
      for (ILayer layer : layerList) {
          activationLastLayer = layer.forwardPass(activationLastLayer);
      }

      // z_output loss
      if (totalIterations++ % 100 == 0) {
        LOG.info(
            "Loss at iteration {} (batch size {}) = {}",
            totalIterations - 1,
                nExamples,
            outputLayer.computeCost(labels, nExamples, true));
      }

      // back propagation through all layers
      // http://neuralnetworksanddeeplearning.com/chap2.html#eqtnBP1
      // BP1: δLj = (∂C/∂aLj) * σ′(zLj) [An equation for the error in the z_output layer, δL]
      // BP2: δl= ((wl+1)T * δl+1) ⊙ σ′(zl) 8An equation for the error δl in terms of the error in the next layer, δl+1]
      // BP3: ∂C / ∂blj = δlj [An equation for the rate of change of the cost with respect to any bias in the network]
      // BP4: ∂C / ∂wljk = al−1k ⊙ δlj [An equation for the rate of change of the cost with respect to any weight in the network]
      INDArray errorPreviousLayer = outputLayer.backwardPass(outputLayer.computeOutputError(labels, nExamples, true));
      for (int i = layerList.size() - 2; i >= 0; i--) {
        errorPreviousLayer = layerList.get(i).backwardPass(errorPreviousLayer);
      }

      // update parameters of all layers
      INDArray lastLayerActivation = features;
      for (ILayer layer : layerList) {
        layer.updateWeights(lastLayerActivation, nExamples, learningRate);
        lastLayerActivation = layer.getActivation();
      }
    }
  }

  @Override
  public void eval(final DataSetIterator dataSetIterator) {
    truePositives = 0;
    trueNegatives = 0;
    falsePositives = 0;
    falseNegatives = 0;
    totalPositives = 0;
    totalNegatives = 0;

    if (dataSetIterator.resetSupported()) {
      dataSetIterator.reset();
    }
    // Batch
    while (dataSetIterator.hasNext()) { // one batch
      // get the next batch of examples
      DataSet batch = dataSetIterator.next();
      // DataSet has shape "numEx,Channels,height,width" -> needs to become "numEx,inputlength"
      // also needs to be transposed so the numExp are columns and inputData is rows
      evalBatch(batch);
    }
    printEvaluation();
  }

  @Override
  public void eval(final DataSet dataSet) {
    truePositives = 0;
    trueNegatives = 0;
    falsePositives = 0;
    falseNegatives = 0;
    totalPositives = 0;
    totalNegatives = 0;

    evalBatch(dataSet);

    printEvaluation();
  }

  private void evalBatch(final DataSet dataSet) {
    List<INDArray> realOutputs = new ArrayList<>();
    List<INDArray> guesses = new ArrayList<>();

    // iterate over all examples
    // TODO: do this without loop for all examples
    for (int n = 0; n < dataSet.numExamples(); n++) {
      // DataSet has shape "numEx,Channels,height,width" -> needs to become "numEx,inputlength"
      // also needs to be transposed so the numExp are columns and inputData is rows
      final INDArray features = dataSet.get(n).getFeatures().reshape(1, inputLength).transpose();
      final INDArray labels = dataSet.get(n).getLabels().transpose();

      final INDArray prediction = predict(features);

      realOutputs.add(labels);
      guesses.add(prediction);
    }

    final int nLabels = dataSet.getLabels().transpose().rows();

    for (int i = 0; i < realOutputs.size(); i++) {
      // argmax - currently we only check if the highest prediction is correct
      final int actual = (int) Nd4j.argMax(realOutputs.get(i), 0).getDouble(0);
      final int predicted = (int) Nd4j.argMax(guesses.get(i), 0).getDouble(0);
      if (actual == predicted) {
        LOG.debug("Example: CORRECT Actual = {} Predicted = {} ",
                realOutputs.get(i).ravel(), guesses.get(i).ravel());
        truePositives++;
        trueNegatives += nLabels - 1;
      } else {
        LOG.debug("Example: INCORRECT Actual = {} Predicted = {} ",
                realOutputs.get(i).ravel(), guesses.get(i).ravel());
        falsePositives++;
        falseNegatives++;
        trueNegatives += nLabels - 2;
      }
    }

    totalPositives += dataSet.numExamples(); // number of examples as each examples has one positive
    totalNegatives += dataSet.numExamples()
            * (nLabels - 1); // number off examples time number of labels - 1 as one is positive
  }

  @Override
  public INDArray predict(final INDArray features) {
    // forward pass through all layers
    INDArray outputLastLayer = features;
    for (ILayer layer : layerList) {
      outputLastLayer = layer.forwardPass(outputLastLayer);
    }
    return outputLastLayer;
  }

  private void printEvaluation() {
    System.out.printf("True Positives  %d%nFalse Positives %d%nTrue Negatives  %d%nFalse Negatives %d%n",
            truePositives, falsePositives, trueNegatives, falseNegatives);
    System.out.printf("Total Positives: %,d%nTotal Negatives: %,d%n", totalPositives, totalNegatives);
    System.out.printf("Recall   : %.4f%n", (double) truePositives / totalPositives);
    System.out.printf("Precision: %.4f%n", (double) truePositives /(truePositives + falsePositives));
    System.out.printf("Accuracy : %.4f%n", (double) (truePositives + trueNegatives)/(totalPositives + totalNegatives));
    System.out.printf("F1Score  : %.4f%n", (double) (2* truePositives)/(2* truePositives + falsePositives + falseNegatives));
  }

  @Override
  public void saveToFile(final String nnSaveFile) {
    try (BufferedOutputStream stream = new BufferedOutputStream(new FileOutputStream(nnSaveFile))) {
      //      ZipOutputStream zipOutputStream = new ZipOutputStream(new
      // CloseShieldOutputStream(stream));
      //
      //      // Save layers as binary
      //      int c=0;
      //      for (ILayer layer : layerList) {
      //        ZipEntry zipLayer = new ZipEntry("layer_" + c++ + ".bin");
      //        zipOutputStream.putNextEntry(zipLayer);
      //        DataOutputStream dos = new DataOutputStream(new
      // BufferedOutputStream(zipOutputStream));
      //        try {
      //          layer.write(dos);
      //        } finally {
      //          dos.flush();
      //        }
      //      }
      //
      //      zipOutputStream.close();
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  @Override
  public void loadFromFile(final String nnSaveFile) {}

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

  @Override
  public void addLayer(final Layer... layer) {
    layerList.addAll(Arrays.asList(layer));
  }


  @Override
  public String toString() {
    return "NeuralNetwork{"
        + "inputLength="
        + inputLength
        + ", outputLength="
        + outputLength
        + ", epochs="
        + epochs
        + ", iterations="
        + iterations
        + ", learningRate="
        + learningRate
        + ", totalIterations="
        + totalIterations
        + ", layerList="
        + Arrays.toString(layerList.toArray())
        + '}';
  }
}
