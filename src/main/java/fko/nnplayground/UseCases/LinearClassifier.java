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

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** LinearClassifier */
public class LinearClassifier {

  private static final Logger LOG = LoggerFactory.getLogger(LinearClassifier.class);

  private static final long seed = 1234;

  // Setup
  private final int height;
  private final int width;
  private final int nChannels;
  private final int nLabels;
  private final int inputLength;

  private INDArray weightsMatrix; // W
  private INDArray biasMatrix; // b
  private INDArray gradientsWMatrix; // Gw
  private INDArray gradientsBMatrix; // Gb

  // configuration
  private double learningRate = 0.001d;
  private boolean useRegularization = true;
  private double regularizationLambda = 0.1d;
  private LossFunction lossFunction = LossFunction.SVM;

  public LinearClassifier(
      final int height, final int width, final int nChannels, final int nLabels) {
    this.height = height;
    this.width = width;
    this.nChannels = nChannels;
    this.nLabels = nLabels;
    this.inputLength = this.height * this.width * this.nChannels;

    // create and initialize weight matrix
    weightsMatrix = Nd4j.rand(nLabels, inputLength, seed);
    biasMatrix = Nd4j.rand(nLabels, 1, seed);
    gradientsWMatrix = Nd4j.zeros(nLabels, inputLength);
    gradientsBMatrix = Nd4j.zeros(nLabels, 1);
  }

  /**
   * Computes the score, loss and gradients
   *
   * <p>TODO: MiniBatch
   */
  public void train(final DataSet dsTrain, final int iterations) {
    int numExamples = dsTrain.numExamples();

    int iterCounter = 0;
    while (iterCounter++ < iterations) {
      LOG.info("Iteration: {} of {}", iterCounter, iterations);
      LOG.debug("Computing Score. Loss, Gradients");

      // collect all losses over all examples
      INDArray lossArray = Nd4j.zeros(numExamples);
      // iterate over all data examples
      for (int i = 0; i < numExamples; i++) {
        LOG.debug("Sample {} of {}", i + 1, numExamples);
        double loss_i = 0d;

        // one sample
        DataSet sample = dsTrain.get(i);
        int correctLabelIdx = sample.outcome();
        LOG.debug("Sample class: {}", dsTrain.getLabelName(correctLabelIdx));

        // compute scores
        INDArray x_i = Nd4j.toFlattened(sample.getFeatures()).transpose();
        INDArray scoreVector = score(x_i);
        int predictedLabelIdx = scoreVector.argMax(0).getInt(0);
        LOG.debug("Predicted class: {} (idx: {})",
            dsTrain.getLabelName(predictedLabelIdx),
            predictedLabelIdx);

        // compute loss and gradient
        if (lossFunction.equals(LossFunction.SVM)) {
          loss_i = lossSVM(x_i, scoreVector, correctLabelIdx);
        } else if (lossFunction.equals(LossFunction.SOFTMAX)) {
          INDArray pScoreVector = softmaxScore(scoreVector);
          loss_i = lossSoftmax(x_i, pScoreVector, correctLabelIdx);
        } else {
          LOG.error("undefined loss function {}", lossFunction);
          throw new RuntimeException("undefined loss function ");
        }

        lossArray.put(0, i, loss_i);
        LOG.debug("Loss Sample {} =  {}", i + 1, loss_i);
      }

      // normalize to average
      LOG.debug("Normalize");
      Number lossAvgAll = lossArray.meanNumber();
      LOG.debug("Overall Loss: {} ({})", lossAvgAll, lossFunction.name());
      // normalize gradients as well
      LOG.debug("Gradients raw {}", gradientsWMatrix.ravel());
      gradientsWMatrix.divi(numExamples);
      LOG.debug("Gradients nor {}", gradientsWMatrix.ravel());

      // regularization
      double regLoss = 0d;
      if (useRegularization) {
        regLoss = regularizationLoss();
      }
      double totalLoss = lossAvgAll.doubleValue() + regLoss;
      LOG.info("Total Loss: {} ({})", totalLoss, lossFunction.name());

      // perform SGD update
      performSGDUpdate();
    }
  }

  /**
   * Updates the current weights and bias matrix with the current gradient matrix
   * and current learning factor.
   */
  public void performSGDUpdate() {
    LOG.debug("SGD Update");
    LOG.debug("Weights {}", weightsMatrix.ravel());
    LOG.debug("Bias {}", biasMatrix.ravel());
    weightsMatrix.addi(gradientsWMatrix.mul(-learningRate));
    biasMatrix.addi(gradientsBMatrix.mul(-learningRate));
    LOG.debug("Weights {}", weightsMatrix.ravel());
    LOG.debug("Bias {}", biasMatrix.ravel());
  }

  /**
   * Computes the regularization loss based on the current weights matrix.<br>
   * This does change the gradients matrix
   * @return
   */
  public double regularizationLoss() {
    double regLoss = 0;
    LOG.debug("Regularization");
    // double rLoss =
    //        regularizationLambda * Transforms.pow(weightsMatrix, 2).sumNumber().doubleValue();
    for (int j = 0; j < weightsMatrix.rows(); j++) {
      for (int i = 0; i < weightsMatrix.columns(); i++) {
        final double wValue = weightsMatrix.getDouble(j, i);
        regLoss += regularizationLambda * wValue * wValue;
        gradientsWMatrix.put(
            j, i, gradientsWMatrix.getDouble(j, i) + 0.5 * regularizationLambda * wValue);
      }
    }
    LOG.debug("Reg loss {}", regLoss);
    LOG.debug("Reg grad {}", gradientsWMatrix.ravel());
    return regLoss;
  }

  /**
   * Multiclass SVM loss / hinge loss)
   *
   * @param x_i the current sample feature vector
   * @param scoreVector the score calculated for a given sample
   * @param correctLabelIndex the true label for this sample
   * @return
   */
  public double lossSVM(final INDArray x_i, INDArray scoreVector, int correctLabelIndex) {
    double score_correctLabel = scoreVector.getDouble(correctLabelIndex, 0);
    double loss_i = 0;

    // hinge lossSVM (sum of difference to correct label +1)
    // TODO: Could this be done without the loop - it is possible in numpy
    // TODO: numpy: np.maximum(0, scores - scores[y] +1)
    for (int j = 0; j < scoreVector.rows(); j++) {
      if (j == correctLabelIndex) continue; // correct class

      double score_j = scoreVector.getDouble(j, 0);
      double loss_ij = score_j - score_correctLabel + 1;

      if (loss_ij > 0) { // max(0, scoreDelta)
        loss_i += loss_ij; // alternative squared hinge
        // ∇wyiLi = −(∑j≠yi 1(wTjxi−wTyixi+Δ>0))xi
        // ∇wjLi  =         1(wTjxi−wTyixi+Δ>0) xi
        gradientsWMatrix.getRow(j).addi(x_i.transpose());
        gradientsBMatrix.getRow(j).addi(1);
        gradientsWMatrix.getRow(correctLabelIndex).subi(x_i.transpose());
        gradientsBMatrix.getRow(correctLabelIndex).subi(1);
      }
    }
    return loss_i;
  }

  /**
   * f(x,W) - calculates the score vector for input vector x and the current weights matrix
   * @param inputX - 1d array with flattened pixel data from the image
   * @return 1d array with score values for each label (y)
   */
  public INDArray score(INDArray inputX) {
    // f(x,W) = Wx +b;
    final INDArray y = weightsMatrix.mmul(inputX).add(biasMatrix);
    return y;
  }

  /**
   * Softmax loss L = -log(e^syi / sum(e^sj)
   *
   * @param x_i the current sample
   * @param softmaxScoreVector the softmax score calculated for a given image
   * @param labelIndex the true label for this image
   * @return
   */
  public double lossSoftmax(final INDArray x_i, INDArray softmaxScoreVector, int labelIndex) {
    // compute loss
    double tmpLoss = -Math.log(softmaxScoreVector.getDouble(labelIndex));
    // compute gradient dW[j, :] += (p-(j == y[i])) * X[:, i]
    for (int j = 0; j < softmaxScoreVector.rows(); j++) {
      if (j == labelIndex) {
        gradientsWMatrix.getRow(j).addi(x_i.mul(softmaxScoreVector.sub(1).getRow(j)).transpose());
      } else {
        gradientsWMatrix.getRow(j).addi(x_i.mul(softmaxScoreVector.getRow(j)).transpose());
      }
    }
    return tmpLoss;
  }

  /**
   * @param scoreVector - 1d vector with scores for labels
   * @return 1d array with score values for each label (y)
   */
  public INDArray softmaxScore(INDArray scoreVector) {
    // s = f(x,W) ==> P = e^sk / sum_j(e^sj);
    // shift the values of f so that the highest number is 0:
    // otherwise e^sk might become a NaN (too high)
    final INDArray s = scoreVector.sub(scoreVector.maxNumber());
    final INDArray exp = Transforms.exp(s);
    return exp.div(exp.sumNumber().doubleValue());
  }

  public void setWeightsMatrix(final INDArray weightsMatrix) {
    this.weightsMatrix = weightsMatrix;
  }

  public void setBiasMatrix(final INDArray biasMatrix) {
    this.biasMatrix = biasMatrix;
  }

  public INDArray getWeightsMatrix() {
    return weightsMatrix;
  }

  public INDArray getBiasMatrix() {
    return biasMatrix;
  }

  public INDArray getGradientsWMatrix() {
    return gradientsWMatrix;
  }

  public void setGradientsWMatrix(final INDArray gradientsWMatrix) {
    this.gradientsWMatrix = gradientsWMatrix;
  }

  public INDArray getGradientsBMatrix() {
    return gradientsBMatrix;
  }

  public void setGradientsBMatrix(final INDArray gradientsBMatrix) {
    this.gradientsBMatrix = gradientsBMatrix;
  }

  public boolean isUseRegularization() {
    return useRegularization;
  }

  public void setUseRegularization(final boolean useRegularization) {
    this.useRegularization = useRegularization;
  }

  public double getRegularizationLamda() {
    return regularizationLambda;
  }

  public void setRegularizationLamda(final double lamda) {
    this.regularizationLambda = lamda;
  }

  public double getLearningRate() {
    return learningRate;
  }

  public void setLearningRate(final double learningRate) {
    this.learningRate = learningRate;
  }

  public LossFunction getLossFunction() {
    return lossFunction;
  }

  public void setLossFunction(final LossFunction lossFunction) {
    this.lossFunction = lossFunction;
  }

  public enum LossFunction {
    SVM,
    SOFTMAX
  }
}
