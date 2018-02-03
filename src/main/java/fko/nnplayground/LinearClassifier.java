package fko.nnplayground;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * LinearClassifier
 */
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
  private boolean useRegularization = true;
  private double lambda = 0.1;
  private LossFunction lossFunction = LossFunction.SVM;

  public LinearClassifier(final int height, final int width, final int nChannels, final int nLabels) {
    this.height = height;
    this.width = width;
    this.nChannels = nChannels;
    this.nLabels = nLabels;
    this.inputLength = this.height * this.width * nChannels;

    // create and initialize weight matrix
    weightsMatrix = Nd4j.rand(nLabels, inputLength, seed);
    biasMatrix = Nd4j.rand(nLabels, 1, seed);
    gradientsWMatrix = Nd4j.rand(nLabels, inputLength, seed);
    gradientsBMatrix = Nd4j.rand(nLabels, inputLength, seed);
  }

  /**
   * Computes the score, loss and gradients
   */
  public void train(DataSet dsTrain, int iterations) {
    int numExamples = dsTrain.numExamples();
    double totalLoss = 0d;

    while (iterations-- > 0) {

      // collect all losses over all examples
      INDArray lossArray = Nd4j.zeros(numExamples);
      // iterate over all data examples
      for (int i = 0; i < numExamples; i++) {
        double loss_i = 0d;

        // one sample
        DataSet sample = dsTrain.get(i);
        int correctLabelIdx = sample.outcome();
        LOG.debug("Sample class: {}", dsTrain.getLabelName(correctLabelIdx));

        // compute scores
        INDArray x_i = Nd4j.toFlattened(sample.getFeatures()).transpose();
        INDArray scoreVector = score(x_i);
        int predictedLabelIdx = scoreVector.argMax(0).getInt(0);
        LOG.debug(
            "Predicted class: {} (idx: {})",
            dsTrain.getLabelName(predictedLabelIdx), predictedLabelIdx);

        // compute loss and gradient
        if (lossFunction.equals(LossFunction.SVM)) {
          loss_i = lossSVM(x_i, scoreVector, correctLabelIdx);
        } else if (lossFunction.equals(LossFunction.SOFTMAX)) {
          INDArray pScoreVector = softmaxScore(scoreVector);
          loss_i = softmaxLoss(x_i, pScoreVector, correctLabelIdx);
        } else {
          LOG.error("undefined loss function {}", lossFunction);
          throw new RuntimeException("undefined loss function ");
        }

        lossArray.put(0, i, loss_i);
        LOG.debug("Loss Sample {} =  {}", i, loss_i);
      }

      // normalize to average
      Number lossAvgAll = lossArray.meanNumber();
      LOG.debug("Overall Loss: {} ({})", lossAvgAll, lossFunction.name());
      // normalize gradients as well
      // TODO
    }

    // perform SGD update
    // TODO
  }


  /**
   * Multiclass SVM loss / hinge loss)
   * TODO: Could this be done without the loop - it is possible in numpy
   * TODO: numpy: np.maximum(0, scores - scores[y] +1)
   * @param x_i the current sample feature vector
   * @param scoreVector the score calculated for a given sample
   * @param labelIndex the true label for this sample
   * @return
   */
  public double lossSVM(final INDArray x_i, INDArray scoreVector, int labelIndex) {
    double syi = scoreVector.getDouble(labelIndex, 0);
    double loss_i = 0;

    // hinge lossSVM (sum of difference to correct label +1)
    for (int j=0; j<scoreVector.rows(); j++) {
      if (j==labelIndex) continue;
      double sj = scoreVector.getDouble(j, 0);
      double loss_ij = sj - syi + 1;
      if (loss_ij > 0 ) { // max(0, scoreDelta)
        loss_i += loss_ij; // alternative squared hinge
        // TODO: gradient calculation
      }
    }

    if (useRegularization) {
      loss_i += regularization();
    }

    return loss_i;
  }

  /**
   * Softmax loss
   * L = -log(e^syi / sum(e^sj)
   *
   * @param x_i
   * @param scoreVector the softmax score calculated for a given image
   * @param labelIndex the true label for this image
   * @return
   */
  public double softmaxLoss(final INDArray x_i, INDArray scoreVector, int labelIndex) {
    double scoreLabel = scoreVector.getDouble(labelIndex);
    double tmpLoss = -Math.log10(scoreLabel);
    // TODO: gradient calculation

    if (useRegularization) {
      tmpLoss += regularization();
    }
    return tmpLoss;
  }

  /**
   *
   * @param inputX - 1d array with flattened pixel data from the image
   * @return 1d array with score values for each label (y)
   */
  public INDArray score(INDArray inputX) {
    // f(x,W) = Wx +b;
    INDArray y = weightsMatrix.mmul(inputX).add(biasMatrix);
    return y;
  }

  /**
   * @param scoreVector - 1d vector with scores for labels
   * @return 1d array with score values for each label (y)
   */
  public INDArray softmaxScore(INDArray scoreVector) {
    // s = f(x,W) ==> P = e^sk / sum_j(e^sj);
    // shift the values of f so that the highest number is 0:
    // otherwise e^sk might become a NaN (too high)
    INDArray s = scoreVector.sub(scoreVector.maxNumber());
    INDArray exp = Transforms.exp(s);
    INDArray p = exp.div(exp.sumNumber().doubleValue());
    return p;
  }

  /**
   * @return the regularization loss for the current weight matrix
   */
  public double regularization() {
    // L2: lambda * R(W) ==> R(W) = sum_k sum_l W^2k,l
    // L1: lambda * R(W) ==> R(W) = sum_k sum_l abs(Wk,l)
    double rLoss = lambda * Transforms.pow(weightsMatrix, 2).sumNumber().doubleValue();
    // double check
    //    double rloss2 = 0d;
    //    System.out.println(weightsMatrix);
    //    for (int i=0; i<weightsMatrix.rows(); i++) {
    //      for (int j=0; j<weightsMatrix.columns(); j++){
    //        System.out.printf("[%.2f] ", weightsMatrix.getDouble(i,j));
    //        rloss2 += lambda * weightsMatrix.getDouble(i,j) * weightsMatrix.getDouble(i,j);
    //      }
    //      System.out.println();
    //    }
    //    LOG.debug("Regularization Loss2 = {}", rloss2);
    return rLoss;
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

  public boolean isUseRegularization() {
    return useRegularization;
  }

  public void setUseRegularization(final boolean useRegularization) {
    this.useRegularization = useRegularization;
  }

  public double getLamda() {
    return lambda;
  }

  public void setLamda(final double lamda) {
    this.lambda = lamda;
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

