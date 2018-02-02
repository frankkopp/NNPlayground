package fko.nnplayground;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.nd4j.linalg.ops.transforms.Transforms.pow;

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

  // configuration
  private boolean useRegularization = true;
  private double lambda = 0.5;

  public LinearClassifier(final int height, final int width, final int nChannels, final int nLabels) {
    this.height = height;
    this.width = width;
    this.nChannels = nChannels;
    this.nLabels = nLabels;
    this.inputLength = this.height * this.width * nChannels;

    // create and initialize weight matrix
    weightsMatrix = Nd4j.rand(nLabels, inputLength, seed);
    biasMatrix = Nd4j.rand(nLabels, 1, seed);

    //LOG.debug("Created weights matrix W: \n{} \nshape: {}", weightsMatrix, weightsMatrix.shapeInfoToString());
    //LOG.debug("Created bias matrix b: \n{} \nshape: {}", biasMatrix, biasMatrix.shapeInfoToString());
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
   * @param scoreMatrix - 1d vector with scores for labels
   * @return 1d array with score values for each label (y)
   */
  public INDArray softmaxScore(INDArray scoreMatrix) {
    // s = f(x,W) ==> P = e^sk / sum_j(e^sj);

    INDArray exp = Transforms.exp(scoreMatrix);
    INDArray p = exp.div(exp.sumNumber().doubleValue());

    return p;
  }

  /**
   * Softmax loss
   * L = -log(e^syi / sum(e^sj)
   * @param scoreVector the score calculated for a given image
   * @param labelIndex the true label for this image
   * @return
   */
  public double softmaxLoss(INDArray scoreVector, int labelIndex) {

    LOG.debug("Label idx: {}", labelIndex);
    LOG.debug("scoreVector: {}", scoreVector);
    double scoreLabel = scoreVector.getDouble(labelIndex);
    LOG.debug("score Label: {}", scoreLabel);
    double tmpLoss = -Math.log10(scoreLabel);
    LOG.debug("loss {}", tmpLoss);

    if (useRegularization) {
      tmpLoss += regularization();
    }

    return tmpLoss;
  }

  /**
   * Multiclass SVM loss / hinge loss)
   * TODO: Could this be done without the loop - it is possible in numpy
   * TODO: numpy: np.maximum(0, scores - scores[y] +1)
   *
   * @param scoreVector the score calculated for a given image
   * @param labelIndex the true label for this image
   * @return
   */
  public double lossSVM(INDArray scoreVector, int labelIndex) {
    //LOG.debug("s = {}", scoreVector);
    //LOG.debug("yi = {}", labelIndex);
    double syi = scoreVector.getDouble(labelIndex, 0);
    //LOG.debug("syi = {}", syi);

    double tmpLoss = 0;

    // hinge lossSVM (sum of difference to correct label +1)
    for (int j=0; j<scoreVector.rows(); j++) {
      if (j==labelIndex) continue;
      double sj = scoreVector.getDouble(j, 0);
      tmpLoss += Math.max(0, sj - syi + 1); // alternative squared hinge
    }

    if (useRegularization) {
      tmpLoss += regularization();
    }

    return tmpLoss;
  }

  /**
   *
   * @return
   */
  private double regularization() {
    // L2: lambda * R(W) ==> R(W) = sum_k sum_l W^2k,l
    // L1: lambda * R(W) ==> R(W) = sum_k sum_l abs(Wk,l)
    // TODO: not sure if this is correct???
    double rLoss = lambda * Transforms.pow(weightsMatrix, 2).sumNumber().doubleValue();
    LOG.debug("rLoss = {}", rLoss);
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

}

