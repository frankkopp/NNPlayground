package fko.nnplayground;

import org.nd4j.linalg.api.ndarray.INDArray;
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

  private final int height;
  private final int width;
  private final int nChannels;
  private final int nLabels;
  private final int inputLength;

  private final INDArray weightsMatrix; // W
  private final INDArray biasMatrix; // b

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
   * Multiclass SVM loss /hinge loss)
   * @param scoreVector the score calculated for a given image
   * @param labelIndex the true label for this image
   * @return
   */
  public double loss(INDArray scoreVector, int labelIndex) {
    LOG.debug("s = {}", scoreVector);
    LOG.debug("yi = {}", labelIndex);
    double syi = scoreVector.getDouble(labelIndex, 0);
    LOG.debug("syi = {}", syi);

    double tmpLoss = 0;

    // hinge loss (sum of difference to correct label +1)
    for (int j=0; j<scoreVector.rows(); j++) {
      if (j==labelIndex) continue;
      double sj = scoreVector.getDouble(j, 0);
      tmpLoss += Math.max(0, sj - syi + 1); // alternative squared hinge
    }

    return tmpLoss;
  }

  public INDArray getWeightsMatrix() {
    return weightsMatrix;
  }

  public INDArray getBiasMatrix() {
    return biasMatrix;
  }

}

