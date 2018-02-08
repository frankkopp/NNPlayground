package fko.nnplayground.MinstNN;

import fko.nnplayground.API.Network;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.nd4j.linalg.ops.transforms.Transforms.exp;
import static org.nd4j.linalg.ops.transforms.Transforms.max;

/**
 * SimpleNeuralNetwork
 */
public class SimpleNeuralNetwork implements Network {

  private static final Logger LOG = LoggerFactory.getLogger(SimpleNeuralNetwork.class);

  private final int height;
  private final int width;
  private final int depth;
  private final int nLabels;
  private final int seed;

  private INDArray W1;
  private INDArray b1;
  private INDArray W2;
  private INDArray b2;

  private int sizeHiddenLayer;
  private int miniBatchSize;

  public SimpleNeuralNetwork(final int height, final int width, final int channels, int nLabels, int sizeHiddenLayer, int seed) {
    this.height = height;
    this.width = width;
    this.depth = channels;
    this.nLabels = nLabels;
    this.sizeHiddenLayer = sizeHiddenLayer;
    this.seed = seed;

    int inputLength = height * width * channels;

    W1 = Nd4j.rand(this.sizeHiddenLayer, inputLength, this.seed);
    b1 = Nd4j.zeros(this.sizeHiddenLayer, 1);
    W2 = Nd4j.rand(this.nLabels, this.sizeHiddenLayer, this.seed);
    b2 = Nd4j.zeros(this.nLabels, 1);
  }

  public void train(DataSetIterator dataSetIter, int epochs, int iterations, double learningRate) {

    // Epoch
    for (int epoch = 0; epoch < epochs; epoch++) {
      LOG.debug("Train epoch {} of {}:", epoch + 1, epochs);

      // Iterations
      int totalIterations = 1;
      for (int iteration=0; iteration<iterations; iteration++) {
        LOG.debug("Train iteration {} of {}:", iteration+1, iterations);

        // Batch
        while (dataSetIter.hasNext()) { // one batch
          totalIterations++;
          LOG.debug("Train iteration {}:", totalIterations);

          // get the next batch of examples
          DataSet batch = dataSetIter.next();
          LOG.debug("Batch has {} examples:", batch.numExamples());

          // do forward pass to compute scores
          INDArray scores = forwardPass(batch);
          LOG.debug("Scores:\n{}\n{}", scores, scores.shapeInfoToString());

          // compute loss for all examples
          // TODO

          // update weights based on loss
          // TODO

        }
      }
      break; // while debugging only use the first batch until this works
    }
  }

  /**
   * Calculate the scores for each example in the batch.
   * Uses ReLU for hidden output and probabilities for classes (in preparation for Softmax)
   * @param batch the batch of examples
   * @return an array of scores for all labels of all examples in the batch
   */
  private INDArray forwardPass(final DataSet batch) {

    final int numExamples = batch.numExamples();

    // each example from batch
    // TODO: can we do this this without the loop??
    // hold all scores for the whole batch
    INDArray scores = Nd4j.zeros(nLabels, numExamples);
    for (int x_i = 0; x_i < numExamples; x_i++) {

      // evaluate class scores
      final INDArray dataT = batch.get(x_i).getFeatures().ravel().transpose(); // n-dim array to vector
      final INDArray hiddenOutput = max(W1.mmul(dataT).add(b1), 0d); // ReLU
      final INDArray output = W2.mmul(hiddenOutput).add(b2);

      // compute class probabilities
      // s = f(x,W) ==> P = e^sk / sum_j(e^sj);
      // shift the values of f so that the highest number is 0:
      // otherwise e^sk might become a NaN (too high)
      final INDArray s = output.sub(output.maxNumber());
      final INDArray exp = exp(s);
      final INDArray outputP = exp.div(exp.sumNumber().doubleValue());

      // save scores
      scores.putColumn(x_i, outputP);
    }
    return scores;
  }

  // TODO - listener pattern for TrainingListener

}
