package fko.nnplayground.MinstNN;

import fko.nnplayground.API.Network;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.nd4j.linalg.ops.transforms.Transforms.exp;

/** SimpleNeuralNetwork */
public class SimpleNeuralNetwork implements Network {

  private static final Logger LOG = LoggerFactory.getLogger(SimpleNeuralNetwork.class);

  private final int inputLength;
  private final int nLabels;
  private final int seed;

  private INDArray W1;
  private INDArray b1;
  private INDArray W2;
  private INDArray b2;

  private int sizeHiddenLayer;
  private int epochs;
  private int iterations;
  private double learningRate;
  private boolean useRegularization;
  private double regularizationStrength;

  private Activations activationHiddenLayer;
  private Activations activationOutputLayer;

  private int totalIterations;

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
    useRegularization = false;
    regularizationStrength = 0.001;
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
   * Train the network with a given DataSet
   *
   * @param features (rows = number of inputs, columns = number of examples)
   * @param labels
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
   * TODO: add Bias
   * TODO: add Regularization
   * TODO: generalize for many layers
   * TODO: add other activations / SOFTMAX
   * TODO: add listener
   *
   * @param features
   * @param labels
   */
  private void optimize(final INDArray features, final INDArray labels) {

    // initialize weights matrices
    initWeights(seed);

    // Iterations
    for (int iteration = 0; iteration < iterations; iteration++) {

      // feed forward
      final INDArray layer_0 = features;
      final INDArray layer_1 = nonLin(Activations.SIGMOID, W1.mmul(layer_0));
      final INDArray layer_2 = nonLin(Activations.SIGMOID, W2.mmul(layer_1));

      // how much did we miss the target value?
      final INDArray layer_2_error = layer_2.sub(labels);
      final double loss = Transforms.abs(layer_2_error).meanNumber().doubleValue();
      totalIterations++;
      if (totalIterations % 100 == 0) {
        LOG.info("Loss at iteration {} (batch size {}) = {}",
                totalIterations, features.columns(), loss);
      }

      // in what direction is the target value?
      // were we really sure? if so, don't change too much.
      final INDArray layer_2_delta = layer_2_error.mul(nonLinDerivative(Activations.SIGMOID, layer_2));

      // how much did each l1 value contribute to the l2 error (according to the weights)?
      final INDArray layer_1_error = W2.transpose().mmul(layer_2_delta);

      // in what direction is the target l1?
      // were we really sure? if so, don't change too much.
      final INDArray layer_1_delta = layer_1_error.mul(nonLinDerivative(Activations.SIGMOID,layer_1));

      // update parameters
      W2.subi(layer_2_delta.mmul(layer_1.transpose()).mul(learningRate));
      W1.subi(layer_1_delta.mmul(layer_0.transpose()).mul(learningRate));
    }
  }

  private void initWeights(final int seed) {
    // XAVIER initialization:
    // ret = Nd4j.randn(order, shape).muli(FastMath.sqrt(2.0 / (fanIn + fanOut)));
    // Where fanIn(k) would be the number of units sending input to k, and fanOut(k)
    // would be the number of units receiving output from k.
    // fanIn = input length, fanOut = input length next layer (size of next layer)
    W1 = Nd4j.randn(this.sizeHiddenLayer, this.inputLength, seed)
                    .muli(FastMath.sqrt(2.0 / (inputLength + this.sizeHiddenLayer)));
    b1 = Nd4j.zeros(this.sizeHiddenLayer, 1);
    W2 = Nd4j.randn(this.nLabels, this.sizeHiddenLayer, this.seed)
                    .muli(FastMath.sqrt(2.0 / (this.sizeHiddenLayer + this.nLabels)));
    b2 = Nd4j.zeros(this.nLabels, 1);
  }



  private INDArray nonLin(final Activations fun, final INDArray in) {
    final INDArray out;
    switch (fun) {
      case IDENTITY:
        out = Transforms.identity(in);
        break;
      case SIGMOID:
        //out = Transforms.sigmoid(in);
        out = Transforms.pow(Transforms.exp(in.mul(-1)).add(1), -1);
        break;
      case TANH:
        out = Transforms.tanh(in);
        break;
      case RELU:
        out = Transforms.relu(in);
        break;
      case LEAKYRELU:
        out = Transforms.leakyRelu(in);
        break;
      case SOFTMAX:
        // Exercise to calculate this manually instead of library call
        // it's a bit faster as well :)

        // compute class probabilities
        // shift the values of f for each example so that the highest number is 0:
        // otherwise e^sk might become a NaN (too high)
        final INDArray s = in.subRowVector(in.max(0));
        // s = f(x,W) ==> P = e^sk / sum_j(e^sj);
        // get un-normalized probabilities
        final INDArray es = exp(s);
        // normalize them for each(!) example - each col sums to one
        out = es.divRowVector(es.sum(0));

        // softmax is a row operation so we need to transpose our
        // matrix before and the result after
        // output = Transforms.softmax(output.transpose()).transpose();
        // hiddenOutput = Transforms.softmax(hiddenOutput);
        break;
      default:
        out = Transforms.identity(in);
    }
    return out;
  }

  private INDArray nonLinDerivative(final Activations fun, final INDArray in) {

    final INDArray dIn;
    switch (fun) {
      case IDENTITY:
        // f(x) = x ==> f'(x) = 1
        dIn = in.assign(1);
        break;
      case SIGMOID:
        // dIn = Transforms.sigmoidDerivative(in); // Library is buggy
        dIn = in.mul(-1).add(1).mul(in);
        break;
      case TANH:
        // f'(x) = 1 - f(x)^2
        dIn = in.mul(in).mul(-1).add(1);
        break;
      case RELU:
        // ReLU(x) = max(0,x) ==> ReLU'(x) = 0 for <=0 OR 1 for >0
        dIn = in.dup();
        BooleanIndexing.replaceWhere(dIn, 0, Conditions.lessThanOrEqual(0));
        BooleanIndexing.replaceWhere(dIn, 1, Conditions.greaterThan(0));
        break;
      case LEAKYRELU:
        dIn = Transforms.leakyReluDerivative(in, 0);
        break;
//      case SOFTMAX: // only useful in output layer
        // get the correct label matrix and transpose it - this creates a matrix
        // identical in shape to scores with 1 where the correct score should have been
        // pk−1(yi=k)
        // the nonLinDerivative is identity pk except for all scores at the correct label there it is pk-1
        //dIn = in.sub(labels.transpose()).div(features.columns());
//        break;
      default:
        // f(x) = x ==> f'(x) = 1
        dIn = in.assign(1);
        break;
    }
    return dIn;
  }

//  private INDArray checkGradients(
//          final INDArray w, final INDArray input, final INDArray labels) {
//
//    double h = 1e-6;
//    INDArray tmpGWMatrix = Nd4j.zerosLike(w);
//    boolean tmpUseReg = useRegularization;
//    useRegularization = false;
//
//    // iterate over all weights
//    for (int j = 0; j < w.rows(); j++) {
//      for (int i = 0; i < w.columns(); i++) {
//
//        // evaluate current loss
//        INDArray output = forwardPass(input);
//        double loss = lossSoftmax(input, labels, output);
//
//        // safe old weight value
//        double oldVal = w.getDouble(j, i);
//        // add h
//        w.put(j, i, w.getDouble(j, i) + h);
//        // evaluate new loss with added h
//        INDArray hOutput = forwardPass(input);
//        double hLoss = lossSoftmax(input, labels, hOutput);
//        // restore old value
//        w.put(j, i, oldVal);
//
//        // compute gradient
//        double grad = (hLoss - loss) / h;
//        tmpGWMatrix.put(j, i, tmpGWMatrix.getDouble(j, i) + grad);
//      }
//    }
//    useRegularization = tmpUseReg;
//    return tmpGWMatrix;
//  }

  public boolean isUseRegularization() {
    return useRegularization;
  }

  public void setUseRegularization(final boolean useRegularization) {
    this.useRegularization = useRegularization;
  }

  public double getRegularizationStrength() {
    return regularizationStrength;
  }

  public void setRegularizationStrength(final double regularizationStrength) {
    this.regularizationStrength = regularizationStrength;
  }

  public Activations getActivationHiddenLayer() {
    return activationHiddenLayer;
  }

  public void setActivationHiddenLayer(final Activations activationHiddenLayer) {
    this.activationHiddenLayer = activationHiddenLayer;
  }

  public Activations getActivationOutputLayer() {
    return activationOutputLayer;
  }

  public void setActivationOutputLayer(final Activations activationOutputLayer) {
    this.activationOutputLayer = activationOutputLayer;
  }

  public double getLearningRate() {
    return learningRate;
  }

  public void setLearningRate(final double learningRate) {
    this.learningRate = learningRate;
  }
}
