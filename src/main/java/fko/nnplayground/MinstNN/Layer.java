package fko.nnplayground.MinstNN;

import fko.nnplayground.API.ILayer;
import org.apache.commons.math3.util.FastMath;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.nd4j.linalg.ops.transforms.Transforms.exp;

/**
 * Layer
 */
public class Layer implements ILayer {

  private static final Logger LOG = LoggerFactory.getLogger(Layer.class);

  private final int inputSize;
  private final int outputSize;
  private final int seed;

  private Activations activationFunction;

  private INDArray weightsMatrix = null;
  private INDArray biasMatrix = null;

  private INDArray output; // before the non linearity function
  private INDArray activation; // after the non linearity


  private INDArray layerGradient; // gradient for layer

  private INDArray previousLayerError; // error on the previous layer

  public Layer(final int inputSize, final int outputSize, final Activations activationFunction, final int seed) {
    this.inputSize = inputSize;
    this.outputSize = outputSize;
    this.activationFunction = activationFunction;
    this.seed = seed;

    initWeights();
  }

  private void initWeights() {
    // XAVIER initialization:
    // ret = Nd4j.randn(order, shape).muli(FastMath.sqrt(2.0 / (fanIn + fanOut)));
    // Where fanIn(k) would be the number of units sending input to k, and fanOut(k)
    // would be the number of units receiving output from k.
    // fanIn = input length, fanOut = input length next layer (outputSize of next layer)
    weightsMatrix = Nd4j.randn(outputSize, inputSize, seed)
            .muli(FastMath.sqrt(2.0 / (inputSize + outputSize)));
    biasMatrix = Nd4j.zeros(outputSize, 1);
  }

  @Override
  public INDArray forwardPass(final INDArray outputLastLayer) {
    final INDArray WdotOutput = weightsMatrix.mmul(outputLastLayer);
    output = WdotOutput.addColumnVector(biasMatrix);
    activation = nonLin(activationFunction, output);
    return getActivation();
  }

  @Override
  public INDArray backwardPass(INDArray error) {
    final INDArray derivative = nonLinDerivative(activationFunction, activation);
    layerGradient = error.mul(derivative);
    previousLayerError = weightsMatrix.transpose().mmul(layerGradient);
    return getPreviousLayerError();
  }

  @Override
  public void updateWeights(final INDArray activationPreviousLayer, final double learningRate) {
    // full change of weights based on gradient and layer_1 output
    final INDArray W2_delta = layerGradient.mmul(activationPreviousLayer.transpose());
    // multiplied with learning rate to adjust step size
    final INDArray W2_change = W2_delta.mul(learningRate);
    // update W2
    weightsMatrix.subi(W2_change);
    biasMatrix.subi(layerGradient.sum(1).mul(learningRate));
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

  @Override
  public int getSeed() {
    return this.seed;
  }

  @Override
  public int getInputSize() {
    return inputSize;
  }

  @Override
  public int getOutputSize() {
    return outputSize;
  }

  @Override
  public void setActivationFunction(final Activations function) {
    this.activationFunction = function;
  }

  @Override
  public Activations getActivationFunction() {
    return activationFunction;
  }

  @Override
  public INDArray getOutput() {
    return output.dup();
  }

  @Override
  public INDArray getActivation() {
    return activation.dup();
  }

  @Override
  public INDArray getWeightsMatrix() {
    return weightsMatrix.dup();
  }

  @Override
  public INDArray getBiasMatrix() {
    return biasMatrix.dup();
  }

  @Override
  public INDArray getLayerGradient() {
    return layerGradient.dup();
  }

  @Override
  public INDArray getPreviousLayerError() {
    return previousLayerError.dup();
  }
}
