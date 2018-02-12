package fko.nnplayground.nn;

import fko.nnplayground.API.ILayer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Layer
 */
public class Layer implements ILayer {

  private static final Logger LOG = LoggerFactory.getLogger(Layer.class);

  private final int inputSize;
  private final int outputSize;
  private final int seed;

  private Activation.Activations activationFunction;

  private final WeightInitializer.WeightInit weightInit;

  private INDArray weightsMatrix;
  private INDArray biasMatrix;

  private INDArray output; // before the non linearity function
  private INDArray activation; // after the non linearity

  private INDArray layerGradient; // gradient for layer

  private INDArray previousLayerError; // error on the previous layer

  public Layer(final int inputSize, final int outputSize, final WeightInitializer.WeightInit weightInit, final Activation.Activations activationFunction, final int seed) {
    this.inputSize = inputSize;
    this.outputSize = outputSize;
    this.weightInit = weightInit;
    this.activationFunction = activationFunction;
    this.seed = seed;

    weightsMatrix = WeightInitializer.initWeights(this.weightInit, this.outputSize, this.inputSize, this.seed);
    biasMatrix = WeightInitializer.initWeights(WeightInitializer.WeightInit.ZERO, this.outputSize, 1, this.seed);

    LOG.debug("Created layer of type {}. InputSize: {} OutputSize: {} WeightInit: {} Activation: {}",
            getClass().getSimpleName(), inputSize, outputSize, weightInit, activationFunction);
  }

  @Override
  public INDArray forwardPass(final INDArray outputLastLayer) {
    final INDArray WdotOutput = weightsMatrix.mmul(outputLastLayer);
    output = WdotOutput.addColumnVector(biasMatrix);
    activation = Activation.applyActivation(activationFunction, output);
    return getActivation();
  }

  @Override
  public INDArray backwardPass(INDArray error) {
    final INDArray derivative = Activation.applyDerivative(activationFunction, activation);
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
  public void setActivationFunction(final Activation.Activations function) {
    this.activationFunction = function;
  }

  @Override
  public Activation.Activations getActivationFunction() {
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

  @Override
  public WeightInitializer.WeightInit getWeightInit() {
    return weightInit;
  }

  @Override
  public String toString() {
    return "Layer{" +
            "inputSize=" + inputSize +
            ", outputSize=" + outputSize +
            ", seed=" + seed +
            ", activationFunction=" + activationFunction +
            ", weightInit=" + weightInit +
            '}';
  }
}
