package fko.nnplayground.API;

import fko.nnplayground.nn.Activation;
import fko.nnplayground.nn.WeightInitializer;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.DataOutputStream;

/**
 * TODO docs
 */
public interface ILayer {

  /**
   * Computes the layer's forward pass
   * @param outputLastLayer the input for the layer
   * @return the activation of the layer (after activation function)
   */
  INDArray forwardPass(INDArray outputLastLayer);

  /**
   * @return the output of the layer before the nonLin activation function
   */
  INDArray getOutput();

  /**
   * @param error of this layer (back propagated from next layer)
   * @return the error of the previous layer
   */
  INDArray backwardPass(INDArray error);

  /**
   * @param activationPreviousLayer
   * @param learningRate the factor for update steps on the weights
   */
  void updateWeights(final INDArray activationPreviousLayer, double learningRate);

  /*
   * Getters and Setters
   */

  WeightInitializer.WeightInit getWeightInit();
  INDArray getActivation();

  INDArray getWeightsMatrix();
  INDArray getBiasMatrix();

  void setActivationFunction(Activation.Activations function);
  Activation.Activations getActivationFunction();

  int getSeed();

  int getInputSize();
  int getOutputSize();

  INDArray getLayerGradient();

  INDArray getPreviousLayerError();

}
