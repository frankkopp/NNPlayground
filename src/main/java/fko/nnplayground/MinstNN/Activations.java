package fko.nnplayground.MinstNN;

/**
 * Transfer Functions / Activation Functions
 */
public enum Activations {

  /**
   * no op
   * f(x) = x
   */
  IDENTITY,

  /**
   * Sigmoid or Logistic Activation Function
   * Squish the output between 0 and 1
   * 1 / (1 + e^(-activation))
   */
  SIGMOID,

  /**
   * Tanh or hyperbolic tangent Activation Function
   * Squish the output between -1 and 1
   * 1 - (2 / (e^2x - 1))
   */
  TANH,

  /**
   * ReLU (Rectified Linear Unit) Activation Function
   * 1 / (1 + e^-x))
   */
  RELU,

  /**
   * Leaky ReLU
   * <0: f(x) = ax (a usually 0.01
   * >0: f(x) = x
   */
  LEAKYRELU,

  /**
   * Softmax
   */
  SOFTMAX


}
