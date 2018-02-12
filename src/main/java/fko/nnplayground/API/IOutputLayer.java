package fko.nnplayground.API;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface IOutputLayer extends ILayer {

  /**
   * Extends the forwardPass from ILayer to also calculate error and totalError in this pass.
   * So there is no need to use computeError or computeTotalError after this. Just use get...()
   * @see fko.nnplayground.API.ILayer#forwardPass(INDArray)
   */
  INDArray forwardPass(INDArray outputLastLayer, boolean computeError);

  /**
   * @param training whether we compute this during training or outside of training
   * @return the array of errors for each example
   */
  INDArray computeError(boolean training);

  /**
   * @param training whether we compute this during training or outside of training
   * @return the array of errors for all examples (total loss)
   */
  double computeTotalError(boolean training);

  /**
   * Uses the internal error calculated based on the labels
   * @see ILayer#backwardPass(INDArray)
   * @return
   */
  INDArray backwardPass();

  INDArray getError();
  double getTotalError();

  INDArray getLabels();

}
