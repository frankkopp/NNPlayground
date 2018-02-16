/*
 * MIT License
 *
 * Copyright (c) 2018 Frank Kopp
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */

package fko.nnplayground.ui;

import com.sun.javafx.application.PlatformImpl;
import fko.nnplayground.API.ILayer;
import fko.nnplayground.API.INeuralNetwork;
import fko.nnplayground.API.ITrainingListener;
import javafx.application.Application;
import javafx.application.Platform;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.XYChart;
import javafx.scene.control.ChoiceBox;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.control.Tooltip;
import javafx.scene.layout.AnchorPane;
import javafx.scene.layout.FlowPane;
import javafx.scene.paint.Color;
import javafx.scene.shape.Rectangle;
import javafx.stage.Stage;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * TrainingUI
 *
 * Implements the ITrainingListener interface it can observe an INeuralNetwork.
 * Starts up JavaFX when instantiating the class.
 */
public class TrainingUI extends Application implements ITrainingListener {

  private static final Logger LOG = LoggerFactory.getLogger(TrainingUI.class);

  private final INeuralNetwork neuralNetwork;
  private final int interval;
  private ILayer chosenLayer;

  private Stage primaryStage;
  private AnchorPane root;

  private XYChart.Series scoreSeries;
  private XYChart.Series f1Series;
  private XYChart.Series accuracySeries;

  private ObservableList<ILayer> layerList;

  /**
   * Creates the TrainingUI and starts up the JavaFX platform and shows windows.
   * @param network the network to observe
   * @param interval any n-th (n=interval) call will update the UI
   */
  public TrainingUI(final INeuralNetwork network, int interval) {

    this.neuralNetwork = network;
    this.interval = interval;

    LOG.info("Starting TrainingUI with interval {} iterations", interval);

    // Start JavaFX platform
    PlatformImpl.startup(
            () -> {
              // Startup the JavaFX platform
              primaryStage = new Stage();
              start(primaryStage);
            });
    Platform.setImplicitExit(true);

    // wait for the UI to show before returning
    do {
      try {
        Thread.sleep(100);
      } catch (InterruptedException ignore) {
      }
    } while (primaryStage == null || !primaryStage.isShowing());
  }

  @Override
  public void start(final Stage stage) {

    LOG.info("Starting JavaFX Application");

    // read FXML file and setup UI
    FXMLLoader fxmlLoader = new FXMLLoader(getClass().getResource("/fxml/TrainingUI.fxml"));
    fxmlLoader.setController(this);
    try {
      root = fxmlLoader.load();
    } catch (IOException e) {
      e.printStackTrace();
    }

    // FXML init
    initialize();

    // set initial window title - will be extended in controller
    primaryStage.setTitle("Training Monitor by Frank Kopp");
    // setup window
    final Scene scene = new Scene(root);
    primaryStage.setScene(scene);
    primaryStage.centerOnScreen();
    primaryStage.setResizable(true);

    // extend the FXML view
    addAdditionalViews();

    // closeAction
    primaryStage.setOnCloseRequest(event -> System.exit(0));

    // finally show window
    primaryStage.show();

    LOG.info("JavaFX Application started");

  }

  private void addAdditionalViews() {
    // learning Score Chart
    learningScoreChart.setCreateSymbols(false); // hide dots
    learningScoreChart.setVerticalGridLinesVisible(false);
    learningScoreChart.setLegendVisible(false);
    scoreSeries = new XYChart.Series();
    learningScoreChart.getData().addAll(scoreSeries);

    // f1 Score chart
    f1ScoreChart.setCreateSymbols(false); // hide dots
    f1ScoreChart.setVerticalGridLinesVisible(false);
    f1ScoreChart.setLegendVisible(false);
    f1Series = new XYChart.Series();
    f1ScoreChart.getData().addAll(f1Series);

    // accuracy Score chart
    accuracyChart.setCreateSymbols(false); // hide dots
    accuracyChart.setVerticalGridLinesVisible(false);
    accuracyChart.setLegendVisible(false);
    accuracySeries = new XYChart.Series();
    accuracyChart.getData().addAll(accuracySeries);

    // choose layer dropdown
    layerList = FXCollections.observableArrayList();
    chooseLayer.setItems(layerList);
    chooseLayer.getSelectionModel().selectedIndexProperty()
               .addListener((ov, value, new_value) -> chooseLayerUpdateView(new_value));

  }

  /**
   * is called when a new layer is chosen in the dropdown
   * @param new_value
   */
  private void chooseLayerUpdateView(final Number new_value) {
    final int layerIdx = new_value.intValue();
    chosenLayer = neuralNetwork.getLayerList().get(layerIdx);

    numberOfRows.setText("" + chosenLayer.getOutputSize());
    numberOfColumns.setText("1");

    updateActivationView();

  }

  /**
   * Is chosen whenever the activation view shall be updated. Usually called after
   * the n-th (n=interval) iteration.
   */
  private void updateActivationView() {
    if (chosenLayer != null && chosenLayer.getActivation() != null) {

      final int rectangleSize = 15;

      final INDArray layerActivation = chosenLayer.getActivation().getColumn(0);

      activationPane.getChildren().clear();
      final int examples = layerActivation.columns();
      for (int j = 0; j < examples; j++) {
        for (int i = 0; i < layerActivation.rows(); i++) {
          final double neuronActivation = layerActivation.getDouble(i, j);
          final int rgbValue = (int) (neuronActivation * 255);
          Color color = Color.rgb(rgbValue, rgbValue, rgbValue);
          // potential dead neurons
          if (neuronActivation < 0.1) {
            color = Color.LIGHTBLUE;
          } else if (neuronActivation > 0.9) {
            color = Color.ORANGERED;
          }
          Rectangle rectangle = new Rectangle(rectangleSize, rectangleSize);
          Tooltip tooltip = new Tooltip("" + neuronActivation);
          Tooltip.install(rectangle, tooltip);
          rectangle.setStroke(Color.WHITE);
          rectangle.setStrokeWidth(2.0);
          rectangle.setFill(color);
          activationPane.getChildren().add(rectangle);
        }
      }
    }
  }

  @Override
  public void onTrainStart() {
    Platform.runLater(this::updateViewsOnTrainStart);
  }

  private void updateViewsOnTrainStart() {
    layerList.addAll(neuralNetwork.getLayerList());
    chooseLayer.getSelectionModel().selectLast();
    inputsLabel.setText("" + neuralNetwork.getInputLength());
    outputsLabel.setText("" + neuralNetwork.getOutputLength());
    layersLabel.setText("" + neuralNetwork.getLayerList().size());
  }

  @Override
  public void onTrainEnd() {

  }

  @Override
  public void onEpochStart(final int epoch, final int batchSize) {
    Platform.runLater(() -> {
      currentEpochLabel.setText("" + epoch);
      batchSizeLabel.setText("" + batchSize);
    });
  }

  @Override
  public void onEpochEnd() {

  }

  @Override
  public void iterationDone(final int iteration) {
    if (iteration % interval == 0) {
      Platform.runLater(() -> updateViewAfterIteration(iteration));
    }
  }

  private void updateViewAfterIteration(final int iteration) {

    currentIterationLabel.setText("" + iteration);

    // total examples seen
    examplesLabel.setText("" + neuralNetwork.getExamplesSeenTraining());

    // score label update
    double score = neuralNetwork.getCurrentScore();
    currentScoreLabel.setText("" + score);
    // score chart upate
    if (score > 0) {
      scoreSeries.getData().add(new XYChart.Data<>(Integer.toString(iteration), score));
    }

    // learning Rate - could be changed after each iteration
    learningRateLabel.setText("" + neuralNetwork.getLearningRate());

    // should be per layer - we simplify here and only use first layer
    if (neuralNetwork.getLayerList().size() > 0) {
      l2StrengthLabel.setText("" + neuralNetwork.getLayerList().get(0).getL2Strength());
    }

    // score chart update
    f1Series.getData().add(new XYChart.Data<>(Integer.toString(iteration), neuralNetwork.getF1score()));

    // accuracy chart update
    accuracySeries.getData().add(new XYChart.Data<>(Integer.toString(iteration), neuralNetwork.getAccuracy()));

    recallLabel.setText("" + neuralNetwork.getRecall());
    precisionLabel.setText("" + neuralNetwork.getPrecision());
    accuracyLabel.setText("" + neuralNetwork.getAccuracy());
    f1scoreLabel.setText("" + neuralNetwork.getF1score());

    updateActivationView();
  }

  @Override
  public void onForwardPass(final List<INDArray> activations) {

  }

  @Override
  public void onForwardPass(final Map<String, INDArray> activations) {

  }

  @Override
  public void onGradientCalculation() {

  }

  @Override
  public void onBackwardPass() {

  }

  @Override
  public void onEvalStart() {

  }

  @Override
  public void onEvalEnd() {
    Platform.runLater(() -> {
      recallEvalLabel.setText("" + neuralNetwork.getRecall());
      precisionEvalLabel.setText("" + neuralNetwork.getPrecision());
      accuracyEvalLabel.setText("" + neuralNetwork.getAccuracy());
      f1ScoreEvalLabel.setText("" + neuralNetwork.getF1score());
      examplesEvalLabel.setText("" + neuralNetwork.getExamplesSeenEval());
    });
  }

  @FXML // fx:id="currentScoreLabel"
  private Label currentScoreLabel; // Value injected by FXMLLoader

  @FXML // fx:id="learningRateLabel"
  private Label learningRateLabel; // Value injected by FXMLLoader

  @FXML // fx:id="l2StrengthLabel"
  private Label l2StrengthLabel; // Value injected by FXMLLoader

  @FXML // fx:id="currentIterationLabel"
  private Label currentIterationLabel; // Value injected by FXMLLoader

  @FXML // fx:id="currentEpochLabel"
  private Label currentEpochLabel; // Value injected by FXMLLoader

  @FXML // fx:id="batchSizeLabel"
  private Label batchSizeLabel; // Value injected by FXMLLoader

  @FXML // fx:id="inputsLabel"
  private Label inputsLabel; // Value injected by FXMLLoader

  @FXML // fx:id="outputsLabel"
  private Label outputsLabel; // Value injected by FXMLLoader

  @FXML // fx:id="layersLabel"
  private Label layersLabel; // Value injected by FXMLLoader

  @FXML // fx:id="chooseLayer"
  private ChoiceBox<ILayer> chooseLayer; // Value injected by FXMLLoader

  @FXML // fx:id="numberOfRows"
  private TextField numberOfRows; // Value injected by FXMLLoader

  @FXML // fx:id="numberOfColumns"
  private TextField numberOfColumns; // Value injected by FXMLLoader

  @FXML // fx:id="activationPane"
  private FlowPane activationPane; // Value injected by FXMLLoader

  @FXML // fx:id="learningScoreChart"
  private LineChart<?, ?> learningScoreChart; // Value injected by FXMLLoader

  @FXML // fx:id="accuracyChart"
  private LineChart<?, ?> accuracyChart; // Value injected by FXMLLoader

  @FXML // fx:id="f1ScoreChart"
  private LineChart<?, ?> f1ScoreChart; // Value injected by FXMLLoader

  @FXML // fx:id="recallLabel"
  private Label recallLabel; // Value injected by FXMLLoader

  @FXML // fx:id="precisionLabel"
  private Label precisionLabel; // Value injected by FXMLLoader

  @FXML // fx:id="accuracyLabel"
  private Label accuracyLabel; // Value injected by FXMLLoader

  @FXML // fx:id="f1scoreLabel"
  private Label f1scoreLabel; // Value injected by FXMLLoader

  @FXML // fx:id="examplesLabel"
  private Label examplesLabel; // Value injected by FXMLLoader

  @FXML // fx:id="examplesEvalLabel"
  private Label examplesEvalLabel; // Value injected by FXMLLoader

  @FXML // fx:id="recallEvalLabel"
  private Label recallEvalLabel; // Value injected by FXMLLoader

  @FXML // fx:id="precisionEvalLabel"
  private Label precisionEvalLabel; // Value injected by FXMLLoader

  @FXML // fx:id="accuracyEvalLabel"
  private Label accuracyEvalLabel; // Value injected by FXMLLoader

  @FXML // fx:id="accuracyEvalLabel"
  private Label f1ScoreEvalLabel; // Value injected by FXMLLoader

  @FXML
    // This method is called by the FXMLLoader when initialization is complete
  void initialize() {
    assert currentScoreLabel != null :
            "fx:id=\"currentScoreLabel\" was not injected: check your FXML file 'TrainingUI.fxml'.";
    assert learningRateLabel != null :
            "fx:id=\"learningRateLabel\" was not injected: check your FXML file 'TrainingUI.fxml'.";
    assert l2StrengthLabel != null :
            "fx:id=\"l2StrengthLabel\" was not injected: check your FXML file 'TrainingUI.fxml'.";
    assert currentIterationLabel != null :
            "fx:id=\"currentIterationLabel\" was not injected: check your FXML file 'TrainingUI.fxml'.";
    assert currentEpochLabel != null :
            "fx:id=\"currentEpochLabel\" was not injected: check your FXML file 'TrainingUI.fxml'.";
    assert batchSizeLabel != null :
            "fx:id=\"batchSizeLabel\" was not injected: check your FXML file 'TrainingUI.fxml'.";
    assert inputsLabel != null : "fx:id=\"inputsLabel\" was not injected: check your FXML file 'TrainingUI.fxml'.";
    assert outputsLabel != null : "fx:id=\"outputsLabel\" was not injected: check your FXML file 'TrainingUI.fxml'.";
    assert layersLabel != null : "fx:id=\"layersLabel\" was not injected: check your FXML file 'TrainingUI.fxml'.";
    assert chooseLayer != null : "fx:id=\"chooseLayer\" was not injected: check your FXML file 'TrainingUI.fxml'.";
    assert numberOfRows != null : "fx:id=\"numberOfRows\" was not injected: check your FXML file 'TrainingUI.fxml'.";
    assert numberOfColumns != null :
            "fx:id=\"numberOfColumns\" was not injected: check your FXML file 'TrainingUI.fxml'.";
    assert activationPane != null :
            "fx:id=\"activationPane\" was not injected: check your FXML file 'TrainingUI.fxml'.";
    assert learningScoreChart != null :
            "fx:id=\"learningScoreChart\" was not injected: check your FXML file 'TrainingUI.fxml'.";
    assert accuracyChart != null : "fx:id=\"accuracyChart\" was not injected: check your FXML file 'TrainingUI.fxml'.";
    assert f1ScoreChart != null : "fx:id=\"f1ScoreChart\" was not injected: check your FXML file 'TrainingUI.fxml'.";
    assert recallLabel != null : "fx:id=\"recallLabel\" was not injected: check your FXML file 'TrainingUI.fxml'.";
    assert precisionLabel != null :
            "fx:id=\"precisionLabel\" was not injected: check your FXML file 'TrainingUI.fxml'.";
    assert accuracyLabel != null : "fx:id=\"accuracyLabel\" was not injected: check your FXML file 'TrainingUI.fxml'.";
    assert f1scoreLabel != null : "fx:id=\"f1scoreLabel\" was not injected: check your FXML file 'TrainingUI.fxml'.";
    assert examplesLabel != null : "fx:id=\"examplesLabel\" was not injected: check your FXML file 'TrainingUI.fxml'.";
    assert examplesEvalLabel != null :
            "fx:id=\"examplesEvalLabel\" was not injected: check your FXML file 'TrainingUI.fxml'.";
    assert recallEvalLabel != null :
            "fx:id=\"recallEvalLabel\" was not injected: check your FXML file 'TrainingUI.fxml'.";
    assert precisionEvalLabel != null :
            "fx:id=\"precisionEvalLabel\" was not injected: check your FXML file 'TrainingUI.fxml'.";
    assert accuracyEvalLabel != null :
            "fx:id=\"accuracyEvalLabel\" was not injected: check your FXML file 'TrainingUI.fxml'.";
    assert f1ScoreEvalLabel != null :
            "fx:id=\"f1ScoreEvalLabel\" was not injected: check your FXML file 'TrainingUI.fxml'.";
  }

}
