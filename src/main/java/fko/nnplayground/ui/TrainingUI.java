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
import fko.nnplayground.API.Network;
import fko.nnplayground.API.TrainingListener;
import javafx.application.Application;
import javafx.application.Platform;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.text.Text;
import javafx.stage.Stage;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;

/** TrainingUI */
public class TrainingUI extends Application implements TrainingListener {

  private static final Logger LOG = LoggerFactory.getLogger(TrainingUI.class);

  public static final int updateInterval = 1; // ms

  private Stage primaryStage;
  private Text score;
  private XYChart.Series scoreSeries;
  private XYChart.Series f1Series;

  private final MultiLayerNetwork multiLayerNetwork;
  private Evaluation evaluation;
  private final int interval;

  public TrainingUI(final MultiLayerNetwork network) {
    this(network, 1);
  }

  public TrainingUI(final MultiLayerNetwork network, int interval) {

    this.multiLayerNetwork = network;
    this.interval = interval;
    this.evaluation = null;

    LOG.info("Starting TrainingUI with interval {} iterations", interval);

    // Startup the JavaFX platform
    Platform.setImplicitExit(false);
    PlatformImpl.startup(
        () -> {
          primaryStage = new Stage();
          start(primaryStage);
        });

    // wait for the UI to show before returning
    do {
      try {
        Thread.sleep(100);
      } catch (InterruptedException ignore) {
      }
    } while (primaryStage == null || !primaryStage.isShowing());
  }

  private void updateUI(final int iteration) {

    if (multiLayerNetwork == null) {
      score.setText("N/A");
    } else {
      double score = multiLayerNetwork.score();
      this.score.setText("" + score);
      if (score > 0) {
        scoreSeries.getData().add(new XYChart.Data(iteration, score));
      }
      if (evaluation != null) {
        double f1 = evaluation.f1();
        evaluation.stats();
        if (f1 > 0) {
          f1Series.getData().add(new XYChart.Data(iteration, f1));
        }
      }
    }
  }

  @Override
  public void start(final Stage stage) {

    // raw score out put row
    HBox scoreRow = new HBox();
    scoreRow.setAlignment(Pos.CENTER);
    Text label = new Text("Score: ");
    score = new Text("n/a");
    scoreRow.getChildren().addAll(label, score);

    // graph
    // defining the axes
    final NumberAxis xAxis = new NumberAxis();
    final NumberAxis yAxis = new NumberAxis();
    xAxis.setLabel("Iterations");
    yAxis.setLabel("Score");
    // creating the chart
    final LineChart<Number, Number> lineChart = new LineChart<Number, Number>(xAxis, yAxis);
    lineChart.setTitle("Training Score");
    lineChart.setCreateSymbols(false); // hide dots
    lineChart.setMinHeight(500);
    // defining a scoreSeries
    scoreSeries = new XYChart.Series();
    scoreSeries.setName("Loss Score");
    f1Series = new XYChart.Series();
    f1Series.setName("F1 Score");
    lineChart.getData().addAll(scoreSeries, f1Series);

    // horizontal box with all rows
    VBox root = new VBox();
    root.setAlignment(Pos.CENTER);
    root.getChildren().addAll(scoreRow);
    root.getChildren().addAll(lineChart);

    Scene scene = new Scene(root, 800, 600);
    stage.setScene(scene);
    stage.setTitle("Training Info");
    stage.setResizable(true);
    stage.show();
  }

  @Override
  public void iterationDone(final Network nn, final int iteration) {
    if (iteration % interval == 0) {
      Platform.runLater(() -> updateUI(iteration));
    }
  }

  @Override
  public void onEpochStart(final Network nn) {

  }

  @Override
  public void onEpochEnd(final Network nn) {

  }

  @Override
  public void onForwardPass(final Network nn, final List<INDArray> activations) {

  }

  @Override
  public void onForwardPass(final Network nn, final Map<String, INDArray> activations) {

  }

  @Override
  public void onGradientCalculation(final Network nn) {

  }

  @Override
  public void onBackwardPass(final Network nn) {

  }

  public Evaluation getEvaluation() {
    return evaluation;
  }

  public void setEvaluation(final Evaluation evaluation) {
    this.evaluation = evaluation;
  }

}
