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

package fko.nnplayground.UseCases;

import com.sun.javafx.application.PlatformImpl;
import javafx.application.Application;
import javafx.application.Platform;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.control.ScrollPane;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.FlowPane;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.stage.Stage;
import org.jetbrains.annotations.NotNull;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** CifarViewer */
public class CifarViewer extends Application {

  private static final Logger LOG = LoggerFactory.getLogger(CifarViewer.class);

  private Stage primaryStage;
  private final FlowPane flowPane = new FlowPane();

  private final int width = 32;
  private final int height = 32;
  private int zoom = 2;

  public CifarViewer() {

    // Startup the JavaFX platform
    Platform.setImplicitExit(true);
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

  @Override
  public void start(final Stage stage) {

    // raw score out put row
    ScrollPane scrollPane = new ScrollPane(flowPane);
    scrollPane.setFitToWidth(true);

    // horizontal box with all rows
    VBox root = new VBox();
    root.setAlignment(Pos.TOP_LEFT);
    root.getChildren().addAll(scrollPane);

    Scene scene = new Scene(root, 520, 300);
    stage.setScene(scene);
    stage.setTitle("Cifar Viewer");
    stage.setResizable(true);
    stage.show();
  }

  public void showImage(final INDArray dataArray) {
//    LOG.debug(
//            "NDArray: \n{} \nshape:\n{}",
//            dataArray,
//            dataArray.shapeInfoToString());

    Image cifarImage = createImage(dataArray);

    ImageView imageView = new ImageView(cifarImage);
    imageView.setFitHeight(zoom * height);
    imageView.setFitWidth (zoom * width);

    Platform.runLater(() -> flowPane.getChildren().add(imageView));
  }

  @NotNull
  private Image createImage(final INDArray dsRow) {

    WritableImage cifarImage = new WritableImage(width, height);

    int w = (int) cifarImage.getWidth();
    int h = (int) cifarImage.getHeight();

    PixelWriter writer = cifarImage.getPixelWriter();

    for (int i = 0; i < w; i++) {
      for (int j = 0; j < h; j++) {
        double red = dsRow.getDouble(0,0,j,i);
        double green = dsRow.getDouble(0,1,j,i);
        double blue = dsRow.getDouble(0,2,j,i);
        Color c = new Color(red, green, blue, 1.0);
        writer.setColor(i, j, c);
      }
    }
    return cifarImage;
  }
}
