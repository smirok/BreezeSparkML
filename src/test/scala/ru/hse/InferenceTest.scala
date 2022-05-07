package ru.hse

import breeze.linalg.{*, DenseMatrix, DenseVector}
import org.apache.spark.sql.SparkSession
import org.scalatest.funsuite.AnyFunSuite
import ru.hse.estimator.LinRegressionEstimator
import org.apache.spark.mllib.evaluation.RegressionMetrics

class InferenceTest extends AnyFunSuite {

  test("Inference") {
    val N = 10
    val d = 5

    val X = DenseMatrix.rand(N, d)
    val weights = DenseVector(1.5, 0.3, -0.7, 1.1, -3.0)
    val y = X * weights
    val data = DenseMatrix.horzcat(X, y.asDenseMatrix.t)

    val spark: SparkSession = SparkSession.builder.master("local[1]").appName("BreezeSparkML").getOrCreate()
    import spark.implicits._

    val df = data(*, ::).iterator
      .map(x => (x(0), x(1), x(2), x(3), x(4), x(5)))
      .toSeq.toDF("x1", "x2", "x3", "x4", "x5", "y")

    val model = new LinRegressionEstimator("1").setLabelCol("y").setMaxIter(100000).fit(df)

    val testX = DenseMatrix.rand(N, d)
    val testY = testX * weights

    val testDF = testX(*, ::).iterator
      .map(x => (x(0), x(1), x(2), x(3), x(4)))
      .toSeq.toDF("x1", "x2", "x3", "x4", "x5")

    val predictionsY = model.transform(testDF)
    val predictionAndObservations = predictionsY.join(testY.toArray.toSeq.toDF("observations")).rdd.map(x => (x(0), x(1)))

    val metrics = new RegressionMetrics(predictionAndObservations)
    assert(metrics.meanSquaredError < 5)
  }
}
