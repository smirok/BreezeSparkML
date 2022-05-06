package ru.hse

import breeze.linalg.{*, DenseMatrix, DenseVector}
import org.apache.spark.sql.SparkSession
import ru.hse.estimator.LinRegressionEstimator


object Main {
  private val N = 100
  private val d = 3
  val spark: SparkSession = SparkSession.builder.master("local[1]").appName("BreezeSparkML").getOrCreate()

  def main(args: Array[String]): Unit = {
    import spark.implicits._

    val X = DenseMatrix.rand(N, d)
    val weights = DenseVector(1.5, 0.3, -0.7)
    val y = X * weights
    val data = DenseMatrix.horzcat(X, y.asDenseMatrix.t)

    val df = data(*, ::).iterator
      .map(x => (x(0), x(1), x(2), x(3)))
      .toSeq.toDF("x1", "x2", "x3", "y")

    val model = new LinRegressionEstimator("1").fit(df)

    model.transform(df)
    println(model.getWeights, model.getBias)
  }
}


