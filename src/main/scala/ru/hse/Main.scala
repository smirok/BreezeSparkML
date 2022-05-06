package ru.hse

import breeze.linalg.{*, DenseMatrix, DenseVector, sum}
import breeze.linalg._
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.VectorAssembler

object Main {
  private val N = 10
  private val d = 3
  private val lambda = 1e-5

  def main(args: Array[String]): Unit = {
    val sparkSession = SparkSession.builder.master("local[1]").appName("BreezeSparkML").getOrCreate()
    import sparkSession.implicits._

    val X = DenseMatrix.rand[Double](N, d)
    val y = X * DenseVector[Double](1.5, 0.3, -0.7)

    var w = DenseVector.rand[Double](d)
    var b = DenseVector.rand[Double](1)

    for (_ <- 0 to 10) {
      val eps = X * w + DenseVector.fill(N) { b(0) } - y
      w = w - lambda / N * (eps.t * X).t
      b = b - lambda / N * sum(eps)
    }
  }
}


