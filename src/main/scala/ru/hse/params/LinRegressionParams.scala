package ru.hse.params

import breeze.linalg.DenseVector
import org.apache.spark.ml.param.shared.{HasLabelCol, HasMaxIter}
import org.apache.spark.ml.param.{DoubleArrayParam, DoubleParam, Params, StringArrayParam}

trait LinRegressionParams extends Params with HasMaxIter with HasLabelCol with HasLambda {
  final var inputCols: StringArrayParam =
    new StringArrayParam(this, "inputCols", "")

  def setInputCols(value: Array[String]): this.type = set(inputCols, value)

  def getInputCols: Array[String] = $(inputCols)

  final var weights: DoubleArrayParam =
    new DoubleArrayParam(this, "weights", "")

  def setWeights(value: DenseVector[Double]): this.type = set(weights, value.toArray)

  def getWeights: DenseVector[Double] = DenseVector($(weights))

  final var bias: DoubleParam =
    new DoubleParam(this, "bias", "")

  def setBias(value: Double): this.type = set(bias, value)

  def getBias: Double = $(bias)
}
