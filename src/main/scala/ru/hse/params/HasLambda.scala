package ru.hse.params

import org.apache.spark.ml.param.{Param, Params}

trait HasLambda extends Params {

  final val lambda: Param[Double] = new Param[Double](this, "lambda", "")

  setDefault(lambda, 1e-2)

  final def getLambda: Double = $(lambda)
}