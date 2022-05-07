package ru.hse.params

import org.apache.spark.ml.param.{Param, Params}

trait HasLearningRate extends Params {

  final val learningRate: Param[Double] =
    new Param[Double](this, "learning rate", "")

  setDefault(learningRate, 1e-2)

  final def getLearningRate: Double = $(learningRate)
}
