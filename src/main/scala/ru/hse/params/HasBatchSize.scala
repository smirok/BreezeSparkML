package ru.hse.params

import org.apache.spark.ml.param.{Param, Params}

trait HasBatchSize extends Params {

  final val batchSize: Param[Int] = new Param[Int](this, "batch size", "")

  setDefault(batchSize, 1)

  final def getBatchSize: Int = $(batchSize)
}
