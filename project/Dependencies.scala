import sbt._

object Dependencies {
  lazy val scalaTest = "org.scalatest" %% "scalatest" % "3.2.12"
  lazy val sparkCore = "org.apache.spark" %% "spark-core" % "3.2.1"
  lazy val sparkMLlib = "org.apache.spark" %% "spark-mllib" % "3.2.1"
  lazy val breeze = "org.scalanlp" %% "breeze" % "2.0.1-RC1"
  lazy val breezeViz = "org.scalanlp" %% "breeze-viz" % "2.0.1-RC1"
}