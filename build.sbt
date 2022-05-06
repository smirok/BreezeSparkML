import Dependencies._

version := "0.1.0-SNAPSHOT"

scalaVersion := "2.13.8"

lazy val root = (project in file("."))
  .settings(
    name := "BreezeSparkML"
  )

val breezeDependencies = Seq(
    breeze,
    breezeViz
)

val sparkDependencies = Seq(
    sparkCore,
    sparkMLlib
)

val testDependencies = Seq(
    scalaTest % Test
)

libraryDependencies ++= breezeDependencies ++ sparkDependencies ++ testDependencies