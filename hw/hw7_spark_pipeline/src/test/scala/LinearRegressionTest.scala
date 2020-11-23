package org.apache.spark.ml.made


import breeze.linalg._
import breeze.numerics._
import breeze.linalg.{*, DenseMatrix, DenseVector}
import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator}
import com.google.common.io.Files
import org.apache.commons.math3.random.MersenneTwister
import org.apache.spark.ml
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.made.{LinearRegressionModel, WithSpark}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.{DataFrame, Row}
import org.scalatest._
import org.scalatest.flatspec._
import org.scalatest.matchers._


class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {

  val delta = 0.0001
  val maxIter = 1000
  lazy val coefficients: Vector = LinearRegressionTest._coefficients
  lazy val intercept: Double = LinearRegressionTest._intercept
  lazy val y: DenseVector[Double] = LinearRegressionTest._y
  lazy val df: DataFrame = LinearRegressionTest._final_df

  private def validateModel(model: ml.Transformer) = {
    // last column "features" replaced with predictions
    val predictions: Array[Double] = model.transform(df).collect().map((x: Row) => {
      x.getAs[Double](df.columns.length - 1)
    })

    predictions.length should be(1000)
    for (i <- predictions.indices) {
      predictions(i) should be(y.valueAt(i) +- delta)
    }
  }

  "Model" should "transform input data to prediction" in {
    val model = new LinearRegressionModel(
      coefficients = coefficients.toDense,
      intercept = intercept
    ).setInputCol("features")
      .setOutputCol("features")
      .setMaxIter(maxIter)

    validateModel(model)
  }

  "Estimator" should "calculate coefficients" in {
    val estimator = new LinearRegression()
      .setInputCol("features")
      .setOutputCol("features")
      .setMaxIter(maxIter)
      .setLabelCol("y")

    val model = estimator.fit(df)

    sum(model.coefficients.asBreeze - coefficients.asBreeze) should be(0.0 +- delta)
  }

  "Estimator" should "calculate intercept" in {
    val estimator = new LinearRegression()
      .setInputCol("features")
      .setOutputCol("features")
      .setMaxIter(maxIter)
      .setLabelCol("y")

    val model = estimator.fit(df)

    model.intercept should be(intercept +- delta)
  }

  "Estimator" should "produce functional model" in {
    val estimator = new LinearRegression()
      .setInputCol("features")
      .setOutputCol("features")
      .setMaxIter(maxIter)
      .setLabelCol("y")

    val model = estimator.fit(df)

    validateModel(model)
  }

  "Estimator" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setInputCol("features")
        .setOutputCol("features")
        .setMaxIter(maxIter)
        .setLabelCol("y")
    ))
    val tmpFolder = Files.createTempDir()

    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)

    val model = Pipeline.load(tmpFolder.getAbsolutePath).fit(df).stages(0).asInstanceOf[LinearRegressionModel]

    validateModel(model)
  }

  "Model" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setInputCol("features")
        .setOutputCol("features")
        .setMaxIter(maxIter)
        .setLabelCol("y")
    ))
    val model = pipeline.fit(df)

    val tmpFolder = Files.createTempDir()

    model.write.overwrite().save(tmpFolder.getAbsolutePath)

    val reRead = PipelineModel.load(tmpFolder.getAbsolutePath)

    validateModel(reRead)
  }
}

object LinearRegressionTest extends WithSpark{
  import spark.implicits._

  implicit val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(0)))
  private val Gausian = breeze.stats.distributions.Gaussian(0.0, 1.0)
  val _X: DenseMatrix[Double] = DenseMatrix.rand(1000, 3, Gausian)
  val _coefficients: Vector = Vectors.dense(0.5, -0.1, 0.2).toDense //DenseVector(0.5, -0.1, 0.2)
  val _intercept: Double = 1.2
  val _y: DenseVector[Double] = _X * _coefficients.asBreeze + _intercept
  val data = DenseMatrix.horzcat(_X, _y.asDenseMatrix.t)
  val _df = data(*, ::).iterator
    .map(x => (x(0), x(1), x(2), x(3)))
    .toSeq.toDF("x1", "x2", "x3", "y")

  val model = new Pipeline().setStages(Array(
    new VectorAssembler()
      .setInputCols(Array("x1", "x2", "x3"))
      .setOutputCol("features")
  )).fit(_df)
  val _final_df = model.transform(_df)
}
