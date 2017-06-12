import org.jfree.chart.plot.ValueMarker
import breeze.stats.mean
import breeze.linalg._
import breeze.numerics.{pow, sqrt}
import org.jfree.chart.plot.ValueMarker
import org.jfree.chart.axis._
import scala.math
import breeze.plot._
import cern.jet.random.tdouble.Gamma  // add jar dependency for colt and parallelcolt
import cern.jet.random.tdouble.engine.DoubleMersenneTwister
import java.util.Date

/**
  * Created by Antonia Kontaratou
  * Simulates data for intercept, b1, b2 and y (n rows)
  * Gibbs sampler to estimate intercept, b1, b2 and y.
  * Beta coefficients are updated together.
  * Plots of the results: trace plots, histograms of the density-marginal distribution, autocorrelation plots.
  */
object GibbsSamplerSim {
  class State(var betaCoefs: DenseVector[Double] ,var tau: Double)
  val n = 100
  val a = 1
  val b = 0.01
  val lambda = 0.001 //K=λI, λ is how uncertain we are about μ. Here we chose a small one that indicates that we are a quite uncertain and it does not affect much the variance where it is used.
  val (simY, simX) = dataSimulation(n) //simDataY=y and x: 1,x1,x2
  val dI = diag(DenseVector.ones[Double](simX.cols))
  val K = lambda*dI
  val rngEngine = new DoubleMersenneTwister(new Date)
  val rngTau = new Gamma(1.0,0.01,rngEngine)


  def nextIter(st: State):State={ //it basically creates the next state

    def multivariateGaussian(m: DenseVector[Double], v: DenseMatrix[Double]): DenseVector[Double]={
      val root: DenseMatrix[Double] = cholesky(v)
      val z: DenseVector[Double] = DenseVector.rand(m.length, breeze.stats.distributions.Gaussian(0, 1))
      val sample = { root * z += m }
      sample
    }

    val KtauXtXinv= inv(K + st.tau * simX.t * simX)
    // val normal01 = breeze.stats.distributions.Gaussian(KtauXtXinv*(tau*Xt*y), KtauXtXinv) Doesn't work when specifying the variables.

    //**** MULTIVARIATE GAUSSIAN (to get the beta coefs, "uncomment" the following line) ****
    val newBetaCoefs = multivariateGaussian(KtauXtXinv * (st.tau * simX.t * simY), KtauXtXinv)

    // ------ OR ------
    //  ***** SVD (to estimate the beta coefficients) ****
//    val svdM = svd.reduced(simX)
//    val meanSVD=svdM.Vt.t * diag(svdM.S:/((pow(svdM.S,2)+lambda* DenseVector.ones[Double](3))))*svdM.U.t*simY
//    val newBetaCoefs = multivariateGaussian(meanSVD, KtauXtXinv)


    val yxb = simY - simX * newBetaCoefs
    val newtau = rngTau.nextDouble( a + simX.rows / 2, b + (yxb.t * yxb) / 2)

    new State(newBetaCoefs,newtau)
    st.betaCoefs = newBetaCoefs
    st.tau = newtau
    st
  }

  @annotation.tailrec
  def nextThinnedIter(s: State,left: Int): State = {
    if (left==0) new State(s.betaCoefs, s.tau)  //if the thin=0, we need to return the state.
    else nextThinnedIter(nextIter(s),left-1)  //Until thin gets 0 to return the state we need to run the function again and minimize the thin by 1, whereas the state should be produced by the nextIter.
  }

  def genIters(s: State,current: Int,stop: Int,thin: Int): DenseMatrix[Double] = {
    val matrix= new DenseMatrix[Double](stop,4) //it can be val bcs we change the values not the pointer.
    var state = s
    (0 until stop).foreach{i =>
      matrix(i,::) := DenseVector(state.betaCoefs(0), state.betaCoefs(1), state.betaCoefs(2), state.tau).t
      state = nextThinnedIter(s,thin)
    }
    matrix
  }


  //Simulate the x1 and x2 covariates and the residuals and save them in a denseMatrix.
  //Returns a tuple with the denseVector y (the simulated observed values) and the denseMatrix xx with the intercept and the values of x1 and x2
  def dataSimulation(sampleSize:Int)={
    val x1=DenseVector(breeze.stats.distributions.Gaussian(1.0,2.0).sample(sampleSize).toArray) //first covariate    /OR in 2 steps val g = breeze.stats.distributions.Gaussian(1.0,2.0)
    val x2=DenseVector(breeze.stats.distributions.Gaussian(0.0,1.0).sample(sampleSize).toArray) //second covariate
    val eps = DenseVector(breeze.stats.distributions.Gaussian(0.0,1.0).sample(sampleSize).toArray)//residual noise

    //Create the matrix xx which will be (1,x1,x2) in order to be multiplied with (1.5,2,1) and then added to eps to get the observations.
    val xx= DenseMatrix.zeros[Double](sampleSize,3)
    xx(::,0):= DenseVector.ones[Double](sampleSize)
    xx(::,1):= x1
    xx(::,2):= x2
    val beta=DenseVector(1.5,2.0,1.0)
    val y= xx*beta + eps
    (y,xx)
  }

  // Function to plot the results of the Gibbs sampler (based on Scala for data science book, p. 63)
  def plotResults(resultsMatrix:DenseMatrix[Double], labels:List[String]): Unit ={
    val time = linspace(1, resultsMatrix.rows, resultsMatrix.rows) // linspace creates a denseVector of Doubles. Otherwise time = convert(DenseVector.range(1,st.rows+1,1), Double), bcs if it is not converted to Double the plot below is not created. Error for yv implisit.
    val fig = Figure("Gibbs sampler results")
    val ncols= resultsMatrix.cols
    require(ncols==labels.size, "Number of columns in feature matrix must match length of labels.")
    fig.clear


    (0 until ncols).foreach{ irow =>
      val pTrace= fig.subplot(ncols-1, ncols,irow)
      tracePlot(pTrace)(time,resultsMatrix(::,irow), "time", labels(irow))
      val pHist= fig.subplot(ncols-1, ncols,irow+ncols)
      plotHistogram(pHist)(resultsMatrix(::,irow), labels(irow))
      val pAutocor= fig.subplot(ncols-1, ncols,irow+2*ncols)
      plotAutocorrelation(pAutocor)(resultsMatrix(::,irow), labels(irow))
    }

    // Function for the traceplots
    def tracePlot(plt: Plot)(xdata: DenseVector[Double], ydata:DenseVector[Double], xlabel:String, ylabel: String): Unit ={
      plt += plot(xdata, ydata, '-')
      plt.xlabel = xlabel
      plt.ylabel = ylabel
      plt.title = "Trace of " + ylabel
    }

    // Function for the histograms
    def plotHistogram(plt:Plot)(data: DenseVector[Double], label:String): Unit ={
      plt += hist(data, bins=20)
      plt.xlabel = label
      plt.title = "Histogram of "+label
    }

    // Function for the autocorrelation plots
    def plotAutocorrelation(plt:Plot)(data: DenseVector[Double], ylabel:String): Unit ={
      val maxLag=25
      val lag = linspace(0, maxLag, maxLag)
      val dataSize = data.length

      def estimateACF(data: DenseVector[Double], maxLag: Int): DenseVector[Double] ={
        val acfRes= new DenseVector[Double](maxLag)
        var x = new DenseVector[Double](data.length)
        x= data-mean(data)
        //acfRes(0)=sum(x(0 to dataSize-1):*x(0 to dataSize-1)) / sum(pow(x,2))
        (0 until maxLag).foreach{K=>
          val M = dataSize-K-1
          acfRes(K)= sum(x(0 to M):*x((K) to (M+K))) / sum(pow(x,2))
        }
        acfRes
      }

      val acfData= estimateACF(data, maxLag)
      plt += plot(lag, acfData, '.')
      plt.xlabel = "lag"
      plt.ylabel = ylabel
      plt.title = "Autocorrelation plot of "+ylabel
      //The correlation coefficient for the scatterplot summarizes the strength of the linear relationship between present and past values. The correlation coefficient is annotated at the top of the plot, along with the correlation that would be considered approximately significant at the 95% significance level against the null hypothesis that true correlation is zero.
      // if a series is completely random, then, for large sample size sample size N, the lagged-correlation coefficient is approximately normally distributed with mean 0 and variance 1/N. The probability is thus roughly ninety-five percent that the correlation falls within two standard deviations, or 2.0/sqrt(N
      val upconf= 1.96/ sqrt(dataSize) // For a 95%-confidence interval, the critical value is  and the confidence interval is sqrt(2)~1.96
      val lconf= -1.96/ sqrt(dataSize)
      plt.plot.addRangeMarker(new ValueMarker(upconf))
      plt.plot.addRangeMarker(new ValueMarker(lconf))
      plt.plot.getRangeAxis.setUpperBound(1.0)
      plt.plot.getRangeAxis.setLowerBound(-1.0)

    }

  }

  def time[A](f: => A) = {
    val s = System.nanoTime
    val ret = f
    println("time: " +(System.nanoTime-s)/1e6 + "ms" )
    ret
  }

  def main(args: Array[String]) {
    val st=time(genIters(new State(DenseVector(0.0, 0.0, 0.0), 0.0), 1, 1000, 10))
    //print(st)

    println("Estimated coefficients " + mean(st(::,*)))
    plotResults(st(100 until st.rows,::), List("Intercept","b1", "b2", "Variance"))

  }
}