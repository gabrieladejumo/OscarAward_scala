/**
  * Created by champion on 5/15/2017.
  */

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, Bucketizer, VectorAssembler}
import org.apache.spark.ml.classification.{DecisionTreeClassifier, LogisticRegression, OneVsRest, RandomForestClassifier}
import org.apache.spark.sql.SparkSession

object oscaraward {
  def getCurrenctDirectory = new java.io.File(".").getCanonicalPath

  ///UDF for the date
  def udfcleanDate(s: String): String = {
    var cleanedDate = ""
    val dateArray: Array[String] = s.split("-")
    try {
      var yr = dateArray(2).toInt
      if (yr < 100) {
        yr = yr + 1900
      }
      cleanedDate = "%02d-%s-%04d".format(dateArray(0).toInt, dateArray(1), yr)
    } catch {
      case e: Exception => None
    }
    cleanedDate
  }

  //udfcleanDate(s:String): String


  //UDF for the new york formatting
  def udfcleanBirthplace(s: String): String = {
    var cleanedBirthplace = ""
    var strArray: Array[String] = s.split(" ")
    if (s == "New York City")
      strArray = strArray ++ Array("USA")
    else if (strArray(strArray.length - 1).length == 2)
      strArray = strArray ++ Array("USA")
    cleanedBirthplace = strArray.mkString(" ")
    cleanedBirthplace
  }


  def main(args: Array[String]) {


    val spark = SparkSession.builder()
      .master("local[*]")
      .appName("Impact of Race on Oscar Award")
      .config("spark.logConf", "true")
      .config("spark.LogLevel", "ERROR")
      .getOrCreate()

    val filepath = "hdfs:///hadoop/datasets/"
    val oscardata = spark.read.option("header", "true")
      .option("inferSchema", "true")
      .csv(filepath + "Oscars-demographics-DFE.csv")

    oscardata.show()

    //Select the columns of that is needed for our research

    val oscarawards = oscardata.select("birthplace", "date_of_birth", "race_ethnicity",
      "year_of_award", "award").toDF("birthplace", "dob", "race", "award_year", "award") // Rename the DF columns
    oscarawards.show()

    oscarawards.createOrReplaceTempView("oscarawards")

    oscarawards.select("award").distinct().show(20, false)
    spark.sql("SELECT distinct award from oscarawards").show(10, false)

    //Check DOB quality. Note that length varies based on month name
    spark.sql("SELECT distinct(length(dob)) from oscarawards").show(10, false)

    spark.sql("SELECT dob FROM oscarawards WHERE length(dob) in (4,9,10)").show()

    //Register the UDF for to be able to use it in the Select String
    spark.udf.register("udfcleanDate", udfcleanDate(_: String))
    spark.udf.register("udfcleanBirthplace", udfcleanBirthplace(_: String))

    var clean_df = spark.sql(s"SELECT " +
      //s"birthplace, " +
      s"udfcleanBirthplace(birthplace) birthplace, " +
     // s"dob  " +
      s"udfcleanDate(dob) dob, " +
      s"substring_index(udfcleanBirthplace(birthplace), ' ',-1) country, " +
      s"(award_year - substring_index(udfcleanDate(dob), '-', -1)) age ," +
      s"race, "+
      s"award " +
      s"from oscarawards")
     clean_df.show()
     //You can drop some null rows
    var clean_data = clean_df.na.drop()
    println("Total number of recs is : " + clean_df.count() )
    println("Total number of recs left is : " + clean_data.count())

    clean_data.groupBy("award","country").count().sort("country","award","count").show(20, false)

    //This will re-register the table
    clean_data.createOrReplaceTempView("oscarawards")

    spark.sql("SELECT count(distinct country) country_count, count(distinct race) race, count(distinct award) award_count from oscarawards").show()

    //Building the pipeLine

    val raceIdxer = new StringIndexer().setInputCol("race").setOutputCol("raceIdx")
    val awardIdxer = new StringIndexer().setInputCol("award").setOutputCol("awardIdx")
    val countryIdxer = new StringIndexer().setInputCol("country").setOutputCol("countryIdx")
    val splits = Array(Double.NegativeInfinity, 35.0,45.0,55.0,Double.PositiveInfinity)
    val bucketizer = new Bucketizer().setSplits(splits).setInputCol("age").setOutputCol("age_buckets")
    val assembler = new VectorAssembler().setInputCols(Array("raceIdx","age_buckets","countryIdx")).setOutputCol("features")

   //Building the pipeLine Stages.
    val df_pipeline = new Pipeline().setStages(Array(raceIdxer,awardIdxer,countryIdxer,bucketizer,assembler))
    //val model = new DecisionTreeClassifier().setLabelCol("awardIdxer").setFeaturesCol("features").fit(trainingData)

    //Using the dataframe with dropped cleaned data, we can fit the pipeline
      clean_data = df_pipeline.fit(clean_data).transform(clean_data)

   // clean_data.show(50,false)

     //Now lets split the data set
     val Array(trainData,testData)= clean_data.randomSplit(Array(0.7,0.3))

     //See the final data matrix
     trainData.show(10, false)
     testData.show(10,false)
    val maxBins = 32
    val dTreemodel = new DecisionTreeClassifier()
                  .setLabelCol("awardIdx")
                  .setFeaturesCol("features")
                  .setMaxBins(34)
                   .fit(trainData)
    val dTreemodelPrediction = dTreemodel.transform(testData)
    dTreemodelPrediction.select("award","awardIdx","prediction").show()

    //
    val mismatch = dTreemodelPrediction.filter(dTreemodelPrediction("awardIdx") =!= dTreemodelPrediction("prediction")).count()
    val testcount = testData.count()
    println(mismatch)
    println(testcount)
    //Predictions match with DecisionTreeClassifier model is about 30%
    println("Percentage of  Decision Tree matched prediction is : " + ((testcount -mismatch)*100/testcount) + "%")

    //Model 2: Try the Random Forest for the model
     val RFmodel = new RandomForestClassifier()
                .setLabelCol("awardIdx")
               .setFeaturesCol("features")
               .setNumTrees(6)
                .setMaxBins(34)
               .fit(trainData)
    //Transform the model to the testdata

    val RFprediction =  RFmodel.transform(testData)
    RFprediction.select("award","awardIdx","prediction").show()

    //
    val rfmismatch = RFprediction.filter(RFprediction("awardIdx") =!= RFprediction("prediction")).count()
    val rftestcount = testData.count()
     println(rfmismatch)
    println(rftestcount)
    //Predictions match with Random Forest Classifier model is about 30%
    println("Percentage of random Forest matched prediction is : " + ((rftestcount -rfmismatch)*100/rftestcount) + "%")

    //Model 3: OneVsRest
    val baseclassifier = new LogisticRegression()
                        .setLabelCol("awardIdx")
                        .setFeaturesCol("features")
                        .setMaxIter(30)
                        .setTol(1E-6)
                        .setFitIntercept(true)
    val ovrModel = new OneVsRest()
                       .setClassifier(baseclassifier )
                      .setLabelCol("awardIdx")
                      .setFeaturesCol("features")
                      .fit(trainData)

    val ovr_predictions = ovrModel.transform(testData)

    ovr_predictions.select("award","awardIdx","prediction").show()

    //
    val ovrmismatch = ovr_predictions.filter(ovr_predictions("awardIdx") =!= ovr_predictions("prediction")).count()
    val ovrtestcount = testData.count()
    println(ovrmismatch)
    println(ovrtestcount)
    //Predictions match with Random Forest Classifier model is about 30%
    println("Percentage of OVR matched prediction is : " + ((ovrtestcount -ovrmismatch)*100/ovrtestcount) + "%")


//Now we evaluate the model now
    val f1_eval = new MulticlassClassificationEvaluator()
               .setLabelCol("awardIdx")

    val wp_eval = new MulticlassClassificationEvaluator()
                  .setMetricName("weightedPrecision")
                  .setLabelCol("awardIdx")

    val wr_eval = new MulticlassClassificationEvaluator()
                  .setMetricName("weightedRecall")
                 .setLabelCol("awardIdx")
   val f1_eval_list = List(dTreemodelPrediction,RFprediction,ovr_predictions) map(y => f1_eval.evaluate(y))
    val wp_eval_list = List(dTreemodelPrediction,RFprediction,ovr_predictions) map(y => wp_eval.evaluate(y))
    val wr_eval_list = List(dTreemodelPrediction,RFprediction,ovr_predictions) map(y => wr_eval.evaluate(y))
   println("Default Metrics: " + f1_eval_list)
    println("Weighted Precision: " + wp_eval_list)
    println("Weighted Recall: "  + wr_eval_list)


  }
}


