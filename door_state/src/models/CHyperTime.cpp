#include "CHyperTime.h"

using namespace std;

CHyperTime::CHyperTime(int id)
{
	type = TT_HYPER;
	delete modelPositive;
	delete modelNegative;
//	modelPositive.release();
//	modelNegative.release();
	spaceDimension = 1;
	timeDimension = 0;
	maxTimeDimension = 10;
	covarianceType = EM::COV_MAT_GENERIC;
	positives = negatives = 0;
	corrective = 1.0;
}

void CHyperTime::init(int iMaxPeriod,int elements,int numClasses)
{
	maxPeriod = iMaxPeriod;
	numElements = elements;
}

CHyperTime::~CHyperTime()
{
}

// adds new state observations at given times
int CHyperTime::add(uint32_t time,float state)
{
	sampleArray[numSamples].t = time;
	sampleArray[numSamples].v = state;
	numSamples++;
	return 0; 
}

/*required in incremental version*/
void CHyperTime::update(int modelOrder,unsigned int* times,float* signal,int length)
{
	int numTraining = numSamples;
	int numEvaluation = 0;
	if (order != modelOrder){
		delete modelPositive;
		delete modelNegative;
//		modelPositive.release();
//	        modelNegative.release();
		order = modelOrder;
	}
	if (modelPositive == NULL) modelPositive = EM::create();
	if (modelNegative == NULL) modelNegative = EM::create();
	modelPositive->setClustersNumber(order);
	modelNegative->setClustersNumber(order);
	modelPositive->setCovarianceMatrixType(covarianceType);
	modelNegative->setCovarianceMatrixType(covarianceType);

	/*separate positives and negative examples*/
	Mat samplesPositive(positives,spaceDimension+timeDimension,CV_32FC1);
	Mat samplesNegative(negatives,spaceDimension+timeDimension,CV_32FC1);
	
	float vDummy = 0.5;
	long int tDummy = 0.5;
	for (int i = 0;i<numTraining;i++){
		vDummy = sampleArray[i].v;
		if (vDummy > 0.5){
			samplesPositive.push_back(vDummy);
			positives++;
		} else{
			samplesNegative.push_back(vDummy);
			negatives++;
		}
	}
	periods.clear();
	bool stop = false;
	do { 
		/*find the gaussian mixtures*/
		if (positives <= order || negatives <= order)break;
		modelPositive->trainEM(samplesPositive);
		modelNegative->trainEM(samplesNegative);
		if (modelNegative->isTrained()) printf(" Positive model\n");
		if (modelPositive->isTrained()) printf(" Negative model\n");
		printf("Model trained with %i clusters, %i dimensions, %i positives and %i negatives\n",order,timeDimension,positives,negatives);
		print();
		/*analyse model error for periodicities*/
		CFrelement fremen(0);
		float err = 0;
		float sumErr = 0;
		fremen.init(maxPeriod,maxTimeDimension,1);

		/*calculate model error across time*/
		for (int i = 0;i<numTraining;i++)
		{
			fremen.add(sampleArray[i].t,estimate(sampleArray[i].t)-sampleArray[i].v);
		}

		/*determine model weights*/
		float integral = 0;
		numEvaluation = numSamples;
		for (int i = 0;i<numEvaluation;i++) integral+=estimate(sampleArray[i].t);
		corrective = corrective*positives/integral;
		
		/*calculate evaluation error*/
		for (int i = 0;i<numEvaluation;i++)
		{
			err = estimate(sampleArray[i].t)-sampleArray[i].v;
			sumErr+=err*err;
		}
		sumErr=sqrt(sumErr/numEvaluation);

		/*retrieve dominant error period*/	
		int maxOrder = 1;
		fremen.update(timeDimension/2+1);
		int period = fremen.getPredictFrelements()[0].period;
		bool expand = true;
		fremen.print(true);
		printf("Model error with %i time dimensions and %i clusters is %.3f\n",timeDimension,order,sumErr);

		/*if the period already exists, then skip it*/
		for (int d = 0;d<timeDimension/2;d++)
		{
			if (period == periods[d]) period = fremen.getPredictFrelements()[d+1].period;
		}
		errors[timeDimension/2] = sumErr;
		//cout << samplesPositive.rowRange(0, 1) << endl;

		/*error has increased: cleanup and stop*/
		if (timeDimension > 1 && errors[timeDimension/2-1] <  sumErr)
		{
			printf("Error increased from %.3f to %.3f\n",errors[timeDimension/2-1],errors[timeDimension/2]);
			timeDimension-=2;
			load("model");
			samplesPositive = samplesPositive.colRange(0, samplesPositive.cols-2);
			samplesNegative = samplesNegative.colRange(0, samplesNegative.cols-2);
			if (order < maxOrder){
				delete modelPositive;
				delete modelNegative;
//				modelPositive->release();
//				modelNegative->release();
				if (modelPositive == NULL) modelPositive = EM::create();
				if (modelNegative == NULL) modelNegative = EM::create();
				modelPositive->setClustersNumber(order);
				modelNegative->setClustersNumber(order);
				modelPositive->setCovarianceMatrixType(covarianceType);
				modelNegative->setCovarianceMatrixType(covarianceType);
			}
			printf("Reducing hypertime dimension to %i: ",timeDimension);
			for (int i = 0;i<timeDimension/2;i++) printf(" %i,",periods[i]);
			printf("\n");
			stop = true;
		}else{
			save("model");
		}
		if (timeDimension >= maxTimeDimension) stop = true;

		/*hypertime expansion*/
		if (stop == false && expand == true){
			printf("Adding period %i \n",period);
			Mat hypertimePositive(positives,2,CV_32FC1);
			Mat hypertimeNegative(negatives,2,CV_32FC1);
			positives = negatives = 0;
			for (int i = 0;i<numTraining;i++)
			{
				vDummy = sampleArray[i].v;
				tDummy = sampleArray[i].t;
				if (vDummy > 0.5){
					hypertimePositive.at<float>(positives,0)=cos((float)tDummy/period*2*M_PI);
					hypertimePositive.at<float>(positives,1)=sin((float)tDummy/period*2*M_PI);
					positives++;
				}else{
					hypertimeNegative.at<float>(negatives,0)=cos((float)tDummy/period*2*M_PI);
					hypertimeNegative.at<float>(negatives,1)=sin((float)tDummy/period*2*M_PI);
					negatives++;
				}
			}
			hconcat(samplesPositive, hypertimePositive,samplesPositive);
			hconcat(samplesNegative, hypertimeNegative,samplesNegative);
			periods.push_back(period);
			timeDimension+=2;
		}
		if (order <  maxOrder) stop = false;
	}while (stop == false);
}

float CHyperTime::estimate(uint32_t t)
{
	/*is the model valid?*/
	if (modelNegative->isTrained() && modelPositive->isTrained()){
		Mat sample(1,spaceDimension+timeDimension,CV_32FC1);
		sample.at<float>(0,0)=1;

		/*augment data sample with hypertime dimensions)*/
		for (int i = 0;i<timeDimension/2;i++){
			sample.at<float>(0,spaceDimension+2*i+0)=cos((float)t/periods[i]*2*M_PI);
			sample.at<float>(0,spaceDimension+2*i+1)=sin((float)t/periods[i]*2*M_PI);
		}
		Mat probs(1,2,CV_32FC1);
		Vec2f a = modelPositive->predict2(sample, probs);

		sample.at<float>(0,0)=0;
		Vec2f b = modelNegative->predict2(sample, probs);

		/*cout << sample << endl;
		cout << a << endl;
		cout << b << endl;*/

		double d = ((positives*exp(a(0))+negatives*exp(b(0))));
		//d = ((exp(a(0))+exp(b(0))));
		//if(d > 0) return exp(a(0))/d;
		if(d > 0) return corrective*positives*exp(a(0))/d;
//		double d = ((exp(a(0))+exp(b(0))));
//		if(d > 0) return exp(a(0))/d;
	}else{
		printf("Model estimation skipped\n");
	}
	/*any data available?*/
	if (negatives+positives > 0) return (float)positives/(positives+negatives);
	return 0.5;
}

void CHyperTime::print(bool all)
{
	Mat meansPositive = modelPositive->getMeans();
	Mat meansNegative = modelNegative->getMeans();
	std::cout << meansPositive << std::endl;
	std::cout << meansNegative << std::endl;
	//std::cout << periods << std::endl;	
	printf("%i %i\n",order,timeDimension);	
}

float CHyperTime::predict(uint32_t time)
{
	return estimate(time);	
}

int CHyperTime::save(const char* name,bool lossy)
{
	FileStorage fsp(name, FileStorage::WRITE);
	fsp << "periods" << periods;
	fsp << "order" << order;
	fsp << "positives" << positives;
	fsp << "negatives" << negatives;
	fsp << "corrective" << corrective;
	cvStartWriteStruct(*fsp, "ModelPositive", CV_NODE_MAP);
	if (modelPositive->isTrained()){
		modelPositive->write(fsp);
		printf("saving positive\n");
	}
	cvEndWriteStruct(*fsp);
	cvStartWriteStruct(*fsp, "ModelNegative", CV_NODE_MAP);
	if (modelNegative->isTrained()){
		modelNegative->write(fsp);
		printf("saving negative\n");
	}
	cvEndWriteStruct(*fsp);
	fsp.release();

	return 0;
}

int CHyperTime::load(const char* name)
{
	FileStorage fs(name, FileStorage::READ);
	fs["periods"] >> periods;
	fs["order"] >> order;
	fs["positives"] >> positives;
	fs["negatives"] >> negatives;
	fs["corrective"] >> corrective;

	modelPositive = EM::create();
	modelNegative = EM::create();
	modelPositive->setClustersNumber(order);
	modelNegative->setClustersNumber(order);
	modelPositive->setCovarianceMatrixType(covarianceType);
	modelNegative->setCovarianceMatrixType(covarianceType);

	modelPositive->read(fs["ModelPositive"]);
	modelNegative->read(fs["ModelNegative"]);
	fs.release();

	timeDimension = periods.size()*2;

	return 0;
}

