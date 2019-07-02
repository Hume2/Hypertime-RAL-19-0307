#include "CHyperTime.h"

using namespace std;

CHyperTime::CHyperTime(int id)
{
	type = TT_HYPER;
	spaceDimension = 1;
	timeDimension = 0;
	maxTimeDimension = 10;
	covarianceType = EM::COV_MAT_GENERIC;
	numSamples = 0;
	corrective = 1.0;
	integral = 0;
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
	integral +=state;
	numSamples++;
	return 0;
}

/*required in incremental version*/
void CHyperTime::update(int modelOrder,unsigned int* times,float* signal,int length)
{
	int numEvaluation = 0;
	if (order != modelOrder){
		delete hyperModel;
		order = modelOrder;
	}
	hyperModel = EM::create();
	hyperModel->setClustersNumber(order);
	hyperModel->setCovarianceMatrixType(covarianceType);

	//if (hyperModel == NULL) hyperModel = new EM(order,covarianceType,TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, EM::DEFAULT_MAX_ITERS, FLT_EPSILON));

	Mat samples(0,spaceDimension+timeDimension,CV_32FC1);

	float vDummy = 0.5;
	long int tDummy = 0.5;
	for (int i = 0;i<numSamples;i++){
		vDummy = sampleArray[i].v;
		samples.push_back(vDummy);
	}
	periods.clear();
	bool stop = false;
	do {
		/*find the gaussian mixtures*/
		if (numSamples <= order) break;
		hyperModel->trainEM(samples);
		Mat means = hyperModel->getMeans();
		cout << means << endl;
		printf("Model trained with %i clusters, %i dimensions, %i data\n",hyperModel->getClustersNumber(),timeDimension,numSamples);

		/*analyse model error for periodicities*/
		CFrelement fremen(0);
		float err = 0;
		float sumErr = 0;
		fremen.init(maxPeriod,maxTimeDimension,1);

		/*calculate model error across time*/
		for (int i = 0;i<numSamples;i++)
		{
			fremen.add(sampleArray[i].t,estimate(sampleArray[i].t)-sampleArray[i].v);
		}

		/*determine model weights*/
		float integralMod = 0;
		numEvaluation = numSamples;
		for (int i = 0;i<numEvaluation;i++) integralMod+=estimate(sampleArray[i].t);
		corrective = corrective*integral/integralMod;

		/*calculate evaluation error*/
		for (int i = 0;i<numEvaluation;i++)
		{
			err = estimate(sampleArray[i].t)-sampleArray[i].v;
			sumErr+=err*err;
		}
		sumErr=sqrt(sumErr/numEvaluation);

		/*retrieve dominant error period*/
		int maxOrder = 1;
		fremen.update(timeDimension/2+1);//+10);
		int period = fremen.predictFrelements[0].period;
		bool expand = true;
		fremen.print(true);
		printf("Model error with %i time dimensions and %i clusters is %.3f\n",timeDimension,order,sumErr);

		/*if the period already exists, then skip it*/
		for (int d = 0;d<timeDimension/2;d++)
		{
			if (period == periods[d] || period < 3*3600){
				period = fremen.predictFrelements[d+1].period;
			}
		}
		errors[timeDimension/2] = sumErr;
		/*error has increased: cleanup and stop*/
		if (timeDimension > 1 && errors[timeDimension/2-1] <  sumErr)
		{
			printf("Error increased from %.3f to %.3f\n",errors[timeDimension/2-1],errors[timeDimension/2]);
			timeDimension-=2;
			load("model");
			samples = samples.colRange(0, samples.cols-2);
			if (order < maxOrder){
				delete hyperModel;
				//hyperModel = new EM(order,covarianceType,TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, EM::DEFAULT_MAX_ITERS, FLT_EPSILON));
				hyperModel = EM::create();
				hyperModel->setClustersNumber(order);
				hyperModel->setCovarianceMatrixType(covarianceType);
			}
			printf("Reducing hypertime dimension to %i: ",timeDimension);
			//for (int i = 0;i<timeDimension/2;i++) printf(" %i,",periods[i]);
			printf("\n");
			stop = true;
		}else{
			save("model");
		}
		if (period < 3*3600 )stop = true;
		if (timeDimension >= maxTimeDimension) stop = true;

		/*hypertime expansion*/
		if (stop == false && expand == true){
			printf("Adding period %i \n",period);
			Mat hypertimeSamples(numSamples,2,CV_32FC1);
			for (int i = 0;i<numSamples;i++)
			{
				vDummy = sampleArray[i].v;
				tDummy = sampleArray[i].t;
				hypertimeSamples.at<float>(i,0)=cos((float)tDummy/period*2*M_PI);
				hypertimeSamples.at<float>(i,1)=sin((float)tDummy/period*2*M_PI);
			}
			hconcat(samples, hypertimeSamples,samples);
			periods.push_back(period);
			timeDimension+=2;
		}
		if (order <  maxOrder) stop = false;
	}while (stop == false);
}

float CHyperTime::estimate(uint32_t t)
{
	float maxVal = 0;
	float maxArg = integral/numSamples;
	float meanSum = 0;
	float corrSum = 0;

	/*is the model valid?*/
	if (hyperModel->isTrained()){
		Mat sample(1,spaceDimension+timeDimension,CV_32FC1);

		/*determine the likelihood distribution of the sample having a given value at time t*/
		for (float estim = 0;estim < 0.15;estim+=0.001){
			/*form the sample*/
			sample.at<float>(0,0)=estim;

			/*augment data sample with hypertime dimensions)*/
			for (int i = 0;i<timeDimension/2;i++){
				sample.at<float>(0,spaceDimension+2*i+0)=cos((float)t/periods[i]*2*M_PI);
				sample.at<float>(0,spaceDimension+2*i+1)=sin((float)t/periods[i]*2*M_PI);
			}
			Mat probs(1,2,CV_32FC1);
			Vec2f a = hyperModel->predict2(sample, probs);

			/*determine the mean value of the distribution*/
			meanSum += estim*(exp(a(0)));
			corrSum += (exp(a(0)));

			/*determine the distribution modus*/
			if (maxVal < (exp(a(0))))
			{
				maxVal = exp(a(0));
				maxArg = estim;
			}
		}

		/*the question is if to use modus, mean or smth else*/
		return maxArg;					//returns the mode
		return corrective*meanSum/corrSum;		//returns the mean
	}
	/*any data available?*/
	return 0.5;
}

void CHyperTime::print(bool all)
{
	/*Mat meansPositive = hyperModel->get<Mat>("means");
	std::cout << meansPositive << std::endl;
	std::cout << meansNegative << std::endl;
	//std::cout << periods << std::endl;
	printf("%i %i\n",order,timeDimension);	*/
}

float CHyperTime::predict(uint32_t time)
{
	return estimate(time);
}

int CHyperTime::save(const char* name,bool lossy)
{
	/*EM models have to be saved separately and adding a '.' to the filename causes the saving to fail*/
	char filename[strlen(name)+5];

	sprintf(filename,"%s",name);
	FileStorage fsp(filename, FileStorage::WRITE);
	fsp << "periods" << periods;
	fsp << "order" << order;
	fsp << "numSamples" << numSamples;
	fsp << "corrective" << corrective;
	cvStartWriteStruct(*fsp, "StatModel", CV_NODE_MAP);
	if (hyperModel->isTrained()) hyperModel->write(fsp);
	cvEndWriteStruct(*fsp);
	fsp.release();

	return 0;
}

int CHyperTime::load(const char* name)
{
	/*EM models have to be saved separately and adding a '.' to the filename causes the saving to fail*/
	char filename[strlen(name)+5];

	sprintf(filename,"%s",name);
	FileStorage fs(filename, FileStorage::READ);
	fs["periods"] >> periods;
	fs["order"] >> order;
	fs["numSamples"] >> numSamples;
	fs["corrective"] >> corrective;

	FileNode fn = fs["StatModel"];
	timeDimension = periods.size()*2;
	if (hyperModel.empty()) {
		hyperModel = EM::create();
	}
	hyperModel->read(fn);
	fs.release();

	return 0;
}

/*this is very DIRTY, but I don't see any other way*/
int CHyperTime::exportToArray(double* array,int maxLen)
{
	save("hypertime.tmp");
	array[0] = TT_HYPER;
	if (hyperModel->isTrained()){
		FILE*  file = fopen("hypertime.tmpneg","r");
		int len = fread(&array[5],1,maxLen,file);
		fclose(file);
		array[1] = len;
		file = fopen("hypertime.tmppos","r");
		len = fread(&array[len+5],1,maxLen,file);
		array[2] = len;
		fclose(file);
		return array[1]+array[2]+5;
	}else{
		array[1] = 0;
		array[2] = numSamples;
		array[3] = 0;
		array[4] = order;
		return 5;
	}
}

/*this is very DIRTY, but I don't see any other way*/
int CHyperTime::importFromArray(double* array,int len)
{
	if (array[1] > 0){
		FILE*  file = fopen("hypertime.tmpneg","w");
		fwrite(&array[5],1,array[1],file);
		fclose(file);
		file = fopen("hypertime.tmppos","w");
		fwrite(&array[(int)array[1]+5],1,array[2],file);
		fclose(file);
		load("hypertime.tmp");
	}else{
		periods.clear();
		numSamples = array[2];
		order = array[4];
		hyperModel = EM::create();
		hyperModel->setClustersNumber(order);
		hyperModel->setCovarianceMatrixType(covarianceType);
		//if (hyperModel == NULL) hyperModel = new EM(order,covarianceType,TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, EM::DEFAULT_MAX_ITERS, FLT_EPSILON));
	}
	return 0;
}
