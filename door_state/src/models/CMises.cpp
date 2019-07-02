#include "CMises.h"

using namespace std;

CMises::CMises(int id)
{
//	strcpy(id,name);
	positives = negatives = 0;
//	for (int i = 0;i<2;i++){
//		sample.x[i] = sample.y[i] = 0;
//		sample.px[i] = sample.py[i] = 0;
//		sample.pxx[i] = sample.pyy[i] = 1;
//		sample.xy[i] = sample.xx[i] = sample.yy[i] = 0;
//	}
	type = TT_MISES;
	modelPositive.release();
	modelNegative.release();
}

void CMises::init(int iMaxPeriod,int elements,int numActivities)
{
	numElements = elements;
}

CMises::~CMises()
{
}

// adds new state observations at given times
int CMises::add(uint32_t time,float state)
{
	if (state > 0.5){
		 positiveArray[positives].x = 1;
		 positiveArray[positives].t = time;
		positives++;
	} else {
		 negativeArray[negatives].x = 0;
		 negativeArray[negatives].t = time;
		negatives++;
	}
	return 0; 
}

/*not required in incremental version*/
void CMises::update(int modelOrder,unsigned int* times,float* signal,int length)
{
	if (order != modelOrder){
		modelPositive.release();
		modelNegative.release();

		order = modelOrder;
	}

	if (modelPositive.empty()) {
		modelPositive = EM::create();
		modelPositive->setClustersNumber(order);
		modelPositive->setCovarianceMatrixType(EM::COV_MAT_DIAGONAL);
		modelPositive->setTermCriteria(TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, EM::DEFAULT_MAX_ITERS, FLT_EPSILON));
	}

	if (modelNegative.empty()) {
		modelNegative = EM::create();
		modelNegative->setClustersNumber(order);
		modelNegative->setCovarianceMatrixType(EM::COV_MAT_DIAGONAL);
		modelNegative->setTermCriteria(TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, EM::DEFAULT_MAX_ITERS, FLT_EPSILON));
	}

	//if (modelPositive == NULL) modelPositive = new EM(order,EM::COV_MAT_DIAGONAL,TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, EM::DEFAULT_MAX_ITERS, FLT_EPSILON));
	//if (modelNegative == NULL) modelNegative = new EM(order,EM::COV_MAT_DIAGONAL,TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, EM::DEFAULT_MAX_ITERS, FLT_EPSILON));
	Mat samplesPositive(positives,4,CV_32FC1);
	Mat responsesPositive(positives,1,CV_32FC1);
	Mat samplesNegative(negatives,4,CV_32FC1);
	Mat responsesNegative(negatives,1,CV_32FC1);
	for (int i = 0;i<positives;i++){
		samplesPositive.at<float>(i,0)=cos((float)positiveArray[i].t/86400*2*M_PI);
		samplesPositive.at<float>(i,1)=sin((float)positiveArray[i].t/86400*2*M_PI);
		samplesPositive.at<float>(i,2)=cos((float)positiveArray[i].t/7/86400*2*M_PI);
		samplesPositive.at<float>(i,3)=sin((float)positiveArray[i].t/7/86400*2*M_PI);
		responsesPositive.at<float>(i,0)=positiveArray[i].x;
	}
	if (positives > 5) modelPositive->train(samplesPositive, ROW_SAMPLE, responsesPositive); //This might be wrong.
	
	for (int i = 0;i<negatives;i++){
		samplesNegative.at<float>(i,0)=cos((float)negativeArray[i].t/86400*2*M_PI);
		samplesNegative.at<float>(i,1)=sin((float)negativeArray[i].t/86400*2*M_PI);
		samplesNegative.at<float>(i,2)=cos((float)negativeArray[i].t/7/86400*2*M_PI);
		samplesNegative.at<float>(i,3)=sin((float)negativeArray[i].t/7/86400*2*M_PI);
		responsesNegative.at<float>(i,0)=negativeArray[i].x;
	}
	if (negatives > 5)modelNegative->train(samplesNegative, ROW_SAMPLE, responsesNegative); //This might be wrong.
}

float CMises::estimate(uint32_t t)
{
	if (modelNegative->isTrained() && modelPositive->isTrained()){
		Mat sample(1,5,CV_32FC1);
		sample.at<float>(0,0)=cos((float)t/86400*2*M_PI);
		sample.at<float>(0,1)=sin((float)t/86400*2*M_PI);
		sample.at<float>(0,2)=cos((float)t/7/86400*2*M_PI);
		sample.at<float>(0,3)=sin((float)t/7/86400*2*M_PI);
		sample.at<float>(0,4)=1;
		Mat probs(1,2,CV_32FC1);
		Vec2f a = modelPositive->predict2(sample, probs);
		sample.at<float>(0,4)=0;
		Vec2f b = modelNegative->predict2(sample, probs);
		//return exp(b(0));
		//printf("Positive %f\n",exp(a(0)));
		//std::cout << a << std::endl;
		//std::cout << probs << std::endl;
		double d = ((exp(a(0))+exp(b(0))));
		if(d > 0) return exp(a(0))/d;
		return 0.5;
		/*modelNegative->predict(sample, probs);
		printf("Neg\n");
		std::cout << probs << std::endl;*/
	}
	return 0.5;
}

void CMises::print(bool all)
{
}

float CMises::predict(uint32_t time)
{
	return estimate(time);	
//	return sample.value;	
}

int CMises::save(const char* name,bool lossy)
{
	FILE* file = fopen(name,"w");
	//Nothing to save
	fclose(file);
	return 0;
}

int CMises::load(const char* name)
{
	FILE* file = fopen(name,"r");
	//Nothing to load
	fclose(file);
	return 0;
}

int CMises::exportToArray(double* array,int maxLen)
{
return 0;
}

int CMises::importFromArray(double* array,int len)
{
return 0;
}
