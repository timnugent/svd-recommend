// SVD-like Matrix Factorization by Tim Nugent 2015

#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <map>
#include <cmath>
#include <algorithm>
#include <eigen/Sparse>
#include "svd_collabfilt.h"

#ifdef THREADS
#include <boost/thread.hpp>  
#endif

using namespace Eigen;
using namespace std;

void SVD::init(){
	K = 2;
	min_epochs = 5;
	max_epochs = 50;
	max_user = 0;
	max_item = 0;
	alpha = 0.001;
	decay = 0.0;
	lambda = 0.02;
	threshold = 0.0001;
	pq_init = 0.01;
	verbose = true;
	stochastic = true;
	round_outputs = false;
	mean_center = false;
	min_pred = 0;
	max_pred = 0;
	threads = 1;
}

vector<string>& SVD::split(const string &s, char delim, vector<string> &elems){
    stringstream ss(s);
    string item;
    while (getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

vector<string> SVD::split(const string &s, char delim){
    vector<string> elems;
    split(s, delim, elems);
    return elems;
}

void SVD::clear(){

	X.resize(0,0);
	Y.resize(0,0);
	P.resize(0,0);
	Q.resize(0,0);
	nonzero.clear();
}

double SVD::predict(DenseBase<Eigen::Matrix<double, -1, -1> >::RowXpr a, DenseBase<Eigen::Matrix<double, -1, -1> >::ColXpr b){
	
	double pred = a.dot(b);
	if(min_pred && pred < min_pred){
		pred = min_pred;
	}
	if(max_pred && pred > max_pred){
		pred = max_pred;
	}
	return(pred);
}

void SVD::threaded_eval(unsigned int t, unsigned int start, unsigned int stop, unsigned int k, double lr){

	//cout << "In thread " << t << endl;
	for (unsigned int i = start; i < stop; i++){
		double prediction = predict(P.row(nonzero[i].first),Q.col(nonzero[i].second));
        double eij = (double)X.coeffRef(nonzero[i].first,nonzero[i].second) - prediction;
        P(nonzero[i].first,k) += lr * (2 * eij * Q(k,nonzero[i].second) - lambda * P(nonzero[i].first,k));
        Q(k,nonzero[i].second) += lr * (2 * eij * P(nonzero[i].first,k) - lambda * Q(k,nonzero[i].second)); 
        // High learning rates can produce nan/inf updates: Figure 6 - http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
        if(std::isnan(P(nonzero[i].first,k))||std::isinf(P(nonzero[i].first,k))){
        	cout << "nan or inf P update.. reduce learning rate alpha!" << endl;
        }
        if(std::isnan(Q(k,nonzero[i].second))||std::isinf(Q(k,nonzero[i].second))){
        	cout << "nan or inf Q update.. reduce learning rate alpha!" << endl;
        }
	} 	
}

void SVD::threaded_eval2(unsigned int t, unsigned int start, unsigned int stop, double lr){

	//cout << "In thread " << t << endl;
	for (unsigned int i = start; i < stop; i++){
        double eij = (double)X.coeffRef(nonzero[i].first,nonzero[i].second) - P.row(nonzero[i].first).dot(Q.col(nonzero[i].second));
        for(unsigned int k = 0; k < K; k++){
            P(nonzero[i].first,k) += lr * (2 * eij * Q(k,nonzero[i].second) - lambda * P(nonzero[i].first,k));
            Q(k,nonzero[i].second) += lr * (2 * eij * P(nonzero[i].first,k) - lambda * Q(k,nonzero[i].second)); 
            // High learning rates can produce nan/inf updates: Figure 6 - http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
            if(std::isnan(P(nonzero[i].first,k)) || std::isinf(P(nonzero[i].first,k))){
            	cout << "nan or inf P update.. reduce learning rate alpha!" << endl;
            }
            if(std::isnan(Q(k,nonzero[i].second)) || std::isinf(Q(k,nonzero[i].second))){
            	cout << "nan or inf Q update.. reduce learning rate alpha!" << endl;
            }
    	}
	} 	
}

void SVD::factorize(){

	if(!max_user || !max_item){
		cout << "Data not loaded!" << endl;
	}else{
	    double e = 0.0, e_old = 0.0, lr = 0.0;
		P.setRandom(max_user,K);
		Q.setRandom(max_item,K);
		P.setConstant(pq_init);
		Q.setConstant(pq_init);
		Q.transposeInPlace();

	    if(verbose) cout << "Starting factorization (one feature at a time):" << endl;	
#ifdef THREADS
		if(threads > 1 && verbose)cout << "Using " << threads << " threads" << endl;
#endif

	    for(unsigned int k = 0; k < K; k++){
	    	for(unsigned int epoch = 0; epoch < max_epochs; epoch++){   
	    		if(stochastic) random_shuffle(nonzero.begin(),nonzero.end());
	    		if(decay){
	    			lr = alpha/(1.0+(decay*epoch));
	    		}else{
	    			lr = alpha;
	    		}
#ifdef THREADS
	    		
	    		// Ratings per thread
				double rpt = 1.0+(int)nonzero.size()/threads;
				boost::thread_group *g = new boost::thread_group();

				unsigned int first_rating_index = 0;
				for(unsigned int t = 0; t < threads; t++){
					unsigned int last_rating_index = (int)rpt*(t+1);
					if(last_rating_index > nonzero.size()){
						last_rating_index = nonzero.size();
					}	
					boost::thread *t1 = new boost::thread(boost::bind(&SVD::threaded_eval,this,t,first_rating_index,last_rating_index,k,lr));
					g->add_thread(t1);
					first_rating_index = (int)rpt*(t+1)+1;
				}	
				// Wait for threads to join
				g->join_all();
				delete g;
#else	    		
	    		for(auto it = nonzero.begin(); it != nonzero.end(); ++it){
	    			double prediction = predict(P.row(it->first),Q.col(it->second));
                    double eij = (double)X.coeffRef(it->first,it->second) - prediction;
	                P(it->first,k) += lr * (2 * eij * Q(k,it->second) - lambda * P(it->first,k));
	                Q(k,it->second) += lr * (2 * eij * P(it->first,k) - lambda * Q(k,it->second)); 
	                // High learning rates can produce nan/inf updates: Figure 6 - http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
                    if(std::isnan(P(it->first,k))||std::isinf(P(it->first,k))){
                    	cout << "nan or inf P update.. reduce learning rate alpha!" << endl;
                    }
                    if(std::isnan(Q(k,it->second))||std::isinf(Q(k,it->second))){
                    	cout << "nan or inf Q update.. reduce learning rate alpha!" << endl;
                    }

	    		}
#endif
	            e_old = e;
	            e = 0.0;
				for (int i = 0; i < X.rows(); ++i){
			    	for (SparseMatrix<double,Eigen::RowMajor>::InnerIterator it(X, i); it; ++it){
	                    e += pow((double)it.value() - P.row(it.row()).dot(Q.col(it.col())), 2) + (lambda/2.0) * (pow(P(it.row(),k),2) + pow(Q(k,it.col()),2));	
				    }
		   		}
	            e = sqrt(e/(double)X.nonZeros());            
	            double e_diff = abs(e-e_old);
	            if((e_diff < threshold && epoch >= min_epochs) || std::isnan(e_diff) || std::isinf(e_diff)){
	                break;
	            }
	            if(verbose) printf("Epoch %i : Feature %i : Learning rate %2.6f : Error %2.8f : Error diff. %2.8f\n",epoch+1,k+1,lr,e,e_diff);		
	        }
	    }
	}
}

// All factors simultaneously
void SVD::factorize2(){

	if(!max_user || !max_item){
		cout << "Data not loaded!" << endl;
	}else{
	    double e = 0.0, e_old = 0.0, lr = 0.0;
		P.setRandom(max_user,K);
		Q.setRandom(max_item,K);
		P.setConstant(pq_init);
		Q.setConstant(pq_init);
		Q.transposeInPlace();

	    if(verbose) cout << "Starting factorization (all features simultaneously):" << endl;	    
#ifdef THREADS
		if(threads > 1 && verbose)cout << "Using " << threads << " threads" << endl;
#endif

		for(unsigned int epoch = 0; epoch < max_epochs; epoch++){ 
    		if(stochastic) random_shuffle(nonzero.begin(),nonzero.end());
			if(decay){
    			lr = alpha/(1.0+(decay*epoch));
    		}else{
    			lr = alpha;
    		}

#ifdef THREADS
    		// Ratings per thread
			double rpt = 1.0+(int)nonzero.size()/threads;
			boost::thread_group *g = new boost::thread_group();

			unsigned int first_rating_index = 0;
			for(unsigned int t = 0; t < threads; t++){
				unsigned int last_rating_index = (int)rpt*(t+1);
				if(last_rating_index > nonzero.size()){
					last_rating_index = nonzero.size();
				}	
				boost::thread *t1 = new boost::thread(boost::bind(&SVD::threaded_eval2,this,t,first_rating_index,last_rating_index,lr));
				g->add_thread(t1);
				first_rating_index = (int)rpt*(t+1)+1;
			}	
			// Wait for threads to join
			g->join_all();
			delete g;
#else    		
    		for(auto it = nonzero.begin(); it != nonzero.end(); ++it){
                double eij = (double)X.coeffRef(it->first,it->second) - P.row(it->first).dot(Q.col(it->second));
                for(unsigned int k = 0; k < K; k++){
	                P(it->first,k) += lr * (2 * eij * Q(k,it->second) - lambda * P(it->first,k));
	                Q(k,it->second) += lr * (2 * eij * P(it->first,k) - lambda * Q(k,it->second)); 
	                // High learning rates can produce nan/inf updates: Figure 6 - http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
                    if(std::isnan(P(it->first,k)) || std::isinf(P(it->first,k))){
                    	cout << "nan or inf P update.. reduce learning rate alpha!" << endl;
                    }
                    if(std::isnan(Q(k,it->second)) || std::isinf(Q(k,it->second))){
                    	cout << "nan or inf Q update.. reduce learning rate alpha!" << endl;
                    }
            	}
    		}
#endif
	        e_old = e;
	        e = 0.0;
			for (int i = 0; i < X.rows(); ++i){
		    	for (SparseMatrix<double,Eigen::RowMajor>::InnerIterator it(X, i); it; ++it){
	                e += pow((double)it.value() - P.row(it.row()).dot(Q.col(it.col())), 2);
	                for(unsigned int k = 0; k < K; k++){
	                	e += (lambda/2.0) * (pow(P(it.row(),k),2) + pow(Q(k,it.col()),2));	
	                }
			    }
	   		}
	        e = sqrt(e/(double)X.nonZeros());            
	        double e_diff = abs(e-e_old);
	        if(e_diff < threshold && epoch >= min_epochs){
	            break;
	        }
	        if(verbose) printf("Epoch %i : Learning rate %2.6f : Error %2.6f : Error diff. %2.6f\n",epoch+1,lr,e,e_diff);			
	    }
    }
}

double SVD::calc_rmse(){

	double sse = 0.0;
	unsigned int missing = 0;
	for (int i = 0; i < Y.rows(); ++i){
    	for (SparseMatrix<double,Eigen::RowMajor>::InnerIterator it(Y, i); it; ++it){
    		if(it.row() < P.rows() && it.col() < Q.cols()){
	    		double pred = P.row(it.row()).dot(Q.col(it.col()));
	    		if(mean_center) pred += means[it.col()];
	    		if(round_outputs) pred = round(pred);
	    		if(min_pred && pred < min_pred) pred = min_pred;
	    		if(max_pred && pred > max_pred) pred = max_pred;
    			sse += pow((double)it.value() - pred, 2);
    		}else{
    			cout << "This user/item not in P/Q matrix - add it to the training set:" << endl;
    			cout << it.row()+1 << "\t" << it.col()+1 << "\t" << it.value() << endl;
    			missing++;
    		}	
	    }
	}
	double r = sqrt(sse/(double)(Y.nonZeros())-missing);
	cout << "RMSE: " << r << endl;
	return r;

}

void SVD::write_pq_matrices(const char* pf, const char* qf){

	//cout << "P dimensions: " << P.rows() << " x " << P.cols() << endl;
	//cout << "Q dimensions: " << Q.rows() << " x " << Q.cols() << endl;
	if(verbose) cout << "Writing user-feature matrix " << pf << " ..." << endl;
	std::ofstream filep(pf);
	if (filep.is_open()){
		filep << P << '\n';
	}	
	filep.close();
	
	if(verbose) cout << "Writing item-feature matrix " << qf << " ..." << endl;
	std::ofstream fileq(qf);
	Q.transposeInPlace();
	if (fileq.is_open()){
		fileq << Q << '\n';
	}	
	Q.transposeInPlace();
	fileq.close();

}

void SVD::write_top_predictions(const char* pf, unsigned int k = 50){

	if(verbose) cout << "Writing predictions file " << pf << " ..." << endl;
	std::setprecision(5);
	std::ofstream filep(pf);
	for(unsigned int i = 0; i < P.rows(); i++){
		multimap<double, unsigned int, std::greater<double> > results;
		for(unsigned int j = 0; j < Q.cols(); j++){
			double pred = P.row(i).dot(Q.col(j));
			if(min_pred && pred < min_pred) pred = min_pred;
			if(max_pred && pred > max_pred) pred = max_pred;
			results.insert(pair<double,unsigned int>(pred,j+1));
		}
		for(struct {unsigned int c; multimap<double, unsigned int>::iterator it;} v = {0, results.begin()}; v.it != results.end() && v.c < k; v.it++, v.c++){
			//filep << user_map2orig[i] << "|" << i << "\t" << item_map2orig[v.it->second] << "|" << v.it->second << "\t" << v.it->first << endl;
			filep << user_map2orig[i] << "\t" << item_map2orig[v.it->second] << "\t" << v.it->first << endl;
		}
	    results.clear();
	}	
	filep.close();
}

void SVD::read_pq_matrices(const char* pf, const char* qf){

	if(verbose) cout << "Reading user-feature matrix " << pf << " ..." << endl;
	vector<double> values;
	unsigned int rows = 0;
	ifstream filep(pf);
	if(filep.is_open()){
		string line;
		while(getline(filep, line)){
			istringstream iss(line);
			string buf;
			vector<string> tokens;
			while (iss >> buf){
				values.push_back(stod(buf));
			}
			rows++;
		}
	}
	filep.close();
	unsigned int cols = values.size()/rows;

	P.resize(rows,cols);
	for (unsigned int i = 0; i < rows; i++){
        for (unsigned int j = 0; j < cols; j++){
            P(i,j) = values[cols*i+j];
        }
	}
	set_features(P.cols());

	if(verbose) cout << "Reading item-feature matrix " << qf << " ..." << endl;
	values.clear();
	rows = 0;
	ifstream fileq(qf);
	if(fileq.is_open()){
		string line;
		while(getline(fileq, line)){
			istringstream iss(line);
			string buf;
			vector<string> tokens;
			while (iss >> buf){
				values.push_back(stod(buf));
			}
			rows++;
		}
	}
	fileq.close();
	cols = values.size()/rows;
	Q.resize(rows,cols);
	for (unsigned int i = 0; i < rows; i++){
        for (unsigned int j = 0; j < cols; j++){
            Q(i,j) = values[cols*i+j];
        }
	}
	Q.transposeInPlace();
	if(verbose) cout << "P dimensions: " << P.rows() << " x " << P.cols() << endl;
	if(verbose) cout << "Q dimensions: " << Q.rows() << " x " << Q.cols() << endl;

}

void SVD::read_training_tsv(const char* data){

	typedef Eigen::Triplet<double> T;
	vector<T> tripletList;
	nonzero.clear();
	map<string,unsigned int> duplicates;
	map<unsigned int,unsigned int> sums, counts;
	max_user = 0;
	max_item = 0;

	ifstream file(data);
	if(file.is_open()){
		string line;
		while(getline(file, line)){
			vector<string> tokens = split(line, '\t');
			if(tokens.size() == 3){	
				unsigned int user = stoi(tokens[0]);
				unsigned int item = stoi(tokens[1]);
				string s = tokens[2];
				s.erase(s.find_last_not_of(" \n\r\t")+1);
				double rating = atof(s.c_str());
				if(mean_center){
					sums[item-1] += rating;
					counts[item-1]++;
				}
				if(user_map.find(user) == user_map.end()){
					user_map[user] = max_user;
					user_map2orig[max_user] = user;
					max_user++;
				}
				if(item_map.find(item) == item_map.end()){
					item_map[item] = max_item;
					item_map2orig[max_item] = item;
					max_item++;
				}
				tripletList.push_back(T(user_map[user],item_map[item],rating));
				nonzero.push_back(pair<unsigned int,unsigned int>(user_map[user],item_map[item]));
				string key = to_string(user_map[user]) + "-" + to_string(item_map[item]);
				duplicates[key]++;
			}
		}
		file.close();

		if(mean_center){
			for(auto it = sums.begin(); it != sums.end(); ++it){
				means[it->first] = (double)sums[it->first]/(double)counts[it->first];
			}
		}

		X.resize(0,0);
		X.resize(max_user,max_item);
		X.setFromTriplets(tripletList.begin(), tripletList.end());

		// Take average where there's a duplicate - currently they are summed by setFromTriplets
		for (int i = 0; i < X.rows(); ++i){
		    for (SparseMatrix<double,Eigen::RowMajor>::InnerIterator it(X, i); it; ++it){
		        string key = to_string(it.row()) + "-" + to_string(it.col());		        
	        	if(duplicates[key] > 1){
		        	X.coeffRef(it.row(),it.col()) = it.value()/(double)duplicates[key];
	        	}
	        	if(mean_center){
	        		X.coeffRef(it.row(),it.col()) = X.coeffRef(it.row(),it.col())-means[it.col()];
	   			}
	   		}
	    }

		if(verbose) cout << "Read file " << data << " ("<< X.nonZeros() << " entries)"<< endl;
		if(verbose) cout <<	"Users with data: " << user_map.size() << endl;
		if(verbose) cout <<	"Items with data: " << item_map.size() << endl;	

	}else{
		if(verbose) cout << "Couldn't read file " << data << endl;
	}

}


void SVD::read_testing_tsv(const char* data){

	typedef Eigen::Triplet<double> T;
	vector<T> tripletList;
	map<string,unsigned int> duplicates;
	map<unsigned int,unsigned int> sums, counts;
	unsigned int missing = 0;

	ifstream file(data);
	if(file.is_open()){
		string line;
		while(getline(file, line)){
			vector<string> tokens = split(line, '\t');
			if(tokens.size() == 3){	
				unsigned int user = stoi(tokens[0]);
				unsigned int item = stoi(tokens[1]);
				string s = tokens[2];
				s.erase(s.find_last_not_of(" \n\r\t")+1);
				double rating = atof(s.c_str());
				if(user_map.find(user) != user_map.end() && item_map.find(item) != item_map.end()){
					tripletList.push_back(T(user_map[user],item_map[item],rating));
					string key = to_string(user_map[user]) + "-" + to_string(item_map[item]);
					duplicates[key]++;
				}else if(user_map.find(user) == user_map.end()){
						//cout << "user " << user << " was missing from the training set - skipping" << endl;
						missing++;
				}else{
						//cout << "item " << item << " was missing from the training set - skipping" << endl;
						missing++;
				}
			}
		}
		file.close();

		Y.resize(0,0);
		Y.resize(max_user,max_item);
		Y.setFromTriplets(tripletList.begin(), tripletList.end());

		// Take average where there's a duplicate - currently they are summed by setFromTriplets
		for (int i = 0; i < X.rows(); ++i){
		    for (SparseMatrix<double,Eigen::RowMajor>::InnerIterator it(Y, i); it; ++it){
		        string key = to_string(it.row()) + "-" + to_string(it.col());		        
	        	if(duplicates[key] > 1){
		        	Y.coeffRef(it.row(),it.col()) = it.value()/(double)duplicates[key];
	        	}
	   		}
	    }
		if(verbose) cout << "Read file " << data << " ("<< Y.nonZeros() << " entries)" << endl;
		if(verbose) cout << missing << " entries were skipped as they were not present in the training data" << endl;
	}else{
		if(verbose) cout << "Couldn't read file " << data << endl;
	}

}
