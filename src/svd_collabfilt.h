// SVD-like Matrix Factorization by Tim Nugent 2015

#ifndef SVD_H
#define SVD_H

#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <eigen/Sparse>

using namespace Eigen;
using namespace std;

class SVD{

public:
	SVD(){init();};
	// Init. with an Eigen sparse matrix object
	explicit SVD(SparseMatrix<double>& d){init(); X = d;}
	// Init. with a tab delimited file
	explicit SVD(const char* data){init(); read_training_tsv(data);}
	// Set number of latent features
	void set_features(unsigned int i){K = i;}
	// Set min. number of epochs for each feature
	void set_min_epochs(unsigned int i){min_epochs = i;if(min_epochs < 1) min_epochs = 1;}
	// Set max. number of epochs for each feature
	void set_max_epochs(unsigned int i){max_epochs = i;}
	// Set learning rate
	void set_alpha(double i){alpha = i;}
	// Set learning rate decay (alpha is initial): lr = inital/(1+ decay * epochs)
	void set_decay(double i){decay = i;}
	// Set regularization parameter
	void set_lambda(double i){lambda = i;}
	// Set init. value for P & Q matrices
	void set_init(double i){pq_init = i;}
	// Move on to next feature is error change is below threshold
	void set_threshold(double i){threshold = i;}
	// Set verbose flag
	void set_verbose(bool i){verbose = i;}
	// Switch between SGD (default) and GD
	void set_stochastic(bool i){stochastic = i;}
	// Read in tab delimited train file
	void read_training_tsv(const char* data);
	// Read in tab delimited test file
	void read_testing_tsv(const char* data);
	// Run SVD factorization, optimizes features one at a time
	void factorize();
	// Run SVD factorization, optimizes features simultaneously
	void factorize2();
	// Calculare RMSE with data that has been loaded
	double calc_rmse();
	// Write P and Q matrices to file
	void write_pq_matrices(const char*, const char*);
	// Write predictions in TSV format
	void write_top_predictions(const char*, unsigned int);
	// Read P and Q matrices from file
	void read_pq_matrices(const char*, const char*);
	// Seed the random number generator used to fill P and Q
	void set_srand(){srand((unsigned int) time(0));}
	void set_srand(unsigned int i){srand(i);}
	// Clear all matrices - X, P and Q
	void clear();
	// Round outputs to nearest integer
	void set_round(bool i){round_outputs = i;}
	// Set lower bound of prediction
	void set_minpred(double i){min_pred = i;}
	// Set upped bound of prediction
	void set_maxpred(double i){max_pred = i;}
	// Mean center data
	void center(bool i){mean_center = i;}
	// Set threads if multithreaded
	 void set_threads(unsigned int i){threads = i;}
	// Get original data matrix
	MatrixXd get_x(){return X;}
	// Get user-feature matrix P
	MatrixXd get_p(){return P;}
	// Get item-feature matrix Q
	MatrixXd get_q(){return Q;}

private:
	void init();
	double predict(Eigen::DenseBase<Eigen::Matrix<double, -1, -1> >::RowXpr, Eigen::DenseBase<Eigen::Matrix<double, -1, -1> >::ColXpr);
	void threaded_eval(unsigned int, unsigned int, unsigned int, unsigned int, double);
	void threaded_eval2(unsigned int, unsigned int, unsigned int, double);
	vector<string>& split(const string&, char, vector<string>&);
	vector<string> split(const string&, char);
	SparseMatrix<double,Eigen::RowMajor> X, Y;
	vector<pair<unsigned int,unsigned int>> nonzero;
	MatrixXd P, Q;
	unsigned int K, min_epochs, max_epochs, max_user, max_item, threads;	
	double alpha, decay, lambda, threshold, min_pred, max_pred, pq_init;
	bool verbose, stochastic, round_outputs, mean_center;
	map<unsigned int,double> means;
	map<unsigned int,unsigned int> user_map, item_map, user_map2orig, item_map2orig;
};

#endif
