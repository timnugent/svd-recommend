CC=g++
CXXFLAGS=-O3 --std=c++11 -Wall -Wextra -Wno-unused-variable -Wno-unused-parameter
LIBS=-lboost_thread -lboost_system
INC=-I.
THREADS=-DTHREADS
SOURCES=src/svd_collabfilt.cpp src/main.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=bin/svd_collabfilt

all: $(SOURCES) $(EXECUTABLE)
    
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(CXXFLAGS) $(INC) $(THREADS) $(OBJECTS) -o $@ $(LIBS) 
.cpp.o:
	$(CC) $(CXXFLAGS) $(INC) $(THREADS) -c $< -o $@
clean:
	rm bin/svd_collabfilt src/*.o
test:
	bin/svd_collabfilt -o data/predictions.tsv -i data/training.tsv


