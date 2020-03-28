all: alexnet

alexnet:
	g++ -mavx alexnet.cpp util.cpp -o alexnet

clean:
	rm -f alexnet
