FILE=main

all:
	mkdir -p bin && gcc -std=c11 -o bin/$(FILE) src/$(FILE).c -I../third_party/vita/inc -I../inc -L../lib -lraccoon -L../third_party/vita/lib -lvita -lm -g
run:
	./bin/$(FILE)
clean:
	rm -rf bin
