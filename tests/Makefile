FILE=main

all:
	mkdir -p bin && gcc -o bin/$(FILE) src/$(FILE).c -I../third_party/vita/inc -I../inc -L../lib -lraccoon -L../third_party/vita/lib -lvita -lcurl -g
run:
	./bin/$(FILE)
clean:
	rm -rf bin
