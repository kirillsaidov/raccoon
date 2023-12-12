FILE=main

all:
	mkdir -p bin && gcc -o bin/$(FILE) src/$(FILE).c -I../third_party -I../inc -L../lib -lvita -lprisma -lcurl -g
run:
	./bin/$(FILE)
clean:
	rm -rf bin
