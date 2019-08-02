#include <string>
#include <iostream>
#include "base64.h"
#include <fstream>

using namespace std;

int main(int argc, char** argv){

    fstream f;
    f.open("../test_image/keliamoniz1.jpg", ios::in|ios::binary);

    f.seekg(0, std::ios_base::end);
    std::streampos sp = f.tellg();
    int size = sp;
    char* buffer = (char*)malloc(sizeof(char)*size);
    f.read(buffer,size);

    cout << "file size:" << size << endl;
    string imgBase64 = base64_encode(buffer, size);
    cout << "img base64 encode size:" << imgBase64.size() << endl;
    string imgdecode64 = base64_decode(imgBase64);
    cout << "img decode size:" << imgdecode64.size() << endl;
    return 0;
}