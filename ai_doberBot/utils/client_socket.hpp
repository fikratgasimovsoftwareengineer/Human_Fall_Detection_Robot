#ifndef FACE_RECOGNITION_CLIENT_H
#define FACE_RECOGNITION_CLIENT_H
/**
    C++ client
*/
#include<iostream>    //cout
#include <sstream>
#include<stdio.h> //printf
#include<string.h>    //strlen
#include<string>  //string
#include<sys/socket.h>    //socket
#include<arpa/inet.h> //inet_addr
#include<netdb.h> //hostent
#include "json.hpp"

using json = nlohmann::json;

using namespace std;

/**
    TCP Client class
*/
class tcp_client  {

private:

    int sock;
    std::string address;
    int port;
    struct sockaddr_in server;

public:

    tcp_client();
    bool conn(string, int);
    bool send_data(string data);
    string receive(int);
};

#endif //FACE_RECOGNITION_CLIENT_H
