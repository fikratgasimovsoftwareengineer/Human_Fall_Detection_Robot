/*
 * Dato:30/07/2022
 * Progetto: Human Following Client
 * Svillupatore: Fikrat Gasimov
 * Occupa: AI & Robotics Scientist
 * CTO : Davide Allegri
 * */

//===============Start=====
// Include Lib
#ifndef FACE_BIOMETRICES
#define FACE_BIOMETRICES

// Include prebuilt lib
#include <iostream>
#include <stdlib.h> // C macros exit or success failure
#include <stdio.h> // fprintf c 
#include <errno.h> // socket err check c
#include <string>

#include <cstring> // string in c++
#include <sstream> // stream text

//Socket libs
//
#include <sys/socket.h> // socket
#include <netinet/in.h> // socket connection
#include <arpa/inet.h> // define ip port
#include <unistd.h>

// Namespace 
//
using namespace std;

class face_biometric {

  private:

	// Manage Buffer
        int size;
	char * buffer_recv;

	//SockFD
 //	int sockFD;
	int optval, queue_connection;
	
	// Server Socket Addr
//	struct sockaddr_in server, client_accept;
	
	//Addr Size
	//socklen_t addr_size;
  public:
	// Constructor
	face_biometric();

       // Connection
	bool sock_connection (int port);
	 
	//Send String message
 	bool send_faceName(string name);

	// Recv Acknowledgement
	bool recvConfirmation(void *sock_fd);

        // New Socket
        int sockFD, new_sockFD;

        // Server Socket Addr
        struct sockaddr_in server, client_accept;
       
	// //Addr Size 
	socklen_t addr_size;
	
	//Destroy
	~face_biometric();
	 

};

#endif FACE_BIOMETRICES
