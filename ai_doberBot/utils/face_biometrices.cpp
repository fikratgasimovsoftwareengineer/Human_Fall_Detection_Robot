#include "face_biometrices.hpp"
//Constructor Implementation
//

face_biometric::face_biometric(){
   
  // Initialize private members
   // Buffer_size
    size = 512;

    queue_connection = 5;
    optval = 1;

    // buffer for recv acknowledge
    buffer_recv = new char[size];

    fprintf(stdout, "Face Object Is Created!\n");
   
};

// create socket
bool face_biometric::sock_connection(int port){

   //if connection = -1, create socket
   //
    
    sockFD = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
     
    // Check if socket is not created!
    if (sockFD < 0){
  
        cerr << "Socket is not Created!" << endl;
        cout << "[ERROR]: " << strerror(errno) << endl;
        return -1; 
  
        
    }
 
    // Server Parameters
    server.sin_family = AF_INET;
    server.sin_addr.s_addr = inet_addr("192.168.123.124");
    server.sin_port = htons(port);

    // Reuse option port
    setsockopt(sockFD, SOL_SOCKET, SO_REUSEPORT, &optval, sizeof(optval));
    
    // Bind
    if (bind (sockFD, (struct sockaddr *)&server, sizeof(server)) < 0){

        cerr << "**BIND FAILED**" <<endl;
        cout << "ERROR [BIND] : " << strerror(errno) << "\n";
        
	// close socket
        close(sockFD);
        return -1;

    }else{
    
        fprintf(stdout, "Bind Successfully: %d \n", port);
    }

    // Listen on created port 
    if (listen(sockFD, queue_connection) == -1){

	cerr << "can not listen on port " << port << endl;
        cout << "[ERROR] [LISTEN] : " << strerror(errno) << endl;
	return -1;
     
    }

    else{
    
        fprintf(stdout, "Server is listening on Port Num: %d \n", port);
    
    }
    

} //Socket Finish

// Send String To Server

bool face_biometric::send_faceName(string name){

   // Send String to Client
    int send_bytes = 0;
    send_bytes = send(sockFD, name.c_str(), strlen(name.c_str()), 0);

    if (send_bytes <= 0){
        
        fprintf(stderr, "face name is not sent \n");
	return false; 
    }
    // Success !
    fprintf(stdout, "Sent is successful"); 
    
    //return True
    return true;
	

}
//Acnowledgment from Client
bool face_biometric::recvConfirmation(void *sock_fd){

   // socket
   //
    int sock = *(int*)sock_fd;
   
    // recv from client
    if(recv(sock, buffer_recv, sizeof(buffer_recv), 0 ) <= 0){
    
    
      cerr  << "Recv Failed from Client : " << endl;
      cout << "[ERROR]: " << strerror(errno) << endl;
      return false;
    
    }
   // recv from client
    string recv_client(buffer_recv);
    cout << "Acknowledgement: " << recv_client << endl;

  // return true
   return true;
}
//Destroy Object
face_biometric::~face_biometric(){

    //Destroy Char Recv
    delete buffer_recv;
    
    // Return Success
    fprintf(stdout, "Object Destroyed\n");
};
